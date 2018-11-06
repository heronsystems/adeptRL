#!python
"""
Copyright (C) 2018 Heron Systems, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os
from copy import deepcopy
import torch
from absl import flags
from mpi4py import MPI as mpi
from tensorboardX import SummaryWriter

from adept.containers import ToweredHost, ToweredWorker
from adept.utils.logging import make_log_id_from_timestamp, make_logger, print_ascii_logo, log_args, write_args_file, \
    SimpleModelSaver
from adept.utils.script_helpers import make_agent, make_network, make_env, get_head_shapes, count_parameters
from datetime import datetime

# hack to use argparse for SC2
FLAGS = flags.FLAGS
FLAGS(['local.py'])

# mpi comm, rank, and size
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def main(args):
    # host needs to broadcast timestamp so all procs create the same log dir
    if rank == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_id = make_log_id_from_timestamp(
            args.tag, args.mode_name, args.agent,
            args.network_vision + args.network_body, timestamp
        )
        log_id_dir = os.path.join(args.log_dir, args.env_id, log_id)
        os.makedirs(log_id_dir)
        saver = SimpleModelSaver(log_id_dir)
        print_ascii_logo()
    else:
        timestamp = None
    timestamp = comm.bcast(timestamp, root=0)

    if rank != 0:
        log_id = make_log_id_from_timestamp(
            args.tag, args.mode_name, args.agent,
            args.network_vision + args.network_body, timestamp
        )
        log_id_dir = os.path.join(args.log_dir, args.env_id, log_id)

    comm.Barrier()

    # construct env
    # unique seed per process
    seed = args.seed if rank == 0 else args.seed + args.nb_env * (rank - 1)
    # don't make a ton of envs if host
    if rank == 0:
        env_args = deepcopy(args)
        env_args.nb_env = 1
        env = make_env(env_args, seed)
    else:
        env = make_env(args, seed)

    # construct network
    torch.manual_seed(args.seed)
    network_head_shapes = get_head_shapes(env.action_space, args.agent)
    network = make_network(env.observation_space, network_head_shapes, args)

    # possibly load network
    initial_step_count = 0
    if args.load_network:
        network.load_state_dict(
            torch.load(
                args.load_network, map_location=lambda storage, loc: storage
            )
        )
        # get step count from network file
        epoch_dir = os.path.split(args.load_network)[0]
        initial_step_count = int(os.path.split(epoch_dir)[-1])
        print('Reloaded network from {}'.format(args.load_network))

    # only sync network params if not loading
    else:
        if rank == 0:
            for v in network.parameters():
                comm.Bcast(v.detach().cpu().numpy(), root=0)
            print('Root variables synced')
        else:
            # can just use the numpy buffers
            variables = [v.detach().cpu().numpy() for v in network.parameters()]
            for v in variables:
                comm.Bcast(v, root=0)
            for shared_v, model_v in zip(variables, network.parameters()):
                model_v.data.copy_(torch.from_numpy(shared_v), non_blocking=True)
            print('{} variables synced'.format(rank))

    # host is rank 0
    if rank != 0:
        # construct logger
        logger = make_logger(
            'ToweredWorker{}'.format(rank),
            os.path.join(log_id_dir, 'train_log_rank{}.txt'.format(rank))
        )
        summary_writer = SummaryWriter(os.path.join(log_id_dir, 'rank{}'.format(rank)))

        # construct agent
        # distribute evenly across gpus
        if isinstance(args.gpu_id, list):
            gpu_id = args.gpu_id[(rank - 1) % len(args.gpu_id)]
        else:
            gpu_id = args.gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        agent = make_agent(network, device, env.gpu_preprocessor, env.engine, env.action_space, args)

        # construct container
        container = ToweredWorker(
            agent, env, args.nb_env, logger, summary_writer, args.summary_frequency
        )

        # Run the container
        try:
            # distribute global step count evenly to workers
            container.run(initial_count=initial_step_count // (size - 1))
        finally:
            env.close()
    # host
    else:
        logger = make_logger(
            'ToweredHost',
            os.path.join(log_id_dir, 'train_log_rank{}.txt'.format(rank))
        )
        log_args(logger, args)
        write_args_file(log_id_dir, args)
        logger.info('Network Parameter Count: {}'.format(count_parameters(network)))

        # no need for the env anymore
        env.close()

        # Construct the optimizer
        def make_optimizer(params):
            opt = torch.optim.RMSprop(
                params, lr=args.learning_rate, eps=1e-5, alpha=0.99
            )
            if args.load_optimizer:
                opt.load_state_dict(
                    torch.load(
                        args.load_optimizer, map_location=lambda storage, loc: storage
                    )
                )
            return opt

        container = ToweredHost(
            comm, args.num_grads_to_drop, network, make_optimizer, saver,
            args.epoch_len, logger
        )

        # Run the container
        if args.profile:
            try:
                from pyinstrument import Profiler
            except:
                raise ImportError('You must install pyinstrument to use profiling.')
            profiler = Profiler()
            profiler.start()
            container.run(10e3)
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))
        else:
            container.run(args.max_train_steps, initial_step_count=initial_step_count)


if __name__ == '__main__':
    import argparse
    from adept.utils.script_helpers import add_base_args, parse_bool

    base_parser = argparse.ArgumentParser(description='AdeptRL Towered Mode')

    def add_args(parser):
        parser = parser.add_argument_group('Towered Mode Args')
        parser.add_argument(
            '--gpu-id',
            type=int,
            nargs='+',
            default=0,
            help='Which GPU(s) to use for training (default: 0)'
        )
        parser.add_argument(
            '--num-grads-to-drop',
            type=int,
            default=0,
            help=
            'The number of gradient receives to drop in a round. https://arxiv.org/abs/1604.00981 recommends dropping'
            ' 10 percent of gradients for maximum speed (default: 0)'
        )

    add_base_args(base_parser, add_args)
    args = base_parser.parse_args()

    if args.debug:
        args.nb_env = 3
        args.log_dir = '/tmp/'
    args.mode_name = 'Towered'

    main(args)
