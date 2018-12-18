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
from mpi4py import MPI as mpi
import torch
from absl import flags
from adept.containers import ImpalaHost, ImpalaWorker
from adept.environments import ParallelEnvManager
from adept.utils.script_helpers import make_agent, make_network, get_head_shapes, count_parameters
from adept.utils.logging import make_log_id_from_timestamp, make_logger, print_ascii_logo, log_args, write_args_file, \
    SimpleModelSaver
from tensorboardX import SummaryWriter
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
        log_id = make_log_id_from_timestamp(args.tag, args.mode_name, args.agent,
                                            args.network_vision + args.network_body,
                                            timestamp)
        log_id_dir = os.path.join(args.log_dir, args.env_id, log_id)
        os.makedirs(log_id_dir)
        saver = SimpleModelSaver(log_id_dir)
        print_ascii_logo()
    else:
        timestamp = None
    timestamp = comm.bcast(timestamp, root=0)

    if rank != 0:
        log_id = make_log_id_from_timestamp(args.tag, args.mode_name, args.agent,
                                            args.network_vision + args.network_body,
                                            timestamp)
        log_id_dir = os.path.join(args.log_dir, args.env_id, log_id)

    comm.Barrier()

    # construct env
    # unique seed per process
    seed = args.seed if rank == 0 else args.seed + args.nb_env * (rank - 1)
    # don't make a ton of envs if host
    if rank == 0:
        env_args = deepcopy(args)
        env_args.nb_env = 1
        env_args.seed = seed
        env = ParallelEnvManager.from_args(env_args)
    else:
        args.seed = seed
        env = ParallelEnvManager.from_args(args)

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

    # construct agent
    # host is always the first gpu, workers are distributed evenly across the rest
    if len(args.gpu_id) > 1:  # nargs is always a list
        if rank == 0:
            gpu_id = args.gpu_id[0]
        else:
            gpu_id = args.gpu_id[1:][(rank - 1) % len(args.gpu_id[1:])]
    else:
        gpu_id = args.gpu_id[-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn = True
    # disable cudnn for dynamic batches
    if rank == 0 and args.max_dynamic_batch > 0:
        cudnn = False

    torch.backends.cudnn.benchmark = cudnn
    agent = make_agent(network, device, env.gpu_preprocessor, env.engine, env.action_space, args)

    # workers
    if rank != 0:
        logger = make_logger('ImpalaWorker{}'.format(rank), os.path.join(log_id_dir, 'train_log{}.txt'.format(rank)))
        summary_writer = SummaryWriter(os.path.join(log_id_dir, str(rank)))
        container = ImpalaWorker(agent, env, args.nb_env, logger, summary_writer,
                                 use_local_buffers=args.use_local_buffers)

        # Run the container
        if args.profile:
            try:
                from pyinstrument import Profiler
            except:
                raise ImportError('You must install pyinstrument to use profiling.')
            profiler = Profiler()
            profiler.start()
            container.run()
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))
        else:
            container.run(initial_step_count)
        env.close()
    # host
    else:
        logger = make_logger('ImpalaHost', os.path.join(log_id_dir, 'train_log{}.txt'.format(rank)))
        summary_writer = SummaryWriter(os.path.join(log_id_dir, str(rank)))
        log_args(logger, args)
        write_args_file(log_id_dir, args)
        logger.info('Network Parameter Count: {}'.format(count_parameters(network)))

        # no need for the env anymore
        env.close()

        # Construct the optimizer
        def make_optimizer(params):
            opt = torch.optim.RMSprop(params, lr=args.learning_rate, eps=1e-5, alpha=0.99)
            if args.load_optimizer:
                opt.load_state_dict(
                    torch.load(
                        args.load_optimizer, map_location=lambda storage, loc: storage
                    )
                )
            return opt

        container = ImpalaHost(agent, comm, make_optimizer, summary_writer, args.summary_frequency, saver,
                               args.epoch_len, args.host_training_info_interval,
                               use_local_buffers=args.use_local_buffers)

        # Run the container
        if args.profile:
            try:
                from pyinstrument import Profiler
            except:
                raise ImportError('You must install pyinstrument to use profiling.')
            profiler = Profiler()
            profiler.start()
            if args.max_dynamic_batch > 0:
                container.run(args.max_dynamic_batch, args.max_queue_length, args.max_train_steps, dynamic=True,
                              min_dynamic_batch=args.min_dynamic_batch)
            else:
                container.run(args.num_rollouts_in_batch, args.max_queue_length, args.max_train_steps)
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))
        else:
            if args.max_dynamic_batch > 0:
                container.run(args.max_dynamic_batch, args.max_queue_length, args.max_train_steps, dynamic=True,
                              min_dynamic_batch=args.min_dynamic_batch)
            else:
                container.run(args.num_rollouts_in_batch, args.max_queue_length, args.max_train_steps)


if __name__ == '__main__':
    import argparse
    from adept.utils.script_helpers import add_base_args, parse_bool

    base_parser = argparse.ArgumentParser(description='AdeptRL IMPALA Mode')

    def add_args(parser):
        parser = parser.add_argument_group('IMPALA Mode Args')
        parser.add_argument('--gpu-id', type=int, nargs='+', default=[0],
                            help='Which GPU to use for training. The host will always be the first gpu, workers are distributed evenly across the rest (default: [0])')
        parser.add_argument(
            '--max-queue-length', type=int, default=(size - 1) * 2,
            help='Maximum rollout queue length. If above the max, workers will wait to append (default: (size - 1) * 2)'
        )
        parser.add_argument(
            '--num-rollouts-in-batch', type=int, default=(size - 1),
            help='The batch size in rollouts (so total batch is this number * nb_env * seq_len). '
                 + 'Not compatible with --dynamic-batch (default: (size - 1))'
        )
        parser.add_argument(
            '--max-dynamic-batch', type=int, default=0,
            help='When > 0 uses dynamic batching (disables cudnn and --num-rollouts-in-batch). '
                 + 'Limits the maximum rollouts in the batch to limit GPU memory usage. (default: 0 (False))'
        )
        parser.add_argument(
            '--min-dynamic-batch', type=int, default=0,
            help='Guarantees a minimum number of rollouts in the batch when using dynamic batching. (default: 0)'
        )
        parser.add_argument(
            '--host-training-info-interval', type=int, default=100,
            help='The number of training steps before the host writes an info summary. (default: 100)'
        )
        parser.add_argument(
            '--use-local-buffers', type=parse_bool, nargs='?', const=True, default=False,
            help='If true all workers use their local network buffers (for batch norm: mean & var are not shared) (default: False)'
        )

    add_base_args(base_parser, add_args)
    args = base_parser.parse_args()

    if args.debug:
        args.nb_env = 3
        args.log_dir = '/tmp/'

    args.mode_name = 'IMPALA'
    main(args)
