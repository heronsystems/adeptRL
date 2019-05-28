#!/usr/bin/env python
# Copyright (C) 2018 Heron Systems, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
             __           __
  ____ _____/ /__  ____  / /_
 / __ `/ __  / _ \/ __ \/ __/
/ /_/ / /_/ /  __/ /_/ / /_
\__,_/\__,_/\___/ .___/\__/
               /_/

IMPALA Mode

Train an agent with multiple GPUs. https://arxiv.org/abs/1802.01561.

Usage:
    impala [options]
    impala (-h | --help)

Agent Options:
    --agent <str>             Name of agent class [default: ActorCriticVtrace]

Environment Options:
    --env <str>               Environment name [default: PongNoFrameskip-v4]

Script Options:
    --gpu-ids <ids>            Comma-separated CUDA IDs [default: 0,1]
    --nb-env <int>             Number of environments per Tower [default: 32]
    --seed <int>               Seed for random variables [default: 0]
    --nb-step <int>            Number of steps to train for [default: 10e6]
    --max-queue-len <int>      Maximum rollout queue length
    --nb-rollout-batch <int>   Number of rollouts in a batch
    --max-dynamic-batch <int>  Limit max rollouts in batch [default: 0]
    --min-dynamic-batch <int>  Limit min rollouts in batch [default: 0]
    --info-interval <int>      Write INFO every <int> training frames [default: 100]
    --use-local-buffers        Workers use local buffers (ie. mean/stddev)
    --load-network <path>      Path to network file
    --load-optim <path>        Path to optimizer file
    -y, --use-defaults         Skip prompts, use defaults

Network Options:
    --net1d <str>           Network to use for 1d input [default: Identity1D]
    --net2d <str>           Network to use for 2d input [default: Identity2D]
    --net3d <str>           Network to use for 3d input [default: FourConv]
    --net4d <str>           Network to use for 4d input [default: Identity4D]
    --netbody <str>         Network to use on merged inputs [default: LSTM]
    --head1d <str>          Network to use for 1d output [default: Identity1D]
    --head2d <str>          Network to use for 2d output [default: Identity2D]
    --head3d <str>          Network to use for 3d output [default: Identity3D]
    --head4d <str>          Network to use for 4d output [default: Identity4D]
    --custom-network        Name of custom network class

Optimizer Options:
    --lr <float>               Learning rate [default: 0.0007]

Logging Options:
    --tag <str>                Name your run [default: None]
    --logdir <path>            Path to logging directory [default: /tmp/adept_logs/]
    --epoch-len <int>          Save a model every <int> frames [default: 1e6]
    --summary-freq <int>       Tensorboard summary frequency [default: 10]

Troubleshooting Options:
    --profile                 Profile this script
"""
import os
from datetime import datetime

import torch
from absl import flags
from mpi4py import MPI as mpi
from tensorboardX import SummaryWriter

from adept.agents.agent_registry import AgentRegistry
from adept.containers import ImpalaHost, ImpalaWorker
from adept.environments import SubProcEnvManager, EnvMetaData
from adept.environments.env_registry import EnvRegistry
from adept.networks.modular_network import ModularNetwork
from adept.networks.network_registry import NetworkRegistry
from adept.utils.logging import (
    make_log_id, make_logger, print_ascii_logo, log_args,
    write_args_file, SimpleModelSaver
)
from adept.utils.script_helpers import (
    count_parameters,
    parse_none,
    parse_list_str,
    parse_path
)
from adept.utils.util import DotDict

# hack to use argparse for SC2
FLAGS = flags.FLAGS
FLAGS(['local.py'])

# mpi comm, rank, and size
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_args():
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.strip('--').replace('-', '_'): v for k, v in args.items()}
    del args['h']
    del args['help']
    args = DotDict(args)

    # Network Options

    # Container Options
    args.logdir = parse_path(args.logdir)
    args.gpu_ids = parse_list_str(args.gpu_ids, int)
    args.nb_env = int(args.nb_env)
    args.seed = int(args.seed)
    args.nb_step = int(float(args.nb_step))
    args.max_queue_len = int(args.max_queue_len) \
        if args.max_queue_len \
        else (size - 1) * 2
    args.nb_rollout_batch = int(args.nb_rollout_batch) \
        if args.nb_rollout_batch \
        else size - 1
    args.nb_proc = size
    args.max_dynamic_batch = int(args.max_dynamic_batch)
    args.min_dynamic_batch = int(args.min_dynamic_batch)
    args.info_interval = int(args.info_interval)

    # Logging Options
    args.tag = parse_none(args.tag)
    args.summary_freq = int(args.summary_freq)
    args.lr = float(args.lr)
    args.epoch_len = int(float(args.epoch_len))

    # Troubleshooting Options
    args.profile = bool(args.profile)
    return args


def main(
    args,
    agent_registry=AgentRegistry(),
    env_registry=EnvRegistry(),
    net_registry=NetworkRegistry()
):
    """
    Run local training.

    :param args: Dict[str, Any]
    :param agent_registry: AgentRegistry
    :param env_registry: EnvRegistry
    :param net_registry: NetworkRegistry
    :return:
    """
    # host needs to broadcast timestamp so all procs create the same log dir
    if rank == 0:
        if args.use_defaults:
            agent_args = agent_registry.lookup_agent(args.agent).args
            env_args = env_registry.lookup_env_class(args.env).args
            if args.custom_network:
                net_args = net_registry.lookup_custom_net(args.net).args
            else:
                net_args = net_registry.lookup_modular_args(args)
        else:
            agent_args = agent_registry.lookup_agent(args.agent).prompt()
            env_args = env_registry.lookup_env_class(args.env).prompt()
            if args.custom_network:
                net_args = net_registry.lookup_custom_net(args.net).prompt()
            else:
                net_args = net_registry.prompt_modular_args(args)
        args = DotDict({**args, **agent_args, **env_args, **net_args})

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_id = make_log_id(
            args.tag, 'Impala', args.agent,
            args.net3d + args.netbody, timestamp
        )
        log_id_dir = os.path.join(args.logdir, args.env, log_id)
        os.makedirs(log_id_dir)
        saver = SimpleModelSaver(log_id_dir)
        args = dict(args)
        print_ascii_logo()
    else:
        timestamp = None
        args = None
    timestamp = comm.bcast(timestamp, root=0)
    args = comm.bcast(args, root=0)
    args = DotDict(args)

    if rank != 0:
        log_id = make_log_id(
            args.tag, 'Impala', args.agent,
            args.net3d + args.netbody, timestamp
        )
        log_id_dir = os.path.join(args.logdir, args.env, log_id)

    comm.Barrier()

    # construct env
    # unique seed per process
    seed = args.seed if rank == 0 else args.seed + args.nb_env * (rank - 1)
    if rank == 0:
        env = EnvMetaData.from_args(args, env_registry)
    else:
        env = SubProcEnvManager.from_args(
            args, seed=seed, registry=env_registry
        )

    # construct network
    torch.manual_seed(args.seed)
    output_space = agent_registry.lookup_output_space(
        args.agent, env.action_space, args
    )
    if args.custom_network:
        network = net_registry.lookup_custom_net(args.custom_network).from_args(
            args,
            env.observation_space,
            output_space,
            net_registry
        )
    else:
        network = ModularNetwork.from_args(
            args,
            env.observation_space,
            output_space,
            net_registry
        )

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
                model_v.data.copy_(
                    torch.from_numpy(shared_v), non_blocking=True
                )
            print('{} variables synced'.format(rank))

    # construct agent
    # host is always the first gpu,
    # workers are distributed evenly across the rest
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    device = torch.device(
        "cuda:{}".format(gpu_id)
        if (torch.cuda.is_available() and gpu_id >= 0)
        else "cpu"
    )
    cudnn = True
    # disable cudnn for dynamic batches
    if rank == 0 and args.max_dynamic_batch > 0:
        cudnn = False

    torch.backends.cudnn.benchmark = cudnn
    agent = agent_registry.lookup_agent(args.agent).from_args(
        args,
        network,
        device,
        env_registry.lookup_reward_normalizer(args.env),
        env.gpu_preprocessor,
        env_registry.lookup_engine(args.env),
        env.action_space
    )

    # workers
    if rank != 0:
        logger = make_logger(
            'ImpalaWorker{}'.format(rank),
            os.path.join(log_id_dir, 'train_log{}.txt'.format(rank))
        )
        summary_writer = SummaryWriter(os.path.join(log_id_dir, str(rank)))
        container = ImpalaWorker(
            agent,
            env,
            args.nb_env,
            logger,
            summary_writer,
            rank,
            use_local_buffers=args.use_local_buffers
        )

        # Run the container
        if args.profile:
            try:
                from pyinstrument import Profiler
            except:
                raise ImportError(
                    'You must install pyinstrument to use profiling.'
                )
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
        logger = make_logger(
            'ImpalaHost',
            os.path.join(log_id_dir, 'train_log{}.txt'.format(rank))
        )
        summary_writer = SummaryWriter(os.path.join(log_id_dir, str(rank)))
        log_args(logger, args)
        write_args_file(log_id_dir, args)
        logger.info(
            'Network Parameter Count: {}'.format(count_parameters(network))
        )

        # Construct the optimizer
        def make_optimizer(params):
            opt = torch.optim.RMSprop(
                params, lr=args.lr, eps=1e-5, alpha=0.99
            )
            if args.load_optim:
                opt.load_state_dict(
                    torch.load(
                        args.load_optim,
                        map_location=lambda storage, loc: storage
                    )
                )
            return opt

        container = ImpalaHost(
            agent,
            comm,
            make_optimizer,
            summary_writer,
            args.summary_freq,
            saver,
            args.epoch_len,
            args.info_interval,
            rank,
            use_local_buffers=args.use_local_buffers
        )

        # Run the container
        if args.profile:
            try:
                from pyinstrument import Profiler
            except:
                raise ImportError(
                    'You must install pyinstrument to use profiling.'
                )
            profiler = Profiler()
            profiler.start()
            if args.max_dynamic_batch > 0:
                container.run(
                    args.max_dynamic_batch,
                    args.max_queue_len,
                    args.nb_step,
                    dynamic=True,
                    min_dynamic_batch=args.min_dynamic_batch
                )
            else:
                container.run(
                    args.nb_rollout_batch, args.max_queue_len,
                    args.nb_step
                )
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))
        else:
            if args.max_dynamic_batch > 0:
                container.run(
                    args.max_dynamic_batch,
                    args.max_queue_len,
                    args.nb_step,
                    dynamic=True,
                    min_dynamic_batch=args.min_dynamic_batch
                )
            else:
                container.run(
                    args.nb_rollout_batch, args.max_queue_len,
                    args.nb_step
                )


if __name__ == '__main__':
    main(parse_args())
