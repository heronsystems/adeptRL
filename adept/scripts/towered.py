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

Towered Mode

Train an agent with multiple GPUs.

Usage:
    towered [options]
    towered (-h | --help)

Agent Options:
    --agent <str>            Name of agent class [default: ActorCritic]

Environment Options:
    --env <str>              Environment name [default: PongNoFrameskip-v4]

Script Options:
    --gpu-ids <ids>          Comma-separated CUDA IDs [default: 0,0]
    --nb-env <int>           Number of environments per Tower [default: 32]
    --seed <int>             Seed for random variables [default: 0]
    --nb-train-frame <int>   Number of frames to train on [default: 10e6]
    --load-network <path>    Path to network file
    --load-optim <path>      Path to optimizer file
    --nb-grad-drop <int>     Number of gradients to drop per round [default: 0]
    -y, --use-defaults       Skip prompts, use defaults

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
    --lr <float>             Learning rate [default: 0.0007]

Logging Options:
    --tag <str>              Name your run [default: None]
    --logdir <path>          Path to logging directory [default: /tmp/adept_logs/]
    --epoch-len <int>        Save a model every <int> frames [default: 1e6]
    --summary-freq <int>     Tensorboard summary frequency [default: 10]

Troubleshooting Options:
    --profile                Profile this script
"""
import os
import torch
from absl import flags
from mpi4py import MPI as mpi
from tensorboardX import SummaryWriter

from adept.agents.agent_registry import AgentRegistry
from adept.containers import ToweredHost, ToweredWorker
from adept.environments import SubProcEnvManager, EnvMetaData
from adept.environments.env_registry import EnvRegistry
from adept.networks.modular_network import ModularNetwork
from adept.networks.network_registry import NetworkRegistry
from adept.utils.logging import (
    make_log_id, make_logger, print_ascii_logo,
    log_args, write_args_file, SimpleModelSaver
)
from adept.utils.script_helpers import (
    count_parameters, parse_list_str, parse_none
)
from adept.utils.util import DotDict
from datetime import datetime

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
    args.gpu_ids = parse_list_str(args.gpu_ids, int)
    args.nb_env = int(args.nb_env)
    args.seed = int(args.seed)
    args.nb_train_frame = int(float(args.nb_train_frame))
    args.nb_grad_drop = int(args.nb_grad_drop)
    args.tag = parse_none(args.tag)
    args.summary_freq = int(args.summary_freq)
    args.lr = float(args.lr)
    args.epoch_len = int(float(args.epoch_len))
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
        args = DotDict(args)
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
            args.tag, 'Towered', args.agent,
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
            args.tag, 'Towered', args.agent,
            args.net3d + args.netbody, timestamp
        )
        log_id_dir = os.path.join(args.logdir, args.env, log_id)

    comm.Barrier()

    # construct env
    # unique seed per manager
    seed = args.seed if rank == 0 else args.seed + args.nb_env * (rank - 1)
    # don't make a ton of envs if host
    if rank == 0:
        env = EnvMetaData.from_args(args, env_registry)
    else:
        env = SubProcEnvManager.from_args(
            args, seed=seed, registry=env_registry
        )

    # construct network
    torch.manual_seed(args.seed)
    output_space = agent_registry.lookup_output_space(
        args.agent, env.action_space
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

    # host is rank 0
    if rank != 0:
        # construct logger
        logger = make_logger(
            'ToweredWorker{}'.format(rank),
            os.path.join(log_id_dir, 'train_log_rank{}.txt'.format(rank))
        )
        summary_writer = SummaryWriter(
            os.path.join(log_id_dir, 'rank{}'.format(rank))
        )

        # construct agent
        # distribute evenly across gpus
        gpu_id = args.gpu_ids[(rank - 1) % len(args.gpu_ids)]
        device = torch.device(
            "cuda:{}".format(gpu_id)
            if (torch.cuda.is_available() and gpu_id >= 0)
            else "cpu"
        )
        torch.backends.cudnn.benchmark = True
        agent = agent_registry.lookup_agent(args.agent).from_args(
            args,
            network,
            device,
            env_registry.lookup_reward_normalizer(args.env),
            env.gpu_preprocessor,
            env.engine,
            env.action_space
        )

        # construct container
        container = ToweredWorker(
            agent, env, args.nb_env, logger, summary_writer,
            args.summary_freq
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

        container = ToweredHost(
            comm, args.nb_grad_drop, network, make_optimizer, saver,
            args.epoch_len, logger
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
            container.run(10e3)
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))
        else:
            container.run(
                args.nb_train_frame, initial_step_count=initial_step_count
            )


if __name__ == '__main__':
    main(parse_args())
