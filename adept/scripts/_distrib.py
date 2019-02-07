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
Distributed worker script. Called from launcher (distrib.py).
"""
import argparse
import json
import os

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

from adept.agents.agent_registry import AgentRegistry
from adept.containers.distrib import DistribHost, DistribWorker
from adept.environments.env_registry import EnvRegistry
from adept.environments.managers.subproc_env_manager import SubProcEnvManager
from adept.networks.modular_network import ModularNetwork
from adept.networks.network_registry import NetworkRegistry
from adept.utils.logging import make_logger, SimpleModelSaver
from adept.utils.script_helpers import (
    count_parameters, LogDirHelper
)
from adept.utils.util import DotDict

MODE = 'Distrib'
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
GLOBAL_RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])

# hack to use bypass pysc2 flags
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['local.py'])

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-id-dir', required=True)
    parser.add_argument(
        '--resume', type=str2bool, nargs='?', const=True, default=False
    )
    parser.add_argument('--load-network', default=None)
    parser.add_argument('--load-optim', default=None)
    parser.add_argument('--initial-step-count', type=int, default=0)
    args = parser.parse_args()
    return args


def main(
    local_args,
    agent_registry=AgentRegistry(),
    env_registry=EnvRegistry(),
    net_registry=NetworkRegistry()
):
    """
    Run distributed training.

    :param args: Dict[str, Any]
    :param agent_registry: AgentRegistry
    :param env_registry: EnvRegistry
    :param net_registry: NetworkRegistry
    :return:
    """
    log_id_dir = local_args.log_id_dir
    initial_step_count = local_args.initial_step_count

    logger = make_logger(
        MODE + str(LOCAL_RANK),
        os.path.join(log_id_dir, 'train_log{}.txt'.format(GLOBAL_RANK))
    )

    helper = LogDirHelper(log_id_dir)
    with open(helper.args_file_path(), 'r') as args_file:
        args = DotDict(json.load(args_file))

    if local_args.resume:
        args = DotDict({**args, **vars(local_args)})

    torch.backends.cudnn.benchmark = True

    dist.init_process_group(
        backend='nccl',
        init_method='file:///tmp/adept_init',
        world_size=WORLD_SIZE,
        rank=LOCAL_RANK
    )

    logger.info('Rank {} initialized.'.format(GLOBAL_RANK))
    seed = args.seed \
        if GLOBAL_RANK == 0 \
        else args.seed + args.nb_env * GLOBAL_RANK
    logger.info('Using {} for rank {} seed.'.format(seed, GLOBAL_RANK))
    env = SubProcEnvManager.from_args(
        args,
        seed=seed,
        registry=env_registry
    )

    # Construct network
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
    if args.load_network:
        network.load_state_dict(
            torch.load(
                args.load_network, map_location=lambda storage, loc: storage
            )
        )
        # get step count from network file
        print('Reloaded network from {}'.format(args.load_network))
    if LOCAL_RANK == 0:
        logger.info('Network Parameter Count: {}'.format(
            count_parameters(network))
        )

    device = torch.device("cuda:{}".format(LOCAL_RANK))
    agent = agent_registry.lookup_agent(args.agent).from_args(
        args,
        network,
        device,
        env_registry.lookup_reward_normalizer(args.env),
        env.gpu_preprocessor,
        env_registry.lookup_policy(env.engine)(env.action_space)
    )

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
            logger.info("Reloaded optimizer from {}".format(args.load_optim))
        return opt
    if LOCAL_RANK == 0:
        summary_writer = SummaryWriter(
            os.path.join(log_id_dir, 'rank{}'.format(GLOBAL_RANK))
        )
        container = DistribHost(
            agent, env, make_optimizer, args.epoch_len, args.nb_env, logger,
            summary_writer, args.summary_freq, SimpleModelSaver(log_id_dir),
            GLOBAL_RANK, WORLD_SIZE
        )
    else:
        container = DistribWorker(
            agent, env, make_optimizer, args.epoch_len, args.nb_env, logger,
            GLOBAL_RANK, WORLD_SIZE
        )

    container.run(args.nb_step, initial_count=initial_step_count)
    env.close()

    if args.eval and GLOBAL_RANK == 0:
        import subprocess
        exit(subprocess.call([
            'python',
            '-m',
            'adept.scripts.evaluate',
            '--log-id-dir',
            log_id_dir,
            '--gpu-id',
            str(0),
            '--nb-episode',
            str(32)  # TODO
        ], env=os.environ))


if __name__ == '__main__':
    main(parse_args())
