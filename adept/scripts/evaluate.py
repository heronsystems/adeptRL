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
Evaluate
Evaluates an agent after training. Computes N-episode average reward by
loading a saved model from each epoch. N-episode averages are computed by
running N environments in parallel.
Usage:
    evaluate (--log-id-dir <path>) [options]
    evaluate (-h | --help)
Required:
    --log-id-dir <path>     Path to train logs (.../logs/<env-id>/<log-id>)
Options:
    --gpu-id <int>          CUDA device ID of GPU [default: 0]
    --nb-episode <int>      Number of episodes to average [default: 30]
    --seed <int>            Seed for random variables [default: 512]
    --custom-network <str>  Name of custom network class
"""
import json
import os
from collections import namedtuple
from glob import glob

import numpy as np
import torch
from absl import flags

from adept.agents.agent_registry import AgentRegistry
from adept.containers import Evaluation
from adept.environments import SubProcEnvManager
from adept.environments.env_registry import EnvRegistry
from adept.networks.modular_network import ModularNetwork
from adept.networks.network_registry import NetworkRegistry
from adept.utils.logging import make_logger, print_ascii_logo, log_args
from adept.utils.util import DotDict
from adept.utils.script_helpers import parse_path

# hack to use argparse for SC2
FLAGS = flags.FLAGS
FLAGS(['local.py'])


def parse_args():
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.strip('--').replace('-', '_'): v for k, v in args.items()}
    del args['h']
    del args['help']
    args = DotDict(args)
    args.log_id_dir = parse_path(args.log_id_dir)
    args.gpu_id = int(args.gpu_id)
    args.nb_episode = int(args.nb_episode)
    args.seed = int(args.seed)
    return args


Result = namedtuple('Result', ['epoch', 'mean', 'std_dev'])
SelectedModel = namedtuple('SelectedModel', ['epoch', 'model_id'])


def main(
    args,
    agent_registry=AgentRegistry(),
    env_registry=EnvRegistry(),
    net_registry=NetworkRegistry()
):
    """
    Run an evaluation.
    :param args: Dict[str, Any]
    :param agent_registry: AgentRegistry
    :param env_registry: EnvRegistry
    :param net_registry: NetworkRegistry
    :return:
    """
    args = DotDict(args)

    print_ascii_logo()
    logger = make_logger(
        'Eval', os.path.join(args.log_id_dir, 'evaluation_log.txt')
    )
    log_args(logger, args)

    epoch_ids = sorted(
        [
            int(dir) for dir in os.listdir(args.log_id_dir)
            if os.path.isdir(os.path.join(args.log_id_dir, dir)) and
            ('rank' not in dir)
        ]
    )

    with open(os.path.join(args.log_id_dir, 'args.json'), 'r') as args_file:
        train_args = DotDict(json.load(args_file))

    # construct env
    env = SubProcEnvManager.from_args(
        train_args,
        seed=args.seed,
        nb_env=args.nb_episode,
        registry=env_registry
    )
    device = torch.device(
        "cuda:{}".format(args.gpu_id)
        if (torch.cuda.is_available() and args.gpu_id >= 0)
        else "cpu"
    )
    output_space = agent_registry.lookup_output_space(
        train_args.agent, env.action_space
    )
    if args.custom_network:
        network = net_registry.lookup_custom_net(
            train_args.custom_network
        ).from_args(
            train_args,
            env.observation_space,
            output_space,
            net_registry
        )
    else:
        network = ModularNetwork.from_args(
            train_args,
            env.observation_space,
            output_space,
            net_registry
        )

    results = []
    selected_models = []
    for epoch_id in epoch_ids:
        network_path = os.path.join(
            args.log_id_dir, str(epoch_id), 'model*.pth'
        )
        network_files = glob(network_path)

        best_mean = -float('inf')
        best_std_dev = 0.
        selected_model = None
        for network_file in network_files:
            # load new network
            network.load_state_dict(
                torch.load(
                    network_file,
                    map_location=lambda storage, loc: storage
                )
            )

            # construct agent
            agent = agent_registry.lookup_agent(train_args.agent).from_args(
                train_args,
                network,
                device,
                env_registry.lookup_reward_normalizer(train_args.env),
                env.gpu_preprocessor,
                env.engine,
                env.action_space,
                nb_env=args.nb_episode
            )
            # container
            container = Evaluation(agent, device, env)

            # Run the container
            mean_reward, std_dev = container.run()

            if mean_reward >= best_mean:
                best_mean = mean_reward
                best_std_dev = std_dev
                selected_model = os.path.split(network_file)[-1]

        result = Result(epoch_id, best_mean, best_std_dev)
        selected_model = SelectedModel(epoch_id, selected_model)
        logger.info(str(result) + ' ' + str(selected_model))
        results.append(np.asarray(result))
        selected_models.append(selected_model)

    # save results
    results = np.stack(results)
    np.savetxt(
        os.path.join(args.log_id_dir, 'eval.csv'),
        results,
        delimiter=',',
        fmt=['%d', '%.3f', '%.3f']
    )

    # save selected models
    with open(os.path.join(args.log_id_dir, 'selected_models.txt'), 'w') as f:
        for sm in selected_models:
            f.write(str(sm) + '\n')

    env.close()


if __name__ == '__main__':
    main(parse_args())
