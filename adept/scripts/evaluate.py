#!python
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
import json
import os
from collections import namedtuple
from glob import glob

import numpy as np
import torch
from absl import flags

from adept.containers import Evaluation
from adept.environments import SubProcEnvManager
from adept.environments.env_registry import EnvPluginRegistry
from adept.utils.logging import make_logger, print_ascii_logo, log_args
from adept.utils.script_helpers import make_agent, make_network, get_head_shapes
from adept.utils.util import dotdict

# hack to use argparse for SC2
FLAGS = flags.FLAGS
FLAGS(['local.py'])

Result = namedtuple('Result', ['epoch', 'mean', 'std_dev'])
SelectedModel = namedtuple('SelectedModel', ['epoch', 'model_id'])


def main(args, env_registry=EnvPluginRegistry()):
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
        train_args = dotdict(json.load(args_file))
    train_args.nb_env = args.nb_episode  # TODO make this line uneccessary

    # construct env
    env = SubProcEnvManager.from_args(
        train_args,
        seed=args.seed,
        nb_env=args.nb_episode,
        registry=env_registry
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network_head_shapes = get_head_shapes(env.action_space, train_args.agent)
    network = make_network(
        env.observation_space, network_head_shapes, train_args
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
            agent = make_agent(
                network, device, env.gpu_preprocessor,
                env_registry.lookup_engine(train_args.env), env.action_space,
                train_args
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
    import argparse

    parser = argparse.ArgumentParser(description='AdeptRL Evaluation Mode')
    parser.add_argument(
        '--log-id-dir', help='path to log dir (.../logs/<env-id>/<log-id>)'
    )
    parser.add_argument(
        '--nb-episode',
        type=int,
        default=30,
        help='number of episodes to evaluate on. (default: 30)'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=512,
        metavar='S',
        help='random seed (default: 512)'
    )
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='Which GPU to use (default: 0)'
    )
    args = parser.parse_args()
    main(args)
