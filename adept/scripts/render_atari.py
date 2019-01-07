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
import json
import os

import torch

from adept.containers import AtariRenderer
from adept.environments import SimpleEnvManager
from adept.environments.env_registry import EnvPluginRegistry, Engines
from adept.utils.logging import print_ascii_logo
from adept.utils.script_helpers import make_agent, make_network, get_head_shapes
from adept.utils.util import DotDict


def main(args, env_registry=EnvPluginRegistry()):
    # construct logging objects
    print_ascii_logo()
    print('Rendering... Press Ctrl+C to stop.')

    with open(args.args_file, 'r') as args_file:
        train_args = DotDict(json.load(args_file))

    engine = env_registry.lookup_engine(train_args.env)
    assert engine == Engines.GYM, "render_atari.py is only for Atari."

    train_args.nb_env = 1
    env = SimpleEnvManager.from_args(
        train_args, seed=args.seed, nb_env=1, registry=env_registry
    )

    # construct network
    network_head_shapes = get_head_shapes(env.action_space, train_args.agent)
    network = make_network(
        env.observation_space, network_head_shapes, train_args
    )
    network.load_state_dict(
        torch.load(
            args.network_file, map_location=lambda storage, loc: storage
        )
    )

    # create an agent (add act_eval method)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = make_agent(
        network, device, env.gpu_preprocessor, env.engine, env.action_space,
        train_args
    )

    # create a rendering container
    renderer = AtariRenderer(agent, device, env)
    try:
        renderer.run()
    finally:
        env.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='AdeptRL Render Mode')
    parser.add_argument(
        '--network-file',
        help='path to args file (.../logs/<env-id>/<log-id>/<epoch>/model.pth)'
    )
    parser.add_argument(
        '--args-file',
        help='path to args file (.../logs/<env-id>/<log-id>/args.json)'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=32,
        metavar='S',
        help='random seed (default: 32)'
    )
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='Which GPU to use (default: 0)'
    )
    args = parser.parse_args()
    main(args)
