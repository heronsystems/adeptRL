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

Render Atari

Renders an agent interacting with an Atari environment.

Usage:
    render_atari (--log-id-dir <path> --epoch <int>) [options]
    render_atari (-h | --help)

Required:
    --log-id-dir <path>     Path to train logs (.../logs/<env-id>/<log-id>)
    --epoch <int>           Epoch number to load

Options:
    --gpu-id <int>          CUDA device ID of GPU [default: 0]
    --seed <int>            Seed for random variables [default: 512]
"""
import json

import torch

from adept.agents.agent_registry import AgentRegistry
from adept.containers import AtariRenderer
from adept.environments import SimpleEnvManager
from adept.environments.env_registry import EnvModuleRegistry, Engines
from adept.utils.logging import print_ascii_logo
from adept.utils.script_helpers import make_network, LogDirHelper
from adept.utils.util import DotDict


def parse_args():
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.strip('--').replace('-', '_'): v for k, v in args.items()}
    del args['h']
    del args['help']
    args = DotDict(args)
    args.epoch = int(float(args.epoch))
    args.gpu_id = int(args.gpu_id)
    args.seed = int(args.seed)
    return args


def main(
    args,
    agent_registry=AgentRegistry(),
    env_registry=EnvModuleRegistry()
):
    """
    Run an evaluation.

    :param args: Dict[str, Any]
    :param agent_registry: AgentRegistry
    :param env_registry: EnvModuleRegistry
    :return:
    """
    # construct logging objects
    print_ascii_logo()
    print('Rendering... Press Ctrl+C to stop.')

    log_dir_helper = LogDirHelper(args.log_id_dir)

    with open(log_dir_helper.args_file_path(), 'r') as args_file:
        train_args = DotDict(json.load(args_file))

    engine = env_registry.lookup_engine(train_args.env)
    assert engine == Engines.GYM, "render_atari.py is only for Atari."

    env = SimpleEnvManager.from_args(
        train_args, seed=args.seed, nb_env=1, registry=env_registry
    )

    # construct network
    network = make_network(
        env.observation_space,
        agent_registry.lookup_output_space(train_args.agent, env.action_space),
        train_args
    )
    network.load_state_dict(
        torch.load(
            log_dir_helper.network_path_at_epoch(args.epoch),
            map_location=lambda storage, loc: storage
        )
    )

    # create an agent (add act_eval method)
    device = torch.device(
        "cuda:{}".format(args.gpu_id)
        if (torch.cuda.is_available() and args.gpu_id >= 0)
        else "cpu"
    )
    agent = agent_registry.lookup_agent(train_args.agent).from_args(
        train_args,
        network,
        device,
        env_registry.lookup_reward_normalizer(train_args.env),
        env.gpu_preprocessor,
        env.engine,
        env.action_space,
        nb_env=1
    )

    # create a rendering container
    renderer = AtariRenderer(agent, device, env)
    try:
        renderer.run()
    finally:
        env.close()


if __name__ == '__main__':
    main(parse_args())
