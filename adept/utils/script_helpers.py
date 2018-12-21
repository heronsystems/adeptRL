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
from argparse import ArgumentParser  # for type hinting

from adept.agents import AGENTS
from adept.environments import SubProcEnvManager, SimpleEnvManager, SC2_ENVS
from adept.environments._env import reward_normalizer_by_env_id
from adept.registries.environment import Engines
from adept.networks import VISION_NETWORKS, DISCRETE_NETWORKS, NETWORK_BODIES
from adept.networks._base import NetworkTrunk, ModularNetwork, NetworkHead
from adept.utils.util import parse_bool

try:
    from adept.environments.deepmind_sc2 import make_sc2_env, SC2AgentOverrides
except ImportError:
    print('SC2 Environment not detected')


def make_env(args, seed, subprocess=True, render=False):
    if args.env_id in SC2_ENVS:
        envs = sc2_from_args(args, seed, subprocess, render)
    else:
        envs = atari_from_args(args, seed, subprocess)
    return envs


def sc2_from_args(args, seed, subprocess=True, render=False):
    if subprocess:
        return SubProcEnvManager([make_sc2_env(args.env_id, seed + i) for i in range(args.nb_env)], Engines.SC2)
    else:
        return SimpleEnvManager([make_sc2_env(args.env_id, seed + i, render=render) for i in range(args.nb_env)], Engines.SC2)


def atari_from_args(args, seed, subprocess=True):
    do_frame_stack = 'Linear' in args.network_body

    env_wrapper_class = SubProcEnvManager if subprocess else SimpleEnvManager
    envs = env_wrapper_class(
        [
            make_atari_env(
                args.env_id,
                args.skip_rate,
                args.max_episode_length,
                do_frame_stack,
                seed + i
            ) for i in range(args.nb_env)
        ], Engines.GYM
    )
    return envs


def make_network(
    observation_space,
    network_head_shapes,
    args,
    chw_networks=VISION_NETWORKS,
    c_networks=DISCRETE_NETWORKS,
    network_bodies=NETWORK_BODIES,
    embedding_size=512
):
    pathways_by_name = {}
    nbr = observation_space.names_by_rank
    ebn = observation_space.entries_by_name
    for rank, names in nbr.items():
        for name in names:
            if rank == 1:
                pathways_by_name[name] = c_networks[args.network_discrete].from_args(ebn[name].shape, args)
            elif rank == 2:
                raise NotImplementedError('Rank 2 inputs not implemented')
            elif rank == 3:
                pathways_by_name[name] = chw_networks[args.network_vision].from_args(ebn[name].shape, args)
            elif rank == 4:
                raise NotImplementedError('Rank 4 inputs not implemented')
            else:
                raise NotImplementedError('Rank {} inputs not implemented'.format(rank))

    trunk = NetworkTrunk(pathways_by_name)
    body = network_bodies[args.network_body].from_args(trunk.nb_output_channel, embedding_size, args)
    head = NetworkHead(body.nb_output_channel, network_head_shapes)
    network = ModularNetwork(trunk, body, head)
    return network


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_agent(network, device, gpu_preprocessor, engine, action_space, args):
    Agent = AGENTS[args.agent]
    reward_normalizer = reward_normalizer_by_env_id(args.env_id)
    return Agent.from_args(network, device, reward_normalizer, gpu_preprocessor, engine, action_space, args)


def get_head_shapes(action_space, agent_name):
    Agent = AGENTS[agent_name]
    return Agent.output_shape(action_space)

def _add_common_agent_args(parser: ArgumentParser):
    parser.add_argument(
        '-al', '--learning-rate', type=float, default=7e-4,
        help='learning rate (default: 7e-4)'
    )
    parser.add_argument(
        '-ad', '--discount', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)'
    )

def _add_agent_args(subparser: ArgumentParser):
    agent_parsers = []
    for agent_name, agent_class in AGENTS.items():
        parser_agent = subparser.add_parser(agent_name)
        agent_group = parser_agent.add_argument_group('Agent Args')
        _add_common_agent_args(agent_group)
        agent_class.add_args(agent_group)
        agent_parsers.append(parser_agent)
    return agent_parsers

def _add_network_args(parser: ArgumentParser):
    subparser = parser.add_argument_group('Network Args')
    subparser.add_argument(
        '-nv', '--network-vision', default='FourConv'
    )
    subparser.add_argument(
        '-nd', '--network-discrete', default='Identity'
    )
    subparser.add_argument(
        '-nb', '--network-body', default='LSTM'
    )
    subparser.add_argument(
        '--normalize', type=parse_bool, nargs='?', const=True, default=True,
        help='Applies batch norm between linear/convolutional layers and layer norm for LSTMs (default: True)'
    )

def _add_reload_args(parser: ArgumentParser):
    subparser = parser.add_argument_group('Reload Args')
    # Reload from save
    subparser.add_argument(
        '-ln', '--load-network', default='',
        help='Load network from this path. Sets initial step count'
    )
    subparser.add_argument(
        '-lo', '--load-optimizer', default='',
        help='Load optimizer from this path'
    )

def _add_env_args(parser: ArgumentParser):
    subparser = parser.add_argument_group('Environment Args')
    subparser.add_argument(
        '-e', '--env-id', default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)'
    )
    subparser.add_argument(
        '-ne', '--nb-env', type=int, default=32,
        help='number of envs to run in parallel (default: 32)'
    )
    subparser.add_argument(
        '-es', '--skip-rate', type=int, default=4,
        help='frame skip rate (default: 4)'
    )
    subparser.add_argument(
        '-em', '--max-episode-length', type=int, default=10000, metavar='MEL',
        help='maximum length of an episode (default: 10000)'
    )

def _add_common_args(parser: ArgumentParser):
    _add_env_args(parser)
    _add_network_args(parser)
    _add_reload_args(parser)
    """
    Common Arguments
    """
    subparser = parser.add_argument_group('Common Args')
    subparser.add_argument(
        '-cl', '--log-dir', default='/tmp/adept_logs/',
        help='Folder to save logs. (default: /tmp/adept_logs)'
    )
    subparser.add_argument(
        '-ct', '--tag', default='',
        help='Identify your experiment with a tag that gets prepended to the experiment log directory'
    )
    subparser.add_argument(
        '-cm', '--max-train-steps', type=int, default=10e6,
        help='Number of steps to train for (default: 10e6)'
    )
    subparser.add_argument(
        '-ce', '--epoch-len', type=int, default=1e6, metavar='FREQ',
        help='Save models every FREQ steps'
    )
    subparser.add_argument(
        '-cf', '--summary-frequency', default=10,
        help='Write tensorboard summaries every FREQ seconds'
    )
    subparser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed (default: 0)'
    )
    subparser.add_argument(
        '--profile',
        type=parse_bool,
        nargs='?',
        const=True,
        default=False,
        help='displays profiling tree after 10e3 steps (default: False)'
    )
    subparser.add_argument(
        '--debug',
        type=parse_bool,
        nargs='?',
        const=True,
        default=False,
        help='debug mode sends the logs to /tmp/ and overrides number of workers to 3 (default: False)'
    )

def _add_args_to_parsers(arg_fn, parsers):
    return [arg_fn(x) for x in parsers]

def add_base_args(parser: ArgumentParser, additional_args_fn=None):
    # TODO: there must be a better way of adding args to subparsers while keeping the help message
    # TODO: some agents may not run in certain modes not sure the best way to handle this
    subparser_agent = parser.add_subparsers(title='Agents', dest='agent')
    subparser_agent.required = True
    agent_parsers = _add_agent_args(subparser_agent)
    _add_args_to_parsers(_add_common_args, agent_parsers)
    _add_args_to_parsers(additional_args_fn, agent_parsers)

    return parser
