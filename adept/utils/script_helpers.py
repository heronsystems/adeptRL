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

from adept.agents import AGENTS, AGENT_ARGS
from adept.environments import SubProcEnv, SC2_ENVS, Engines, DummyVecEnv
from adept.environments import reward_normalizer_by_env_id
from adept.environments.atari import make_atari_env
from adept.networks import VISION_NETWORKS, DISCRETE_NETWORKS, NETWORK_BODIES
from adept.networks._base import NetworkTrunk, ModularNetwork, NetworkHead
from adept.utils.util import parse_bool

try:
    from adept.environments.sc2 import make_sc2_env, SC2AgentOverrides
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
        return SubProcEnv([make_sc2_env(args.env_id, seed + i) for i in range(args.nb_env)], Engines.SC2)
    else:
        return DummyVecEnv([make_sc2_env(args.env_id, seed + i, render=render) for i in range(args.nb_env)], Engines.SC2)


def atari_from_args(args, seed, subprocess=True):
    do_frame_stack = 'Linear' in args.network_body

    env_wrapper_class = SubProcEnv if subprocess else DummyVecEnv
    envs = env_wrapper_class(
        [
            make_atari_env(
                args.env_id,
                args.skip_rate,
                args.max_episode_length,
                args.zscore_norm_env,
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
                pathways_by_name[name] = c_networks[args.discrete_network].from_args(ebn[name].shape, args)
            elif rank == 2:
                raise NotImplementedError('Rank 2 inputs not implemented')
            elif rank == 3:
                pathways_by_name[name] = chw_networks[args.vision_network].from_args(ebn[name].shape, args)
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


def get_agent_class(agent_name, engine):
    if engine == Engines.SC2:
        agent_class = type(agent_name, (SC2AgentOverrides, AGENTS[agent_name]), {})
    else:
        agent_class = AGENTS[agent_name]
    return agent_class


def make_agent(network, device, engine, gpu_preprocessor, args):
    agent_class = get_agent_class(args.agent, engine)
    reward_normalizer = reward_normalizer_by_env_id(args.env_id)
    return agent_class(network, device, reward_normalizer, gpu_preprocessor, *AGENT_ARGS[args.agent](args))


def get_head_shapes(action_space, engine, agent_name):
    agent_class = get_agent_class(agent_name, engine)
    return agent_class.output_shape(action_space)


def add_base_args(parser):
    """
    Common Arguments
    """
    parser.add_argument(
        '--learning-rate', '-lr', type=float, default=7e-4, metavar='LR',
        help='learning rate (default: 7e-4)'
    )
    parser.add_argument(
        '--discount', type=float, default=0.99, metavar='D',
        help='discount factor for rewards (default: 0.99)'
    )
    parser.add_argument(
        '-s', '--seed', type=int, default=0, metavar='S',
        help='random seed (default: 0)'
    )
    parser.add_argument(
        '-n', '--nb-env', type=int, default=32, metavar='N',
        help='number of envs to run in parallel (default: 32)'
    )
    parser.add_argument(
        '-mel', '--max-episode-length', type=int, default=10000, metavar='MEL',
        help='maximum length of an episode (default: 10000)'
    )
    parser.add_argument(
        '--env-id', default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)'
    )
    parser.add_argument(
        '--log-dir', default='/tmp/adept_logs/',
        help='folder to save logs. (default: /tmp/adept_logs)'
    )
    parser.add_argument(
        '-mts', '--max-train-steps', type=int, default=10e6, metavar='MTS',
        help='number of steps to train for (default: 10e6)'
    )
    parser.add_argument(
        '--nb-top-model', type=int, default=3, metavar='N',
        help='number of top models to save per epoch'
    )
    parser.add_argument(
        '--epoch-len', type=int, default=1e6, metavar='FREQ',
        help='save top models every FREQ steps'
    )
    parser.add_argument(
        '-sf', '--summary-frequency', default=10, metavar='FREQ',
        help='write tensorboard summaries every FREQ seconds'
    )
    parser.add_argument(
        '-t', '--tag', default='',
        help='identify your experiment with a tag that gets prepended to experiment log directory'
    )

    """
    Env Arguments
    """
    # Atari
    parser.add_argument(
        '--zscore-norm-env', type=parse_bool, nargs='?', const=True, default=False,
        help='Normalize the environment using running statistics'
    )
    parser.add_argument(
        '--skip-rate', type=int, default=4,
        help='frame skip rate (default: 4)'
    )

    """
    Agent Arguments
    """
    parser.add_argument(
        '-gae', '--generalized-advantage-estimation', type=parse_bool, nargs='?', const=True, default=True,
        help='use generalized advantage estimation'
    )
    parser.add_argument(
        '--tau', type=float, default=1.00,
        help='parameter for GAE (default: 1.00)'
    )
    # Actor Critic
    parser.add_argument(
        '--exp-length', '--nb-rollout', type=int, default=20,
        help='number of rollouts or size of experience replay'
    )
    parser.add_argument(
        '--ppo-nb-epoch', type=int, default=3,
        help='number of times to learn on a rollout (default: 3)'
    )
    parser.add_argument(
        '--ppo-mb-size', type=int, default=32,
        help='PPO minibatch size (default: 32)'
    )
    parser.add_argument(
        '--ppo-clip', type=float, default=0.1,
        help='PPO policy surrogate loss clipping (default: 0.1)'
    )

    """
    Network Arguments
    """
    parser.add_argument(
        '--normalize', type=parse_bool, nargs='?', const=True, default=True,
        help='applies batch norm between linear/convolutional layers and layer norm for LSTMs'
    )

    # Attention
    parser.add_argument(
        '--nb-head', type=int, default=1,
        help='number of attention heads. unused if no attention in network.'
    )

    return parser
