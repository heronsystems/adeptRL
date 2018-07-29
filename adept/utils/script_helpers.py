import os

from gym import spaces
from adept.agents import AGENTS, AGENT_ARGS
from adept.environments import SubProcEnv, SC2_ENVS, Engines, DummyVecEnv
from adept.networks import VISION_NETWORKS, DISCRETE_NETWORKS, NETWORK_BODIES
from adept.networks._base import NetworkTrunk, ModularNetwork, NetworkHead
from adept.utils.util import parse_bool, json_to_dict
from adept.environments.atari import make_atari_env
from adept.environments import reward_normalizer_by_env_id

try:
    from adept.environments.sc2 import make_sc2_env, SC2AgentOverrides
except ImportError:
    print('SC2 Environment not detected')


def make_env(args, seed):
    if args.env_id in SC2_ENVS:
        envs = sc2_from_args(args, seed)
    else:
        envs = atari_from_args(args, seed)
    return envs


def sc2_from_args(args, seed):
    return SubProcEnv([make_sc2_env(args.env_id, seed + i) for i in range(args.nb_env)], Engines.SC2)


def atari_from_args(args, seed, subprocess=True):
    do_frame_stack = args.network_body == 'Linear'

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
        ], Engines.ATARI
    )
    return envs


def make_network(observation_space, network_head_shapes, args):
    # split args into args.visual_pathway, args.discrete_pathway, args.network_body, args.metalearning
    # TODO support different resolutions
    nb_discrete_channel = 0
    nb_visual_channel = 0

    if isinstance(observation_space, spaces.Dict):
        for space in observation_space.spaces.values():
            if isinstance(space, spaces.Box):
                nb_visual_channel += space.shape[0]
            elif isinstance(space, spaces.Discrete):
                nb_discrete_channel += space.shape[0]
            else:
                raise NotImplementedError('This observation space is not currently supported: {}'.format(space))
    elif isinstance(observation_space, spaces.Box):
        nb_visual_channel += observation_space.shape[0]
    else:
        raise NotImplementedError('This observation space is not currently supported: {}'.format(observation_space))

    pathways_by_name = {}
    if nb_visual_channel > 0:
        pathways_by_name['visual'] = VISION_NETWORKS[args.visual_pathway].from_args(nb_visual_channel, args)
    if nb_discrete_channel > 0:
        if args.metalearning:
            nb_metalearning_channel = None
            pathways_by_name['discrete'] = DISCRETE_NETWORKS[args.discrete_pathway].from_args(
                nb_discrete_channel + nb_metalearning_channel,
                args
            )
        else:
            pathways_by_name['discrete'] = DISCRETE_NETWORKS[args.discrete_pathway].from_args(nb_discrete_channel, args)

    trunk = NetworkTrunk(pathways_by_name)
    body = NETWORK_BODIES[args.network_body].from_args(trunk.nb_output_channel, 512, args)  # TODO don't hardcode
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


def make_agent(network, device, engine, args):
    agent_class = get_agent_class(args.agent, engine)
    reward_normalizer = reward_normalizer_by_env_id(args.env_id)
    return agent_class(network, device, reward_normalizer, *AGENT_ARGS[args.agent](args))


def get_head_shapes(action_space, engine, agent_name):
    agent_class = get_agent_class(agent_name, engine)
    return agent_class.output_shape(action_space)


def add_base_args(parser):
    root_dir = os.path.abspath(os.pardir)
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
        '-s', '--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)'
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
        '--log-dir', default=os.path.join(root_dir, 'logs/'),
        help='folder to save logs'
    )
    parser.add_argument(
        '-mts', '--max-train-steps', type=int, default=10e6, metavar='MTS',
        help='number of steps to train for (default: 10e6)'
    )
    parser.add_argument(
        '--nb-top-model', type=int, default=3, metavar='N',
        help='number of top models to save'
    )
    parser.add_argument(
        '--epoch-len', type=int, default=1e6, metavar='FREQ',
        help='save top models every FREQ steps'
    )
    parser.add_argument(
        '-sf', '--summary-frequency', default=10, metavar='FREQ',
        help='write tensorboard summaries every FREQ seconds'
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
        help='number of attention heads'
    )

    return parser
