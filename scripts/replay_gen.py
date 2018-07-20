import json
import os

import torch
from absl import flags

from adept.containers import ReplayGenerator
from adept.environments import SubProcEnv, Engines
from adept.environments.sc2 import make_sc2_env
from adept.utils.script_helpers import make_agent, make_network, agent_output_shape
from adept.utils.util import dotdict
from adept.utils.logging import print_ascii_logo

# hack to use argparse for SC2
FLAGS = flags.FLAGS
FLAGS(['local.py'])


def main(args):
    print_ascii_logo()
    print('Saving replays... Press Ctrl+C to stop.')

    with open(args.args_file, 'r') as args_file:
        train_args = dotdict(json.load(args_file))
    train_args.nb_env = 1

    # construct env
    replay_dir = os.path.split(args.network_file)[0]
    env = SubProcEnv([make_sc2_env(train_args.env_id, train_args.seed, replay_dir=replay_dir)], Engines.SC2)

    # construct network
    network_head_shapes = agent_output_shape(env.action_space, env.engine, train_args)
    network = make_network(
        env.observation_space,
        network_head_shapes,
        train_args
    )
    network.load_state_dict(torch.load(args.network_file))

    # create an agent (add act_eval method)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    agent = make_agent(network, device, env.engine, train_args)

    # create a rendering container
    # TODO: could terminate after a configurable number of replays instead of running indefinitely
    renderer = ReplayGenerator(agent, env, device)
    try:
        renderer.run()
    finally:
        env.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='AdeptRL Renderer')
    parser.add_argument(
        '--network-file',
        help='path to args file (.../logs/<env-id>/<log-id>/<epoch>/model.pth)'
    )
    parser.add_argument(
        '--args-file',
        help='path to args file (.../logs/<env-id>/<log-id>/args.json)'
    )
    parser.add_argument('--gpu-id', type=int, default=0, help='Which GPU to use for training (default: 0)')
    args = parser.parse_args()
    main(args)
