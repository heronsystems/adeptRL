import json
import os

import torch

from adept.containers import Renderer
from adept.utils.logging import print_ascii_logo
from adept.utils.script_helpers import make_agent, make_network, get_head_shapes, atari_from_args
from adept.utils.util import dotdict


def main(args):
    # construct logging objects
    print_ascii_logo()
    print('Rendering... Press Ctrl+C to stop.')

    with open(args.args_file, 'r') as args_file:
        train_args = dotdict(json.load(args_file))
    train_args.nb_env = 1
    train_args.seed = 2

    # construct env
    env = atari_from_args(train_args, train_args.seed, subprocess=False)

    # construct network
    network_head_shapes = get_head_shapes(env.action_space, env.engine, train_args)
    network = make_network(
        env.observation_space,
        network_head_shapes,
        train_args
    )
    network.load_state_dict(torch.load(args.network_file))

    # create an agent (add act_eval method)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = make_agent(network, device, env.engine, train_args)

    # create a rendering container
    renderer = Renderer(agent, env, device)
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
    parser.add_argument('--gpu-id', type=int, default=0, help='Which GPU to use (default: 0)')
    args = parser.parse_args()
    main(args)
