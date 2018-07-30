from adept.environments import ATARI_6_ENVS
from adept.utils.util import parse_bool
from scripts.local import main

if __name__ == '__main__':
    import argparse
    from adept.utils.script_helpers import add_base_args

    parser = argparse.ArgumentParser(description='AdeptRL Local Mode')
    parser = add_base_args(parser)
    parser.add_argument('--gpu-id', type=int, default=0, help='Which GPU to use for training (default: 0)')
    parser.add_argument(
        '--visual-pathway', default='Nature',
        help='name of preset network (default: Nature)'
    )
    parser.add_argument(
        '--discrete-pathway', default='DiscreteIdentity',
    )
    parser.add_argument(
        '--network-body', default='Linear',
    )
    parser.add_argument(
        '--agent', default='ActorCritic',
        help='name of preset agent (default: ActorCritic)'
    )
    parser.add_argument(
        '--profile', type=parse_bool, nargs='?', const=True, default=False,
        help='displays profiling tree after 10e3 steps (default: False)'
    )

    args = parser.parse_args()

    args.max_train_steps = 10e6
    args.mode_name = 'Local'

    for env_id in ATARI_6_ENVS:
        args.env_id = env_id
        main(args)
