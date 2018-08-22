#!python
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
from adept.environments import ATARI_6_ENVS
from adept.utils.util import parse_bool
from .local import main

if __name__ == '__main__':
    import argparse
    from adept.utils.script_helpers import add_base_args

    parser = argparse.ArgumentParser(description='AdeptRL Local Mode')
    parser = add_base_args(parser)
    parser.add_argument('--gpu-id', type=int, default=0, help='Which GPU to use for training (default: 0)')
    parser.add_argument(
        '-vn', '--vision-network', default='Nature',
        help='name of preset network (default: Nature)'
    )
    parser.add_argument(
        '-dp', '--discrete-pathway', default='DiscreteIdentity',
    )
    parser.add_argument(
        '-nb', '--network-body', default='Linear',
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

    args.mode_name = 'Local'

    for env_id in ATARI_6_ENVS:
        args.env_id = env_id
        main(args)
