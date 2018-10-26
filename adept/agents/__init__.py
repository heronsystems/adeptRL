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
from adept.utils.util import parse_bool
from .actor_critic import ActorCritic
from .impala import ActorCriticVtrace
from ._base import EnvBase

AGENTS = {'ActorCritic': ActorCritic, 'ActorCriticVtrace': ActorCriticVtrace}
AGENT_ARGS = {
    'ActorCritic': lambda args: (
        args.env_nb, args.exp_length, args.discount, args.generalized_advantage_estimation,
        args.tau, args.normalize_advantage, args.entropy_weight
    ),
    'ActorCriticVtrace': lambda args: (args.env_nb, args.exp_length, args.discount),
}


def ActorCritic_ArgParse(parser: ArgumentParser):
    parser.add_argument(
        '-ae',
        '--exp-length',
        type=int,
        default=20,
        help='Experience length (default: 20)'
    )
    parser.add_argument(
        '-ag',
        '--generalized-advantage-estimation',
        type=parse_bool,
        nargs='?',
        const=True,
        default=True,
        help='Use generalized advantage estimation for the policy loss. (default: True)'
    )
    parser.add_argument(
        '-at',
        '--tau',
        type=float,
        default=1.00,
        help='parameter for GAE (default: 1.00)'
    )
    parser.add_argument(
        '--entropy-weight',
        type=float,
        default=0.01,
        help='Entropy penalty (default: 0.01)'
    )
    parser.add_argument(
        '--normalize-advantage',
        type=parse_bool,
        nargs='?',
        const=True,
        default=False,
        help='Normalize the advantage when calculating policy loss. (default: False)'
    )


def ActorCriticVtrace_ArgParse(parser: ArgumentParser):
    parser.add_argument(
        '-ae',
        '--exp-length',
        type=int,
        default=20,
        help='Experience length (default: 20)'
    )


AGENT_ARG_PARSE = {
    'ActorCritic': ActorCritic_ArgParse,
    'ActorCriticVtrace': ActorCriticVtrace_ArgParse
}
