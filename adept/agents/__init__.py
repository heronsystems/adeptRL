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
from .actor_critic import ActorCritic
from .impala import ActorCriticVtrace
from ._base import EnvBase

AGENTS = {
    'ActorCritic': ActorCritic,
    'ActorCriticVtrace': ActorCriticVtrace
}
AGENT_ARGS = {
    'ActorCritic': lambda args: (
        args.nb_env, args.exp_length, args.discount, args.generalized_advantage_estimation, args.tau
    ),
    'ActorCriticVtrace': lambda args: (args.nb_env, args.exp_length, args.discount),
}
