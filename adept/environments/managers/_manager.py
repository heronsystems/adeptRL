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
import abc

from adept.environments._env import EnvBase
from adept.registries.environment import EnvPluginRegistry


class AdeptEnvManager(EnvBase, metaclass=abc.ABCMeta):
    def __init__(self, env_fns, engine):
        self._env_fns = env_fns
        self._engine = engine

    @property
    def env_fns(self):
        return self._env_fns

    @property
    def engine(self):
        return self._engine

    @property
    def nb_env(self):
        return len(self._env_fns)

    @classmethod
    def from_args(
        cls, args, seed_start=None, registry=EnvPluginRegistry(), **kwargs
    ):
        if seed_start is None:
            seed_start = int(args.seed)

        engine = registry.lookup_engine(args.env_id)
        env_class = registry.lookup_env_class(args.env_id)

        env_fns = []
        for i in range(args.nb_env):
            env_fns.append(
                env_class.from_args_curry(args, seed_start + i, **kwargs)
            )
        return cls(env_fns, engine)
