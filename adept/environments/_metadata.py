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
from adept.environments._env import HasEnvMetaData
from adept.environments.env_registry import EnvRegistry


class EnvMetaData(HasEnvMetaData):
    """
    Used to provide environment metadata without spawning multiple processes.

    Networks need an action_space and observation_space
    Agents need an gpu_preprocessor, engine, and action_space
    """

    def __init__(self, env_plugin_class, args):
        dummy_env = env_plugin_class.from_args(args, 0)
        dummy_env.close()

        self._action_space = dummy_env.action_space
        self._observation_space = dummy_env.observation_space
        self._cpu_preprocessor = dummy_env.cpu_preprocessor
        self._gpu_preprocessor = dummy_env.gpu_preprocessor

    @classmethod
    def from_args(cls, args, registry=EnvRegistry()):
        """
        Mimic the EnvModule.from_args API to simplify interface.

        :param args: Arguments object
        :param registry: Optionally provide to avoid recreating.
        :return: EnvMetaData
        """
        module_class = registry.lookup_env_class(args.env)
        return cls(module_class, args)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def cpu_preprocessor(self):
        return self._cpu_preprocessor

    @property
    def gpu_preprocessor(self):
        return self._gpu_preprocessor
