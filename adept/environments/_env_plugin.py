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
import json

from adept.environments._env import EnvBase


class EnvPlugin(EnvBase, metaclass=abc.ABCMeta):
    """
    Implement this class to add your custom environment. Don't forget to
    implement defaults.
    """
    defaults = None

    @classmethod
    def check_defaults(cls):
        if cls.defaults is None:
            raise NotImplementedError(
                'Subclass must define class attribute: defaults'
            )

    def __init__(self, action_space, cpu_preprocessor, gpu_preprocessor):
        """
        :param observation_space: ._spaces.Spaces
        :param action_space: ._spaces.Spaces
        :param cpu_preprocessor: adept.preprocess.observation.ObsPreprocessor
        :param gpu_preprocessor: adept.preprocess.observation.ObsPreprocessor
        """
        self._action_space = action_space
        self._cpu_preprocessor = cpu_preprocessor
        self._gpu_preprocessor = gpu_preprocessor

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args, seed, **kwargs):
        """
        Construct from arguments. For convenience.

        :param args: Arguments object
        :param seed: Integer used to seed this environment.
        :param kwargs: Any custom arguments are passed through kwargs.
        :return: EnvPlugin instance.
        """
        raise NotImplementedError

    @classmethod
    def from_args_curry(cls, args, seed, **kwargs):
        def _f():
            return cls.from_args(args, seed, **kwargs)

        return _f

    @property
    def observation_space(self):
        return self._gpu_preprocessor.observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def cpu_preprocessor(self):
        return self._cpu_preprocessor

    @property
    def gpu_preprocessor(self):
        return self._gpu_preprocessor

    @classmethod
    def prompt(cls):
        """
        Display defaults as JSON, prompt user for changes.

        :return: Dict[str, Any] Updated config dictionary.
        """
        if not cls.defaults:
            return cls.defaults

        user_input = input(
            '\n{} Defaults:\n{}\nPress ENTER to use defaults. Otherwise, '
            'modify JSON keys then press ENTER.\n'.format(
                cls.__name__,
                json.dumps(cls.defaults, indent=2, sort_keys=True)
            )
        )

        # use defaults if no changes specified
        if user_input == '':
            return cls.defaults

        updates = json.loads(user_input)
        return {**cls.defaults, **updates}
