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


class HasEnvMetaData(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def observation_space(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_space(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cpu_preprocessor(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gpu_preprocessor(self):
        raise NotImplementedError


class EnvBase(HasEnvMetaData, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def defaults(self):
        """
        :return: Dictionary of defaults.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError

    def prompt(self):
        """
        Display defaults as JSON, prompt user for changes.

        :return: Dict[str, Any] Updated config dictionary.
        """
        if not self.defaults:
            return self.defaults

        user_input = input(
            '\n{} Defaults:\n{}\nPress ENTER to use defaults. Otherwise, '
            'modify JSON keys then press ENTER.\n'.format(
                self.__class__.__name__,
                json.dumps(self.defaults,  indent=2, sort_keys=True)
            )
        )

        # use defaults if no changes specified
        if user_input == '':
            return self.defaults

        updates = json.loads(user_input)
        return {**self.defaults, **updates}


def reward_normalizer_by_env_id(env_id):
    from adept.utils.normalizers import Clip, Scale
    norm_by_id = {
        'DefeatRoaches': Scale(0.1),
        'DefeatZerglingsAndBanelings': Scale(0.2)
    }
    if env_id not in norm_by_id:
        return Clip()
    else:
        return norm_by_id[env_id]
