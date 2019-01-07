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
    @abc.abstractmethod
    def step(self, action):
        """
        :param action: Dict[ActionID, Any] Action dictionary
        :return: Tuple[Observation, Reward, Terminal, Info]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, **kwargs):
        """
        :param kwargs:
        :return: Dict[ObservationID, Any] Observation dictionary
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """
        Close environment. Release resources.

        :return:
        """
        raise NotImplementedError
