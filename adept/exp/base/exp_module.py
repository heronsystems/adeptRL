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

from adept.utils.requires_args import RequiresArgsMixin


class ExpModule(RequiresArgsMixin, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def write_actor(self, experience):
        raise NotImplementedError

    @abc.abstractmethod
    def write_env(self, obs, rewards, terminals, infos):
        raise NotImplementedError

    @abc.abstractmethod
    def read(self):
        raise NotImplementedError

    @abc.abstractmethod
    def is_ready(self):
        raise NotImplementedError

    @abc.abstractmethod
    def to(self, device):
        raise NotImplementedError

    @abc.abstractclassmethod
    def from_args(cls, args, spec_builder):
        raise NotImplementedError

