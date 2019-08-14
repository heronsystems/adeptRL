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
"""
A Learner receives agent-environment interactions which are used to compute
loss.
"""
import abc
from adept.utils.requires_args import RequiresArgsMixin


class LearnerMixin:
    """
    This mixin interface is used for Agents.
    """

    @abc.abstractmethod
    def compute_loss(self, experiences, next_obs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def network(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gpu_preprocessor(self):
        raise NotImplementedError


class LearnerModule(LearnerMixin, RequiresArgsMixin, metaclass=abc.ABCMeta):
    """
    This one of the modules to use for custom Actor-Learner code.
    """
    def __init__(self, network, gpu_preprocessor):
        self._network = network
        self._gpu_preprocessor = gpu_preprocessor

    @property
    def network(self):
        return self._network

    @property
    def gpu_preprocessor(self):
        return self._gpu_preprocessor
