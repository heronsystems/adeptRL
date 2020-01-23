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

import torch

from adept.utils.requires_args import RequiresArgsMixin


class LearnerModule(RequiresArgsMixin, metaclass=abc.ABCMeta):
    """
    This one of the modules to use for custom Actor-Learner code.
    """
    def __init__(self, optimizer):
        super(LearnerModule, self).__init__()
        self.optimizer = optimizer

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args, reward_normalizer, optimizer):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, network, experiences, next_obs, internals):
        raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.optimizer.step()

    def load_optim(self, optimizer, path):
        optimizer.load_state_dict(
            torch.load(
                path,
                map_location=lambda storage, loc: storage
            )
        )
        self.optimizer = optimizer
        return optimizer
