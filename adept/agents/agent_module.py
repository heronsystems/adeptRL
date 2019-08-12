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
An Agent interacts with the environment and accumulates experience.
"""
import abc

from adept.utils import listd_to_dlist
from adept.utils.requires_args import RequiresArgsMixin

from adept.actor import ActorMixin
from adept.learner import LearnerMixin


class AgentModule(
    ActorMixin, LearnerMixin, RequiresArgsMixin, metaclass=abc.ABCMeta
):
    """
    An Agent is an Actor (chooses actions) and a Learner (updates parameters).

    Actors and Learners are treated separately for Actor-Learner architectures
    where multiple actors send their experience to a single learner.
    """

    def __init__(
        self,
        network,
        device,
        reward_normalizer,
        gpu_preprocessor,
        policy,
        nb_env
    ):
        self._network = network.to(device)
        self._internals = listd_to_dlist(
            [self.network.new_internals(device) for _ in range(nb_env)]
        )
        self._device = device
        self._reward_normalizer = reward_normalizer
        self._gpu_preprocessor = gpu_preprocessor
        self._policy = policy

    @classmethod
    @abc.abstractmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor, policy,
        **kwargs
    ):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def exp_cache(self):
        """Get experience cache"""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def output_space(action_space):
        raise NotImplementedError

    @property
    def device(self):
        """Get experience cache"""
        return self._device

    @property
    def network(self):
        """Get network"""
        return self._network

    @property
    def internals(self):
        """A list of internals"""
        return self._internals

    @internals.setter
    def internals(self, new_internals):
        self._internals = new_internals

    @property
    def gpu_preprocessor(self):
        """Get network"""
        return self._gpu_preprocessor

    @property
    def policy(self):
        return self._policy

    def observe(self, obs, rewards, terminals, infos):
        self.exp_cache.write_env(obs, rewards, terminals, infos)
        self.reset_internals(terminals)
        return rewards, terminals, infos

    def reset_internals(self, terminals):
        for i, terminal in enumerate(terminals):
            if terminal:
                self._reset_internals_at_index(i)

    def _reset_internals_at_index(self, env_idx):
        for k, v in self.network.new_internals(self.device).items():
            self.internals[k][env_idx] = v

    def detach_internals(self):
        for k, vs in self.internals.items():
            self.internals[k] = [v.detach() for v in vs]
