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
from adept.utils.requires_args import RequiresArgs


class AgentPlugin(RequiresArgs, metaclass=abc.ABCMeta):
    """
    An Agent interacts with the environment and accumulates experience.
    """

    @classmethod
    @abc.abstractmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor, engine,
        action_space, **kwargs
    ):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def exp_cache(self):
        """Get experience cache"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self):
        """Get experience cache"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def network(self):
        """Get network"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def internals(self):
        """A list of internals"""
        raise NotImplementedError

    @internals.setter
    @abc.abstractmethod
    def internals(self, new_internals):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def output_shape(action_space):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def act_eval(self, obs):
        raise NotImplementedError

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
