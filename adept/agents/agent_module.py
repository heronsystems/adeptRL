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


class AgentModule(RequiresArgsMixin, metaclass=abc.ABCMeta):
    """
    An Agent is an Actor (chooses actions) and a Learner (updates parameters)
    and maintains a cache to store a rollout or experience replay.

    Actors and Learners are treated separately for Actor-Learner architectures
    where multiple actors send their experience to a single learner.
    """

    def __init__(
        self,
        network,
        reward_normalizer,
        gpu_preprocessor,
        action_space
    ):
        self._network = network
        self._reward_normalizer = reward_normalizer
        self._gpu_preprocessor = gpu_preprocessor
        self._action_space = action_space

        # agent only ever trains, never eval
        self.network.train()

    @classmethod
    @abc.abstractmethod
    def from_args(
        cls, args, network, reward_normalizer, gpu_preprocessor,
        action_space, **kwargs
    ):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def exp_cache(self):
        """Get experience cache"""
        raise NotImplementedError

    @property
    def network(self):
        """Get network"""
        return self._network

    @property
    def gpu_preprocessor(self):
        """Get network"""
        return self._gpu_preprocessor

    @property
    def action_space(self):
        return self._action_space

    @property
    def action_keys(self):
        return list(sorted(self.action_space.keys()))

    @staticmethod
    @abc.abstractmethod
    def output_space(action_space):
        raise NotImplementedError

    @abc.abstractmethod
    def process_predictions(self, predictions, available_actions):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, experiences, next_obs):
        raise NotImplementedError

    def act(self, obs, prev_internals):
        """
        :param obs: Dict[str, Tensor]
        :param prev_internals: previous interal states. Dict[str, Tensor]
        :return:
            actions: Dict[ActionKey, LongTensor (B)]
            internal_states: Dict[str, Tensor]
        """
        predictions, internal_states = self.network(
            self.gpu_preprocessor(obs),
            prev_internals
        )

        if 'available_actions' in obs:
            av_actions = obs['available_actions']
        else:
            av_actions = None

        actions, experience = self.process_predictions(predictions, av_actions)
        self.exp_cache.write_forward(actions, experience)
        return actions, internal_states

    def observe(self, obs, rewards, terminals, infos):
        self.exp_cache.write_env(obs, rewards, terminals, infos)
        return rewards, terminals, infos
