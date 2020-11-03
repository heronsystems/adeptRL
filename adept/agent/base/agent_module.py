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
from collections import defaultdict

import abc

from adept.exp import ExpSpecBuilder
from adept.utils.requires_args import RequiresArgsMixin


class AgentModule(RequiresArgsMixin, metaclass=abc.ABCMeta):
    """
    An Agent is an Actor (chooses actions) and a Learner (updates parameters)
    and maintains a cache to store a rollout or experience replay.

    Actors and Learners are treated separately for Actor-Learner architectures
    where multiple actors send their experience to a single learner.
    """

    def __init__(self, reward_normalizer, action_space):
        self._reward_normalizer = reward_normalizer
        self._action_space = action_space

    @classmethod
    @abc.abstractmethod
    def from_args(
        cls, args, reward_normalizer, action_space, spec_builder, **kwargs
    ):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def exp_cache(self):
        """Get experience cache"""
        raise NotImplementedError

    @property
    def action_space(self):
        return self._action_space

    @property
    def action_keys(self):
        return list(sorted(self.action_space.keys()))

    @classmethod
    def exp_spec_builder(cls, obs_space, act_space, internal_space, batch_sz):
        def build_fn(exp_len):
            exp_space = cls._exp_spec(
                exp_len, batch_sz, obs_space, act_space, internal_space
            )
            env_space = {
                "rewards": (exp_len, batch_sz),
                "terminals": (exp_len, batch_sz),
            }
            return {**exp_space, **env_space}

        key_types = cls._key_types(obs_space, act_space, internal_space)
        exp_keys = cls._exp_keys(obs_space, act_space, internal_space)
        return ExpSpecBuilder(
            obs_space, act_space, internal_space, key_types, exp_keys, build_fn
        )

    @classmethod
    @abc.abstractmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space):
        raise NotImplementedError

    @classmethod
    def _exp_keys(cls, obs_space, act_space, internal_space):
        dummy = cls._exp_spec(1, 1, obs_space, act_space, internal_space)
        return dummy.keys()

    @classmethod
    def _key_types(cls, obs_space, act_space, internal_space):
        return defaultdict(lambda: "float")

    @staticmethod
    @abc.abstractmethod
    def output_space(action_space):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_action_exp(
        self, predictions, internals, obs, available_actions
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def learn_step(self, updater, network, next_obs, internals):
        raise NotImplementedError

    def is_ready(self):
        return self.exp_cache.is_ready()

    def clear(self):
        self.exp_cache.clear()

    def act(self, network, obs, prev_internals):
        """
        :param network: NetworkModule
        :param obs: Dict[str, Tensor]
        :param prev_internals: previous interal states. Dict[str, Tensor]
        :return:
            actions: Dict[ActionKey, LongTensor (B)]
            internal_states: Dict[str, Tensor]
        """
        predictions, internal_states, pobs = network(obs, prev_internals)

        if "available_actions" in obs:
            av_actions = obs["available_actions"]
        else:
            av_actions = None

        actions, experience = self.compute_action_exp(
            predictions, prev_internals, pobs, av_actions
        )
        self.exp_cache.write_actor(experience)
        return actions, internal_states

    def observe(self, obs, rewards, terminals, infos):
        self.exp_cache.write_env(obs, rewards, terminals, infos)
        return rewards, terminals, infos

    def to(self, device):
        self.exp_cache.to(device)
        return self
