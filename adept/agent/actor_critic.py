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
from adept.actor import ACRolloutActorTrain
from adept.exp import Rollout
from adept.learner import ACRolloutLearner
from .base.agent_module import AgentModule


class ActorCritic(AgentModule):
    args = {
        **Rollout.args,
        **ACRolloutActorTrain.args,
        **ACRolloutLearner.args
    }

    def __init__(
            self,
            reward_normalizer,
            action_space,
            spec_builder,
            rollout_len,
            discount,
            normalize_advantage,
            entropy_weight,
            return_scale
    ):
        super(ActorCritic, self).__init__(
            reward_normalizer,
            action_space
        )
        self.discount = discount
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight

        self._exp_cache = Rollout(spec_builder, rollout_len)
        self._actor = ACRolloutActorTrain(action_space)
        self._learner = ACRolloutLearner(
            reward_normalizer,
            discount,
            normalize_advantage,
            entropy_weight,
            return_scale
        )

    @classmethod
    def from_args(
        cls, args, reward_normalizer,
        action_space, spec_builder, **kwargs
    ):
        return cls(
            reward_normalizer, action_space, spec_builder,
            rollout_len=args.rollout_len,
            discount=args.discount,
            normalize_advantage=args.normalize_advantage,
            entropy_weight=args.entropy_weight,
            return_scale=args.return_scale
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space):
        return ACRolloutActorTrain._exp_spec(
            exp_len, batch_sz, obs_space, act_space, internal_space
        )

    @staticmethod
    def output_space(action_space):
        return ACRolloutActorTrain.output_space(action_space)

    def compute_action_exp(self, predictions, internals, available_actions):
        return self._actor.compute_action_exp(predictions, internals, available_actions)

    def compute_loss(self, network, next_obs, internals):
        return self._learner.compute_loss(network, self.exp_cache.read(),
                                          next_obs, internals)
