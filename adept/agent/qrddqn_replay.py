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
import torch

from adept.actor import QRDDQNReplayActor
from adept.exp import ExperienceReplay
from adept.learner import QRDDQNRolloutLearner
from .base.agent_module import AgentModule


class QRDDQNReplay(AgentModule):
    args = {
        **ExperienceReplay.args,
        **QRDDQNReplayActor.args,
        **QRDDQNRolloutLearner.args
    }

    def __init__(
            self,
            reward_normalizer,
            action_space,
            spec_builder,
            exp_size, 
            exp_min_size, 
            exp_update_rate,
            rollout_len,
            discount,
            nb_env,
            return_scale,
            double_dqn,
            num_atoms
    ):
        super(QRDDQNReplay, self).__init__(
            reward_normalizer,
            action_space
        )
        self._exp_cache = ExperienceReplay(spec_builder, exp_size,
                                           exp_min_size, rollout_len, exp_update_rate)
        self._actor = QRDDQNReplayActor(action_space, nb_env)
        self._learner = QRDDQNRolloutLearner(
            reward_normalizer,
            discount,
            return_scale,
            double_dqn,
            num_atoms
        )

    @classmethod
    def from_args(
        cls, args, reward_normalizer,
        action_space, spec_builder, **kwargs
    ):
        return cls(
            reward_normalizer, action_space, spec_builder,
            exp_size=args.exp_size,
            exp_min_size=args.exp_min_size,
            rollout_len=args.rollout_len,
            exp_update_rate=args.exp_update_rate,
            discount=args.discount,
            nb_env=args.nb_env,
            return_scale=args.return_scale,
            double_dqn=args.double_dqn,
            num_atoms=args.num_atoms
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space):
        return QRDDQNReplayActor._exp_spec(
            exp_len, batch_sz, obs_space, act_space, internal_space
        )

    @staticmethod
    def output_space(action_space):
        return QRDDQNReplayActor.output_space(action_space)

    def act(self, network, obs, prev_internals):
        with torch.no_grad():
            return super().act(network, obs, prev_internals)

    def compute_action_exp(self, predictions, internals, available_actions):
        with torch.no_grad():
            return self._actor.compute_action_exp(predictions, internals, available_actions)

    def compute_loss(self, network, next_obs, internals):
        return self._learner.compute_loss(network, self.exp_cache.read(),
                                          next_obs, internals)

