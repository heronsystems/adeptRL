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

from adept.expcaches.rollout import ACRollout
from adept.agents.agent_module import AgentModule
from adept.learner import ACRolloutLearnerMixin
from adept.actor import ACRolloutActorTrainMixin


class ActorCriticAgent(
    AgentModule,
    ACRolloutActorTrainMixin,
    ACRolloutLearnerMixin
):
    args = {
        'nb_rollout': 20,
        'discount': 0.99,
        'gae': True,
        'tau': 1.,
        'normalize_advantage': False,
        'entropy_weight': 0.01
    }

    def __init__(
        self,
        network,
        device,
        reward_normalizer,
        gpu_preprocessor,
        action_space,
        nb_env,
        nb_rollout,
        discount,
        gae,
        tau,
        normalize_advantage,
        entropy_weight
    ):
        super(ActorCriticAgent, self).__init__(
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            action_space,
            nb_env
        )
        self.discount, self.gae, self.tau = discount, gae, tau
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight

        self._exp_cache = ACRollout(nb_rollout, device, reward_normalizer)

    @classmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor,
        action_space, nb_env=None
    ):
        if nb_env is None:
            nb_env = args.nb_env

        return cls(
            network, device, reward_normalizer, gpu_preprocessor, action_space,
            nb_env=nb_env,
            nb_rollout=args.nb_rollout,
            discount=args.discount,
            gae=args.gae,
            tau=args.tau,
            normalize_advantage=args.normalize_advantage,
            entropy_weight=args.entropy_weight
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @property
    def is_train(self):
        return True

    @staticmethod
    def output_space(action_space):
        return ACRolloutActorTrainMixin.output_space(action_space)

    def process_predictions(self, predictions, available_actions):
        return ACRolloutActorTrainMixin.process_predictions(self, predictions, available_actions)

    def compute_loss(self, experiences, next_obs):
        return ACRolloutLearnerMixin.compute_loss(self, experiences, next_obs)
