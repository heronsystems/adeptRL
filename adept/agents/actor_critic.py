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
from adept.agents.agent_module import AgentModule
from adept.expcaches.rollout import ACRollout
from adept.learner import ACRolloutLearner


class ActorCritic(AgentModule):
    args = {
        'nb_rollout': 20,
        **ACRolloutActorTrain.args,
        **ACRolloutLearner.args
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
        super(ActorCritic, self).__init__(
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            action_space
        )
        self.discount, self.gae, self.tau = discount, gae, tau
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight

        self._exp_cache = ACRollout(nb_rollout, device, reward_normalizer)
        self._actor = ACRolloutActorTrain(network, gpu_preprocessor, action_space)
        self._learner = ACRolloutLearner(network, gpu_preprocessor)

    @classmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor,
        action_space
    ):

        return cls(
            network, device, reward_normalizer, gpu_preprocessor, action_space,
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
        return ACRolloutActorTrain.output_space(action_space)

    def process_predictions(self, predictions, available_actions):
        return self._actor.process_predictions(predictions, available_actions)

    def compute_loss(self, experiences, next_obs):
        return self._learner.compute_loss(experiences, next_obs)
