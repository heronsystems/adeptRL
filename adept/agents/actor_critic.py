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

from adept.expcaches.rollout import RolloutCache
from adept.agents.agent_module import AgentModule
from adept.learner import ActorCriticLearnerMixin


class ActorCritic(AgentModule, ActorCriticLearnerMixin):
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
        policy,
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
            policy,
            nb_env
        )
        self.discount, self.gae, self.tau = discount, gae, tau
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight

        self._exp_cache = RolloutCache(
            nb_rollout, device, reward_normalizer,
            ['values', 'log_probs', 'entropies']
        )
        self._func_id_to_headnames = None

    @classmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor, policy,
        nb_env=None
    ):
        if nb_env is None:
            nb_env = args.nb_env

        return cls(
            network, device, reward_normalizer, gpu_preprocessor, policy,
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

    @staticmethod
    def output_space(action_space):
        head_dict = {'critic': (1, ), **action_space}
        return head_dict

    def act(self, obs):
        self.network.train()
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals
        )
        values = predictions['critic'].squeeze(1)
        if 'available_actions' in obs:
            actions, log_probs, entropies = self.policy.act(
                predictions, obs['available_actions']
            )
        else:
            actions, log_probs, entropies = self.policy.act(predictions)

        self.exp_cache.write_forward(
            values=values, log_probs=log_probs, entropies=entropies
        )
        self.internals = internals
        return actions

    def act_eval(self, obs):
        self.network.eval()
        with torch.no_grad():
            predictions, internals = self.network(
                self.gpu_preprocessor(obs, self.device), self.internals
            )
        if 'available_actions' in obs:
            actions = self.policy.act_eval(
                predictions, obs['available_actions']
            )
        else:
            actions = self.policy.act_eval(predictions)

        self.internals = internals
        return actions
