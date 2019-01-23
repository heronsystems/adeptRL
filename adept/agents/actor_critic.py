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
from collections import OrderedDict

import torch
from torch.nn import functional as F

from adept.expcaches.rollout import RolloutCache
from adept.agents.agent_module import AgentModule


class ActorCritic(AgentModule):
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
        engine,
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
            engine,
            action_space,
            nb_env
        )
        self.discount, self.gae, self.tau = discount, gae, tau
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight

        self._exp_cache = RolloutCache(
            nb_rollout, device, reward_normalizer,
            ['values', 'log_probs', 'entropies']
        )
        self._action_keys = list(sorted(action_space.keys()))
        self._func_id_to_headnames = None
        if self.engine == 'AdeptSC2Env':
            from adept.environments.deepmind_sc2 import SC2ActionLookup
            self._func_id_to_headnames = SC2ActionLookup()

    @classmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor, engine,
        action_space, nb_env=None
    ):
        if nb_env is None:
            nb_env = args.nb_env

        return cls(
            network, device, reward_normalizer, gpu_preprocessor, engine,
            action_space,
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

        if self.engine == 'AdeptSC2Env':
            return self._act_sc2(obs)
        else:
            return self._act_gym(obs)

    def _act_gym(self, obs):
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals
        )
        values = predictions['critic'].squeeze(1)

        # reduce feature dim, build action_key dim
        actions = OrderedDict()
        log_probs = []
        entropies = []
        # TODO support multi-dimensional action spaces?
        for key in self._action_keys:
            logit = predictions[key]
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            action = prob.multinomial(1)
            log_prob = log_prob.gather(1, action)

            actions[key] = action.squeeze(1).cpu().numpy()
            log_probs.append(log_prob)
            entropies.append(entropy)

        log_probs = torch.cat(log_probs, dim=1)
        entropies = torch.cat(entropies, dim=1)

        self.exp_cache.write_forward(
            values=values, log_probs=log_probs, entropies=entropies
        )
        self.internals = internals
        return actions

    def _act_sc2(self, obs):
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals
        )
        values = predictions['critic'].squeeze(1)

        # reduce feature dim, build action_key dim
        actions = OrderedDict()
        head_masks = OrderedDict()
        log_probs = []
        entropies = []
        # TODO support multi-dimensional action spaces?
        for key in self._action_keys:
            logit = predictions[key]
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            action = prob.multinomial(1)
            log_prob = log_prob.gather(1, action)

            actions[key] = action.squeeze(1).cpu().numpy()
            log_probs.append(log_prob)
            entropies.append(entropy)

            # Initialize masks
            if key == 'func_id':
                head_masks[key] = torch.ones_like(entropy)
            else:
                head_masks[key] = torch.zeros_like(entropy)

        log_probs = torch.cat(log_probs, dim=1)
        entropies = torch.cat(entropies, dim=1)

        # Mask invalid actions with NOOP and fill masks with ones
        for batch_idx, action in enumerate(actions['func_id']):
            # convert unavailable actions to NOOP
            if obs['available_actions'][batch_idx][action] == 0:
                actions['func_id'][batch_idx] = 0

            # build SC2 action masks
            func_id = actions['func_id'][batch_idx]
            # TODO this can be vectorized via gather
            for headname in self._func_id_to_headnames[func_id].keys():
                head_masks[headname][batch_idx] = 1.

        head_masks = torch.cat(
            [head_mask for head_mask in head_masks.values()], dim=1
        )
        log_probs = log_probs * head_masks
        entropies = entropies * head_masks

        self.exp_cache.write_forward(
            values=values, log_probs=log_probs, entropies=entropies
        )
        self.internals = internals
        return actions

    def act_eval(self, obs):
        self.network.eval()

        if self.engine == 'AdeptSC2Env':
            return self._act_eval_sc2(obs)
        else:
            return self._act_eval_gym(obs)

    def _act_eval_gym(self, obs):
        with torch.no_grad():
            predictions, internals = self.network(
                self.gpu_preprocessor(obs, self.device), self.internals
            )

            # reduce feature dim, build action_key dim
            actions = OrderedDict()
            for key in self._action_keys:
                logit = predictions[key]
                prob = F.softmax(logit, dim=1)
                action = torch.argmax(prob, 1)
                actions[key] = action.cpu().numpy()

        self.internals = internals
        return actions

    def _act_eval_sc2(self, obs):
        with torch.no_grad():
            predictions, internals = self.network(
                self.gpu_preprocessor(obs, self.device), self.internals
            )

            # reduce feature dim, build action_key dim
            actions = OrderedDict()
            for key in self._action_keys:
                logit = predictions[key]
                prob = F.softmax(logit, dim=1)
                action = torch.argmax(prob, 1)
                actions[key] = action.cpu().numpy()

        for batch_idx, action in enumerate(actions['func_id']):
            # convert unavailable actions to NOOP
            if action not in obs['available_actions'][batch_idx]:
                actions['func_id'][batch_idx] = 0

        self.internals = internals
        return actions

    def compute_loss(self, rollouts, next_obs):
        # estimate value of next state
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            results, _ = self.network(next_obs_on_device, self.internals)
            last_values = results['critic'].squeeze(1).data

        # compute nstep return and advantage over batch
        batch_values = torch.stack(rollouts.values)
        value_targets, batch_advantages = self._compute_returns_advantages(
            batch_values, last_values, rollouts.rewards, rollouts.terminals
        )

        # batched value loss
        value_loss = 0.5 * torch.mean((value_targets - batch_values).pow(2))

        # normalize advantage so that an even number
        # of actions are reinforced and penalized
        if self.normalize_advantage:
            batch_advantages = (batch_advantages - batch_advantages.mean()) \
                               / (batch_advantages.std() + 1e-5)
        policy_loss = 0.
        entropy_loss = 0.

        rollout_len = len(rollouts.rewards)
        for i in range(rollout_len):
            log_probs = rollouts.log_probs[i]
            entropies = rollouts.entropies[i]

            policy_loss = policy_loss - (
                log_probs * batch_advantages[i].unsqueeze(1).data
            ).sum(1)
            entropy_loss = entropy_loss - (
                self.entropy_weight * entropies
            ).sum(1)

        batch_size = policy_loss.shape[0]
        nb_action = log_probs.shape[1]

        denom = batch_size * rollout_len * nb_action
        policy_loss = policy_loss.sum(0) / denom
        entropy_loss = entropy_loss.sum(0) / denom

        losses = {
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss
        }
        metrics = {}
        return losses, metrics

    def _compute_returns_advantages(
        self, values, estimated_value, rewards, terminals
    ):
        if self.gae:
            gae = 0.
            gae_advantages = []

        next_value = estimated_value
        # First step of nstep reward target is estimated value of t+1
        target_return = estimated_value
        nstep_target_returns = []
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            terminal = terminals[i]

            # Nstep return is always calculated for the critic's target
            # using the GAE target for the critic results in the
            # same or worse performance
            target_return = reward + self.discount * target_return * terminal
            nstep_target_returns.append(target_return)

            # Generalized Advantage Estimation
            if self.gae:
                delta_t = reward \
                          + self.discount * next_value * terminal \
                          - values[i].data
                gae = gae * self.discount * self.tau * terminal + delta_t
                gae_advantages.append(gae)
                next_value = values[i].data

        # reverse lists
        nstep_target_returns = torch.stack(
            list(reversed(nstep_target_returns))
        ).data

        if self.gae:
            advantages = torch.stack(list(reversed(gae_advantages))).data
        else:
            advantages = nstep_target_returns - values.data

        return nstep_target_returns, advantages
