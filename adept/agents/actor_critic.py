"""
Copyright (C) 2018 Heron Systems, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from collections import OrderedDict

import torch
from adept.environments import Engines
from torch.nn import functional as F

from adept.expcaches.rollout import RolloutCache
from adept.utils.util import listd_to_dlist
from ._base import Agent


class ActorCritic(Agent):
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
            entropy_weight=0.01
    ):
        self.discount, self.gae, self.tau = discount, gae, tau
        self.entropy_weight = entropy_weight
        self.gpu_preprocessor = gpu_preprocessor
        self.engine = engine

        self._network = network.to(device)
        self._exp_cache = RolloutCache(nb_rollout, device, reward_normalizer, ['values', 'log_probs', 'entropies'])
        self._internals = listd_to_dlist([self.network.new_internals(device) for _ in range(nb_env)])
        self._device = device
        self.action_space = action_space
        self._action_keys = list(sorted(action_space.entries_by_name.keys()))
        self._func_id_idx = None

    @classmethod
    def from_args(cls, network, device, reward_normalizer, gpu_preprocessor, engine, action_space, args):
        return cls(
            network, device, reward_normalizer, gpu_preprocessor, engine, action_space,
            args.nb_env, args.exp_length, args.discount, args.generalized_advantage_estimation, args.tau
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @property
    def network(self):
        return self._network

    @property
    def device(self):
        return self._device

    @property
    def internals(self):
        return self._internals

    @internals.setter
    def internals(self, new_internals):
        self._internals = new_internals

    @staticmethod
    def output_shape(action_space):
        ebn = action_space.entries_by_name
        actor_outputs = {name: entry.shape[0] for name, entry in ebn.items()}
        head_dict = {'critic': 1, **actor_outputs}
        return head_dict

    def act(self, obs):
        self.network.train()

        if self.engine == Engines.GYM:
            return self._act_atari(obs)
        elif self.engine == Engines.SC2:
            return self._act_sc2(obs)
        else:
            raise NotImplementedError()
        # generate the action masks
        # save action masks in exp cache


    def _act_atari(self, obs):
        predictions, internals = self.network(self.gpu_preprocessor(obs, self.device), self.internals)
        values = predictions['critic']

        # reduce feature dim, build action_key dim
        actions = OrderedDict()
        log_probs = []
        entropies = []
        # TODO support multi-dimensional action spaces?
        for key in self._action_keys:
            logit = predictions[key]
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)

            action = prob.multinomial(1)
            log_prob = log_prob.gather(1, action)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            actions[key] = action.squeeze(1).cpu().numpy()
            log_probs.append(log_prob)
            entropies.append(entropy)

        log_probs = torch.cat(log_probs, dim=1)
        entropies = torch.cat(entropies, dim=1)

        self.exp_cache.write_forward(
            values=values,
            log_probs=log_probs,
            entropies=entropies
        )
        self.internals = internals
        return actions

    def _act_sc2(self, obs):
        predictions, internals = self.network(self.gpu_preprocessor(obs, self.device), self.internals)
        values = predictions['critic']

        if self._func_id_idx is None:
            self._func_id_idx = self._action_keys.index('func_id')

        # reduce feature dim, build action_key dim
        actions = OrderedDict()
        log_probs = []
        entropies = []
        action_masks = []
        for key in self._action_keys:
            logit = predictions[key]
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)

            action = prob.multinomial(1)
            log_prob = log_prob.gather(1, action)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            actions[key] = action.squeeze(1).cpu().numpy()
            log_probs.append(log_prob)
            entropies.append(entropy)
            if key == 'func_id':
                action_masks.append(torch.ones_like(entropy))
            else:
                action_masks = torch.zeros_like(entropy)

        log_probs = torch.cat(log_probs, dim=1)
        entropies = torch.cat(entropies, dim=1)

        action_masks = torch.cat(action_masks, dim=1)
        for i in range(actions.shape[0]):
            for j in range(actions.shape[1]):
                pass  # TODO

        # TODO apply mask to log probs and entropies

        self.exp_cache.write_forward(
            values=values,
            log_probs=log_probs,
            entropies=entropies
        )
        self.internals = internals
        return actions

    def act_eval(self, obs):
        self.network.eval()
        with torch.no_grad():
            predictions, internals = self.network(self.gpu_preprocessor(obs, self.device), self.internals)

        # reduce feature dim, build action_key dim
        actions = OrderedDict()
        for key in self._action_keys:
            logit = predictions[key]
            prob = F.softmax(logit, dim=1)
            action = torch.argmax(prob, 1, keepdim=True)
            actions[key] = action.squeeze(1).cpu().numpy()

        self.internals = internals
        return actions

    def preprocess_logits(self, logits):  # TODO delete
        return logits['Discrete']

    def process_logits(self, logits, obs, deterministic):  # TODO delete
        prob = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropies = -(log_probs * prob).sum(1)
        if not deterministic:
            actions = prob.multinomial(1)
        else:
            actions = torch.argmax(prob, 1, keepdim=True)
        log_probs = log_probs.gather(1, actions)

        return actions.squeeze(1).cpu().numpy(), log_probs.squeeze(1), entropies

    def compute_loss(self, rollouts, next_obs):
        # estimate value of next state
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            results, _ = self.network(next_obs_on_device, self.internals)
            last_values = results['critic'].squeeze(1).data

        r = rollouts
        policy_loss = 0.
        value_loss = 0.
        entropy_loss = 0.
        nstep_returns = last_values
        gae = torch.zeros_like(nstep_returns)

        rollout_len = len(r.rewards)
        for i in reversed(range(rollout_len)):
            rewards = r.rewards[i]
            terminals = r.terminals[i]
            values = r.values[i]        # N, F
            log_probs = r.log_probs[i]  # N, F
            entropies = r.entropies[i]  # N, F

            nstep_returns = rewards + self.discount * nstep_returns * terminals
            advantages = nstep_returns.data - values
            value_loss = value_loss + 0.5 * advantages.pow(2)

            # Generalized Advantage Estimation
            if self.gae:
                if i == rollout_len - 1:
                    nxt_values = last_values
                else:
                    nxt_values = r.values[i + 1]
                delta_t = rewards + self.discount * nxt_values.data * terminals - values.data
                gae = gae * self.discount * self.tau * terminals + delta_t
                advantages = gae

            if isinstance(log_probs, dict):
                for k in log_probs.keys():
                    policy_loss = policy_loss - (log_probs[k] * advantages.data)
                    entropy_loss = entropy_loss - self.entropy_weight * entropies[k]
            else:
                # expand gae dim for broadcasting if there are multiple channels of log_probs / entropies (SC2)
                if log_probs.dim() == 2:
                    policy_loss = policy_loss - (log_probs * advantages.unsqueeze(1).data).sum(1)
                    entropy_loss = entropy_loss - (self.entropy_weight * entropies).sum(1)
                else:
                    policy_loss = policy_loss - log_probs * advantages.data
                    entropy_loss = entropy_loss - self.entropy_weight * entropies

        policy_loss = torch.mean(policy_loss / rollout_len)
        entropy_loss = torch.mean(entropy_loss / rollout_len)
        value_loss = 0.5 * torch.mean(value_loss / rollout_len)
        losses = {'value_loss': value_loss, 'policy_loss': policy_loss, 'entropy_loss': entropy_loss}
        metrics = {}
        return losses, metrics
