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
import torch
from torch.nn import functional as F

from adept.expcaches.rollout import RolloutCache
from adept.utils.util import listd_to_dlist
from ._base import Agent, EnvBase


class ActorCriticPPO(Agent, EnvBase):
    def __init__(self, network, device, reward_normalizer, gpu_preprocessor, nb_env, nb_rollout, discount, gae, tau, nb_epoch):
        self.discount, self.gae, self.tau = discount, gae, tau
        self.gpu_preprocessor = gpu_preprocessor

        self._network = network.to(device)
        self._exp_cache = RolloutCache(nb_rollout, device, reward_normalizer, ['obs', 'actions', 'values', 'log_probs', 'entropies'])
        self._internals = listd_to_dlist([self.network.new_internals(device) for _ in range(nb_env)])
        self._device = device
        self.network.train()
        self.epoch_complete = False
        self.log_probs = None
        self._cur_epoch = 0
        self.nb_epoch = nb_epoch

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
        results, internals = self.network(self.gpu_preprocessor(obs, self.device), self.internals)
        values = results['critic'].squeeze(1)
        logits = {k: v for k, v in results.items() if k != 'critic'}

        logits = self.preprocess_logits(logits)
        actions, log_probs, entropies = self.process_logits(logits, obs, deterministic=False)

        self.exp_cache.write_forward(
            obs=obs,
            actions=actions,
            values=values,
            log_probs=log_probs,
            entropies=entropies
        )
        self.internals = internals
        return actions

    def act_eval(self, obs):
        self.network.eval()
        with torch.no_grad():
            results, internals = self.network(self.gpu_preprocessor(obs, self.device), self.internals)
            logits = {k: v for k, v in results.items() if k != 'critic'}

            logits = self.preprocess_logits(logits)
            actions, _, _ = self.process_logits(logits, obs, deterministic=True)
        self.internals = internals
        return actions

    def preprocess_logits(self, logits):
        return logits['Discrete']

    def process_logits(self, logits, obs, deterministic):
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
        nstep_returns = last_values
        gae = torch.zeros_like(nstep_returns)

        rollout_len = len(r.rewards)

        for i in reversed(range(rollout_len)):
            rewards = r.rewards[i]
            terminals = r.terminals[i]
            values = r.values[i]
            log_probs = r.log_probs[i]
            entropies = r.entropies[i]
            obs = r.obs[i]
            actions = r.actions[i]

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

            # expand gae dim for broadcasting if there are multiple channels of log_probs / entropies (SC2)
            if self.log_probs is None:
                self.log_probs = log_probs
                self.internals = self.epoch_internals
            else:
                results, self.internals = self.network(self.gpu_preprocessor(obs, self.device), self.internals)
                logits = {k: v for k, v in results.items() if k != 'critic'}
                logits = self.preprocess_logits(logits)
                prob = F.softmax(logits, dim=1)
                self.log_probs = F.log_softmax(logits, dim=1)
                entropies = -(self.log_probs * prob).sum(1)
                self.log_probs = self.log_probs.gather(1, torch.from_numpy(actions).to(self.log_probs.device).unsqueeze(1))

            if log_probs.dim() == 2:
                # policy_loss = policy_loss - (log_probs * advantages.unsqueeze(1).data + 0.01 * entropies).sum(1)
                raise NotImplementedError
            else:
                # policy_loss = policy_loss - log_probs * advantages.data - 0.01 * entropies

                surrogate_ratio = torch.exp(self.log_probs - log_probs.data)
                surrogate_ratio_clipped = torch.clamp(surrogate_ratio, 0.8, 1.2)
                policy_loss = policy_loss - torch.min(surrogate_ratio, surrogate_ratio_clipped) * advantages.data - 0.01 * entropies

        policy_loss = torch.mean(policy_loss / rollout_len)
        value_loss = 0.5 * torch.mean(value_loss / rollout_len)
        losses = {'value_loss': value_loss, 'policy_loss': policy_loss}
        metrics = {}
        self._cur_epoch += 1
        if self._cur_epoch == self.nb_epoch:
            self.epoch_complete = True
            self.epoch_internals = self.internals
        return losses, metrics
