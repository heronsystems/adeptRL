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


class ActorCritic(Agent, EnvBase):
    def __init__(self, network, device, reward_normalizer, gpu_preprocessor, nb_env, nb_rollout,
                 discount, gae, tau, normalize_advantage, entropy_weight=0.01):
        self.discount, self.gae, self.tau = discount, gae, tau
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight
        self.gpu_preprocessor = gpu_preprocessor

        self._network = network.to(device)
        self._exp_cache = RolloutCache(nb_rollout, device, reward_normalizer, ['values', 'log_probs', 'entropies'])
        self._internals = listd_to_dlist([self.network.new_internals(device) for _ in range(nb_env)])
        self._device = device
        self.network.train()

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
        rollout_len = len(r.rewards)

        # compute nstep return over batch
        next_values = last_values
        batch_values = torch.stack(r.values)
        if self.gae:
            gae = 0.

        nstep_target_returns = []
        for i in reversed(range(rollout_len)):
            rewards = r.rewards[i]
            terminals = r.terminals[i]

            # Generalized Advantage Estimation
            if self.gae:
                delta_t = rewards + self.discount * next_values * terminals - batch_values[i].data
                gae = gae * self.discount * self.tau * terminals + delta_t
                gae_target_returns = gae + batch_values[i].data
                nstep_target_returns.append(gae_target_returns)
                next_values = batch_values[i].data
            # Nstep return
            else:
                # First step of nstep reward target is estimated value of t+1
                if i == rollout_len - 1:
                    target_returns = next_values
                target_returns = rewards + self.discount * target_returns * terminals
                nstep_target_returns.append(target_returns)

        nstep_target_returns = torch.stack(list(reversed(nstep_target_returns))).data
        # batch advantage
        batch_advantages = nstep_target_returns - batch_values
        value_loss = 0.5 * torch.mean(batch_advantages.pow(2))

        # normalized advantage
        if self.normalize_advantage:
            batch_advantages = (batch_advantages - batch_advantages.mean()) / \
                               (batch_advantages.std() + 1e-5)
        policy_loss = 0.
        entropy_loss = 0.

        for i in range(rollout_len):
            log_probs = r.log_probs[i]
            entropies = r.entropies[i]

            if isinstance(log_probs, dict):
                for k in log_probs.keys():
                    policy_loss = policy_loss - (log_probs[k] * batch_advantages[i].data)
                    entropy_loss = entropy_loss - self.entropy_weight * entropies[k]
            else:
                # expand gae dim for broadcasting if there are multiple channels of log_probs / entropies (SC2)
                if log_probs.dim() == 2:
                    policy_loss = policy_loss - (log_probs * batch_advantages[i].unsqueeze(1).data).sum(1)
                    entropy_loss = entropy_loss - (self.entropy_weight * entropies).sum(1)
                else:
                    policy_loss = policy_loss - log_probs * batch_advantages[i].data
                    entropy_loss = entropy_loss - self.entropy_weight * entropies

        policy_loss = torch.mean(policy_loss / rollout_len)
        entropy_loss = torch.mean(entropy_loss / rollout_len)
        losses = {'value_loss': value_loss, 'policy_loss': policy_loss, 'entropy_loss': entropy_loss}
        metrics = {}
        return losses, metrics
