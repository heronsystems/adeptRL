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

from adept.expcaches.replay import ExperienceReplay
from adept.utils.util import listd_to_dlist
from ._base import Agent, EnvBase


class Acer(Agent, EnvBase):
    def __init__(self, network, device, reward_normalizer, gpu_preprocessor, nb_env, nb_rollout,
                 discount, gae, tau, normalize_advantage, entropy_weight=0.01):
        self.discount, self.gae, self.tau = discount, gae, tau
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight
        self.gpu_preprocessor = gpu_preprocessor

        self._network = network.to(device)
        self._exp_cache = ExperienceReplay(32, nb_rollout, 50000, reward_normalizer, 
                                           ['actions', 'log_probs', 'internals'])
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
        with torch.no_grad():
            results, internals = self.network(self.gpu_preprocessor(obs, self.device), self.internals)
            values = results['critic'].squeeze(1)
            logits = {k: v for k, v in results.items() if k != 'critic'}

            logits = self.preprocess_logits(logits)
            actions, log_probs_all, _ = self.process_logits(logits, obs, deterministic=False)

        self.exp_cache.write_forward(
            log_probs=log_probs_all,
            actions=actions,
            internals=self.internals
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

        return actions.squeeze(1).cpu().numpy(), log_probs, entropies

    def compute_loss(self, rollouts, next_obs):
        # estimate value of next state
        # nextobs should be [batch, ...]
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            results, _ = self.network(next_obs_on_device, self.internals)
            last_Q = results['critic'].squeeze(1).data
            # Qret == V(s)
            last_Qret = (last_Q * last_probs).sum(1) 

        r = rollouts
        actions = r.actions
        terminals = r.terminals
        rewards = r.rewards
        behavior_log_probs = r.log_probs

        # assume that everything is in batch format [seq, bs, ...]
        # recompute forward pass
        current_probs, current_log_probs, current_Q, entropies = self.compute_forward_batch()
        current_log_probs_action = current_log_probs.gather(-1, actions)
        values = (current_probs * current_Q).sum(-1)

        # loop over seq
        Qret = last_Qret
        next_importance = 1.
        batch_size = len(r.rewards[0])
        rollout_len = len(r.rewards)
        for i in range(rollout_len):
            terminal_mask = terminals[i]
            current_Q_action = current_Q[i].gather(1, actions[i])

            # targets and constants have no grad
            with torch.no_grad():
                # importance ratio
                log_diff = current_log_probs[i] - behavior_log_probs[i]
                importance = torch.exp(log_diff)
                importance_action = importance.gather(1, actions[i])
                importance_action_clipped = torch.clamp(importance_action, max=1)

                # discounted Retrace return 
                discount_mask = self.discount * terminal_mask
                delta_Qret = (Qret - current_Q_action) * next_importance
                Qret = rewards[i] + discount_mask * delta_Qret \
                       + discount_mask * next_value

                # advantage
                advantage = Qret - values[i]

            # critic loss
            critic_loss_seq += 0.5 * F.mse_loss(current_Q_action, Qret)

            # on policy loss first half of eq(9)
            policy_loss = -importance_action_clipped * current_log_probs_action[i] * advantage
            policy_loss /= batch_size

            # off policy bias correction last half of eq(9)
            bias_advantage = (current_Q_action - values[i]).data
            bias_weight = F.relu((importance - self.retrace_clip) / importance)
            # sum over actions
            off_policy_loss = (-bias_weight * current_log_probs * bias_advantage).sum(1)
            off_policy_loss /= batch_size

            # sum policy loss over seq
            policy_loss_seq += policy_loss + off_policy_loss

            # variables for next step
            next_value = values[i].data
            next_importance = importance_action_clipped.data

        print(critic_loss_seq.shape, policy_loss_seq.shape)
        critic_loss = critic_loss_seq / rollout_len
        policy_loss = torch.mean(policy_loss_seq / rollout_len)
        entropy_loss = -torch.mean(self.entropy_weight * entropies)
        losses = {'critic_loss': critic_loss, 'policy_loss': policy_loss, 'entropy_loss': entropy_loss}
        metrics = {}
        return losses, metrics

