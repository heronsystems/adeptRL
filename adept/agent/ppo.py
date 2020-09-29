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
from adept.actor import PPOActorTrain
from adept.actor.base.ac_helper import ACActorHelperMixin
from adept.exp import Rollout
from .base.agent_module import AgentModule
from adept.utils import listd_to_dlist, dlist_to_listd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SequentialSampler


class PPO(AgentModule):
    # PPO Atari defaults https://arxiv.org/abs/1707.06347 pg 10.
    args = {
        'rollout_len': 128,
        'discount': 0.99,
        'normalize_advantage': True,
        'entropy_weight': 0.01,
        'gae_discount': 0.95,
        'rollout_minibatch_len': 32,
        'nb_rollout_epoch': 4,
        'policy_clipping': 0.2
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
            gae_discount,
            rollout_minibatch_len,
            nb_rollout_epoch,
            policy_clipping
    ):
        super().__init__(
            reward_normalizer,
            action_space
        )
        self.discount = discount
        self.normalize_advantage = normalize_advantage
        self.entropy_weight = entropy_weight

        self._exp_cache = Rollout(spec_builder, rollout_len)
        self._actor = PPOActorTrain(action_space)
        self.reward_normalizer = reward_normalizer
        self.gae_discount = gae_discount
        self.rollout_minibatch_len = rollout_minibatch_len
        self.nb_rollout_epoch = nb_rollout_epoch
        self.policy_clipping = policy_clipping

        if rollout_len % rollout_minibatch_len != 0:
            raise ValueError('Rollout length must be divisible by number of minibatches')

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
            gae_discount=args.gae_discount,
            rollout_minibatch_len=args.rollout_minibatch_len,
            nb_rollout_epoch=args.nb_rollout_epoch,
            policy_clipping=args.policy_clipping
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space):
        return PPOActorTrain._exp_spec(
            exp_len, batch_sz, obs_space, act_space, internal_space
        )

    @staticmethod
    def output_space(action_space):
        return PPOActorTrain.output_space(action_space)

    def act(self, *args, **kwargs):
        # PPO recomputes actions so we don't need grads
        with torch.no_grad():
            return super().act(*args, **kwargs)

    def compute_action_exp(self, predictions, internals, obs, available_actions):
        return self._actor.compute_action_exp(predictions, internals, obs, available_actions)

    def learn_step(self, updater, network, next_obs, next_internals):
        r = self.exp_cache.read()
        device = r.rewards[0].device
        rollout_len = self.exp_cache.rollout_len

        # estimate value of next state
        with torch.no_grad():
            pred, _, _ = network(next_obs, next_internals)
            last_values = pred['critic'].squeeze(-1).data

        # calc nsteps
        gae = 0.
        next_values = last_values
        gae_returns = []
        for i in reversed(range(rollout_len)):
            rewards = self.reward_normalizer(r.rewards[i])
            terminal_mask = 1. - r.terminals[i].float()
            current_values = r.values[i].squeeze(-1)
            # generalized advantage estimation
            delta_t = rewards + self.discount * next_values.data * terminal_mask - current_values
            gae = gae * self.discount * self.gae_discount * terminal_mask + delta_t
            gae_returns.append(gae + current_values)
            next_values = current_values.data
        gae_returns = torch.stack(list(reversed(gae_returns))).data

        # Convert to torch tensors of [seq, num_env]
        old_values = torch.stack(r.values).squeeze(-1)
        adv_targets_batch = (gae_returns - old_values).data
        old_log_probs_batch = torch.stack(r.log_probs).data
        # keep a copy of terminals on the cpu it's faster
        rollout_terminals = torch.stack(r.terminals).cpu().numpy()

        # Normalize advantage
        if self.normalize_advantage:
            adv_targets_batch = (adv_targets_batch - adv_targets_batch.mean()) / \
                                (adv_targets_batch.std() + 1e-5)

        for e in range(self.nb_rollout_epoch):
            # setup minibatch iterator
            minibatch_inds = list(BatchSampler(
                SequentialSampler(range(rollout_len)),
                self.rollout_minibatch_len, drop_last=False)
            )
            # randomize sequences to sample NOTE: in-place operation
            np.random.shuffle(minibatch_inds)
            for i in minibatch_inds:
                # TODO: detach internals, no_grad in compute_action_exp takes care of this
                starting_internals = {k: ts[i[0]].unbind(0) for k, ts in r.internals.items()}
                gae_return = gae_returns[i]
                old_log_probs = old_log_probs_batch[i]
                sampled_actions = [r.actions[x] for x in i]
                batch_obs = [r.observations[x] for x in i]
                # needs to be seq, batch, broadcast dim
                adv_targets = adv_targets_batch[i].unsqueeze(-1)
                terminals_batch = rollout_terminals[i]

                # forward pass
                cur_log_probs, cur_values, entropies = self.act_batch(
                    network, batch_obs, terminals_batch, sampled_actions,
                    starting_internals, device
                )
                value_loss = 0.5 * torch.mean((cur_values - gae_return).pow(2))

                # calculate surrogate loss
                surrogate_ratio = torch.exp(cur_log_probs - old_log_probs)
                surrogate_loss = surrogate_ratio * adv_targets
                surrogate_loss_clipped = torch.clamp(
                    surrogate_ratio,
                    min=1 - self.policy_clipping,
                    max=1 + self.policy_clipping
                ) * adv_targets
                policy_loss = torch.mean(-torch.min(surrogate_loss, surrogate_loss_clipped))
                entropy_loss = torch.mean(-self.entropy_weight * entropies)

                losses = {
                    'value_loss': value_loss,
                    'policy_loss': policy_loss,
                    'entropy_loss': entropy_loss
                }
                total_loss = torch.sum(torch.stack(
                    tuple(loss for loss in losses.values())
                ))

                updater.step(total_loss)

        # TODO: metrics: average loss, policy % change, loss over epochs?, value change
        metrics = {'advantage': torch.mean(adv_targets_batch)}
        return losses, metrics

    def act_batch(self, network, batch_obs, batch_terminals, batch_actions, internals, device):
        exp_cache = []

        for obs, actions, terminals in zip(batch_obs, batch_actions, batch_terminals):
            preds, internals, _ = network(obs, internals)
            exp_cache.append(self._process_exp(preds, actions))

            # where returns a single element tuple with the indexes
            terminal_inds = np.where(terminals)[0]
            for i in terminal_inds:
                for k, v in network.new_internals(device).items():
                    internals[k][i] = v

        exp = listd_to_dlist(exp_cache)
        return torch.stack(exp['log_probs']), torch.stack(exp['values']), torch.stack(exp['entropies'])

    def _process_exp(self, preds, sampled_actions):
        values = preds['critic'].squeeze(1)
        log_probs = []
        entropies = []

        for key in self.action_keys:
            logit = ACActorHelperMixin.flatten_logits(preds[key])

            log_softmax, softmax = ACActorHelperMixin.log_softmax(logit), ACActorHelperMixin.softmax(logit)
            entropy = ACActorHelperMixin.entropy(log_softmax, softmax)
            entropies.append(entropy)

            action = sampled_actions[key]
            log_probs.append(ACActorHelperMixin.log_probability(log_softmax, action))

        # we can cat here for whatever reason
        log_probs = torch.cat(log_probs, dim=1)
        entropies = torch.cat(entropies, dim=1)

        return {
            'log_probs': log_probs,
            'entropies': entropies,
            'values': values
        }
