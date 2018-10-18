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
import numpy as np
from adept.expcaches.rollout import RolloutCache
from adept.utils.util import listd_to_dlist
from ._base import Agent, EnvBase
from adept.networks._base import ModularNetwork
from torch.utils.data.sampler import SequentialSampler, BatchSampler


class ActorCriticPPO(Agent, EnvBase):
    def __init__(self, network, device, reward_normalizer, gpu_preprocessor, nb_env, nb_rollout,
                 discount, gae, tau, normalize_advantage, nb_epoch, batch_size, loss_clip):
        self.discount, self.gae, self.tau = discount, gae, tau
        self.normalize_advantage = normalize_advantage
        self.gpu_preprocessor = gpu_preprocessor

        self._network = network.to(device)
        self._exp_cache = RolloutCache(nb_rollout, device, reward_normalizer, ['obs', 'actions', 'log_probs', 'internals', 'values'])
        self._internals = listd_to_dlist([self.network.new_internals(device) for _ in range(nb_env)])
        self._device = device
        self.network.train()
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.loss_clip = loss_clip 

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
        logits = {k: v for k, v in results.items() if k != 'critic'}

        logits = self.preprocess_logits(logits)
        actions, log_probs, entropies = self.process_logits(logits, obs, deterministic=False)

        self.exp_cache.write_forward(
            obs={k: v.clone() for k, v in obs.items()},
            actions=actions,
            log_probs=log_probs,
            internals=self.internals,
            values=results['critic']
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

    def process_logits_from_actions(self, logits, actions):
        prob = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropies = -(log_probs * prob).sum(1)
        log_probs = log_probs.gather(1, actions.unsqueeze(1))

        return log_probs.squeeze(1), entropies

    def act_batch(self, obs, terminal_masks, sampled_actions, internals):
        """
            This is the method to recompute the forward pass during the ppo minibatch, it must return log_probs, values and entropies
            Obs, sampled_actions, terminal_masks here are [seq, batch], internals must be reset if terminal
        """
        values = []
        log_probs_of_action = []
        entropies = []

        seq_len, batch_size = terminal_masks.shape

        # if network is modular, trunk can be sped up by combining batch & seq dim
        def get_results_generator():
            if isinstance(self.network, ModularNetwork):
                pathway_dict = self.gpu_preprocessor(obs, self.device)
                # flatten obs
                flat_obs = {k: v.view(-1, *v.shape[2:]) for k, v in pathway_dict.items()}
                embeddings = self.network.trunk.forward(flat_obs)
                # add back in seq dim
                seq_embeddings = embeddings.view(seq_len, batch_size, embeddings.shape[-1])

                def get_results(seq_ind, internals):
                    embedding = seq_embeddings[seq_ind]
                    pre_result, internals = self.network.body.forward(embedding, internals)
                    return self.network.head.forward(pre_result, internals)
                return get_results
            else:
                obs_on_device = self.seq_obs_to_pathways(obs, self.device)

                def get_results(seq_ind, internals):
                    obs_of_seq_ind = obs_on_device[seq_ind]
                    return self.network(obs_of_seq_ind, internals)
                return get_results

        result_fn = get_results_generator()
        for seq_ind in range(terminal_masks.shape[0]):
            results, internals = result_fn(seq_ind, internals)
            logits_seq = {k: v for k, v in results.items() if k != 'critic'}
            logits_seq = self.preprocess_logits(logits_seq)
            log_probs_action_seq, entropies_seq = self.process_logits_from_actions(logits_seq, sampled_actions[seq_ind])
            # seq lists
            values.append(results['critic'].squeeze(1))
            log_probs_of_action.append(log_probs_action_seq)
            entropies.append(entropies_seq)

            # if this state was terminal reset internals
            for batch_ind, t_mask in enumerate(terminal_masks[seq_ind]):
                if t_mask == 0:
                    reset_internals = self.network.new_internals(self.device)
                    for k, v in reset_internals.items():
                        internals[k][batch_ind] = v

        # TODO: can't stack if dict
        return torch.stack(log_probs_of_action), torch.stack(values), torch.stack(entropies)

    def compute_loss(self, rollouts, next_obs, update_handler):
        r = rollouts
        rollout_len = len(r.rewards)

        # estimate value of next state
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            pred, _ = self.network(next_obs_on_device, self.internals)
            last_values = pred['critic'].squeeze(1).data

        # calc nsteps
        gae = 0.
        next_values = last_values
        gae_returns = []
        for i in reversed(range(rollout_len)):
            rewards = r.rewards[i]
            terminals = r.terminals[i]
            current_values = r.values[i].squeeze(1)
            # generalized advantage estimation
            delta_t = rewards + self.discount * next_values.data * terminals - current_values
            gae = gae * self.discount * self.tau * terminals + delta_t
            gae_returns.append(gae + current_values)
            next_values = current_values.data
        gae_returns = torch.stack(list(reversed(gae_returns))).data

        # Convert to torch tensors of [seq, num_env]
        old_values = torch.stack(r.values).squeeze(-1)
        adv_targets_batch = (gae_returns - old_values).data
        actions_device = torch.from_numpy(np.asarray(r.actions)).to(self.device)
        old_log_probs_batch = torch.stack(r.log_probs).data
        terminals_batch = torch.stack(r.terminals)

        # Normalize advantage
        if self.normalize_advantage:
            adv_targets_batch = (adv_targets_batch - adv_targets_batch.mean()) / \
                                (adv_targets_batch.std() + 1e-5)

        for e in range(self.nb_epoch):
            # setup minibatch iterator
            minibatch_inds = BatchSampler(SequentialSampler(range(rollout_len)), self.batch_size, drop_last=False)
            for i in minibatch_inds:
                starting_internals = self._detach_internals(r.internals[i[0]])
                gae_return = gae_returns[i]
                old_log_probs = old_log_probs_batch[i]
                sampled_actions = actions_device[i]
                adv_targets = adv_targets_batch[i]
                terminal_masks = terminals_batch[i]

                # States are list(dict) select batch and convert to dict(list)
                obs = listd_to_dlist([r.obs[batch_ind] for batch_ind in i])
                # convert to dict(tensors)
                obs = {k: torch.stack(v) for k, v in obs.items()}

                # forward pass
                cur_log_probs, cur_values, entropies = self.act_batch(obs, terminal_masks, sampled_actions,
                                                                      starting_internals)
                value_loss = 0.5 * torch.mean((cur_values - gae_return).pow(2))

                # calculate surrogate loss
                surrogate_ratio = torch.exp(cur_log_probs - old_log_probs)
                surrogate_loss = surrogate_ratio * adv_targets
                surrogate_loss_clipped = torch.clamp(surrogate_ratio, 1 - self.loss_clip,
                                                     1 + self.loss_clip) * adv_targets
                policy_loss = torch.mean(-torch.min(surrogate_loss, surrogate_loss_clipped)) 
                entropy_loss = torch.mean(-0.01 * entropies)

                losses = {'value_loss': value_loss, 'policy_loss': policy_loss, 'entropy_loss':
                          entropy_loss}
                # print('losses', losses)
                total_loss = torch.sum(torch.stack(tuple(loss for loss in losses.values())))
                update_handler.update(total_loss)

        metrics = {'advantage': torch.mean(adv_targets_batch)}
        return losses, metrics
