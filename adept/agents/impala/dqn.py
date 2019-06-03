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
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from adept.expcaches.rollout import RolloutCache
from adept.utils.util import listd_to_dlist, dlist_to_listd
from adept.agents.dqn import BaseDQN


class ActorLearnerDQN(BaseDQN):
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
        target_copy_steps,
        double_dqn
    ):
        super(ActorLearnerDQN, self).__init__(
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            engine,
            action_space,
            nb_env,
            nb_rollout,
            discount,
            target_copy_steps,
            double_dqn
        )

        # base dqn doesn't set exp cache, actor learner only needs actions
        self._exp_cache = RolloutCache(
            nb_rollout, device, reward_normalizer,
            ['actions']
        )

    ### Same overrides as Exp Replay DQN
    def act(self, obs):
        self._prepare_act()
        # exp replay doesn't save grads during forward
        with torch.no_grad():
            return self._act_gym(obs)

    def _get_rollout_values(self, *args, **kwargs):
        return None
    ### End same overrides

    def _write_exp_cache(self, values, actions):
        # Actor Learner container handles sending first internals
        # But need to compress actions to a torch tensor so they can be sent by MPI
        compressed_actions = torch.stack([torch.from_numpy(actions[k]) for k in self._action_keys], dim=1)
        self.exp_cache.write_forward(actions=compressed_actions)

    def seq_obs_to_pathways(self, obs, device):
        """
            Converts a dict of sequential observations to a list(of seq len) of dicts
        """
        pathway_dict = self.gpu_preprocessor(obs, device)
        return dlist_to_listd(pathway_dict)

    def act_on_host(
        self, obs, next_obs, terminal_masks, sampled_actions, internals
    ):
        """
        This is the method to recompute the forward pass on the host, it
        must return values. Obs, sampled_actions,
        terminal_masks here are [seq, batch], internals must be reset if
        terminal
        """
        self.network.train()
        next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)

        values = []

        # numpy to vectorize check for terminals
        terminal_masks = terminal_masks.numpy()

        # if network is modular,
        # trunk can be sped up by combining batch & seq dim
        def get_results_generator():
            obs_on_device = self.seq_obs_to_pathways(obs, self.device)

            def get_results(seq_ind, internals):
                obs_of_seq_ind = obs_on_device[seq_ind]
                return self.network(obs_of_seq_ind, internals)

            return get_results

        result_fn = get_results_generator()
        for seq_ind in range(terminal_masks.shape[0]):
            results, internals = result_fn(seq_ind, internals)
            qvals = self._get_qvals_from_pred_sampled(
                results, sampled_actions[seq_ind]
            )
            # seq lists
            values.append(qvals)

            # if this state was terminal reset internals
            terminals = np.where(terminal_masks[seq_ind] == 0)[0]
            for batch_ind in terminals:
                reset_internals = self.network.new_internals(self.device)
                for k, v in reset_internals.items():
                    internals[k][batch_ind] = v

        return torch.stack(values), internals

    def compute_loss(self, rollouts):
        # TODO: this is a similar process to ActorCriticVtrace maybe some way to merge common
        # rollouts here are a list of [seq, nb_env]
        # cat along the 1 dim gives [seq, batch = nb_env*nb_batches]
        # pull from rollout and convert to tensors of [seq, batch, ...]
        rewards = torch.cat(rollouts['rewards'], 1).to(self.device)
        terminals_mask = torch.cat(rollouts['terminals'], 1)  # cpu
        terminals_mask_gpu = terminals_mask.to(self.device)
        states = {
            k.split('-')[-1]: torch.cat(rollouts[k], 1)
            for k, v in rollouts.items() if 'rollout_obs-' in k
        }
        next_obs = {
            k.split('-')[-1]:
            torch.cat(rollouts[k],
                      0)  # 0 dim here is batch since next obs has no seq
            for k, v in rollouts.items() if 'next_obs-' in k
        }
        rollout_actions = torch.cat(rollouts['actions'],
                                            1).long().to(self.device)
        # convert actions from [seq, batch, actio_ind] to list of dict of tensors
        rollout_actions = torch.unbind(rollout_actions)
        rollout_actions = [{k: x[:, a_ind] for a_ind, k in enumerate(self._action_keys)} for x in rollout_actions]
        # internals are prefixed like internals-
        # they are a list[]
        rollout_internals = {
            # list flattening
            # https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
            k.split('-')[-1]:
            [item.to(self.device) for sublist in v for item in sublist]
            for k, v in rollouts.items() if 'internals' in k
        }
        # end processing 

        # Learner does not act so increment act count here
        self._act_count += np.prod(rewards.shape)

        # TODO: this is copied from DQN with exp replay maybe some way to merge
        self._possible_update_target()

        # recompute forward pass to get value estimates for states
        batch_values, internals = self._batch_forward(states, rollout_actions, rollout_internals, terminals_mask)

        # estimate value of next state
        last_values = self._compute_estimated_values(next_obs, internals)

        # compute nstep return and advantage over batch
        value_targets = self._compute_returns_advantages(last_values, rewards, terminals_mask_gpu)

        # batched loss
        value_loss = self._loss_fn(batch_values, value_targets).squeeze(-1)

        losses = {'value_loss': value_loss.mean()}
        metrics = {}
        return losses, metrics

