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
from adept.agents.agent_module import AgentModule


class DistDQN(AgentModule):
    args = {
        'nb_rollout': 20,
        'discount': 0.99,
        'egreedy_final': 0.1,
        'egreedy_steps': 1000000,
        'target_copy_steps': 10000,
        'double_dqn': True
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
        egreedy_final,
        egreedy_steps,
        target_copy_steps,
        double_dqn
    ):
        super(DistDQN, self).__init__(
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            engine,
            action_space,
            nb_env
        )
        self.discount, self.egreedy_steps, self.egreedy_final = discount, egreedy_steps / nb_env, egreedy_final
        self.double_dqn = double_dqn
        self.target_copy_steps = target_copy_steps
        self._next_target_copy = self.target_copy_steps
        self._target_net = deepcopy(network)
        self._target_net.eval()
        self._act_count = 0

        self._exp_cache = RolloutCache(
            nb_rollout, device, reward_normalizer,
            ['sampled_action']
        )
        self._action_keys = list(sorted(action_space.keys()))

    @classmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor, engine,
        action_space, nb_env=None
    ):
        if nb_env is None:
            nb_env = args.nb_env

        denom = 1
        if hasattr(args, 'nb_proc') and args.nb_proc is not None:
            denom = args.nb_proc

        return cls(
            network, device, reward_normalizer, gpu_preprocessor, engine,
            action_space,
            nb_env=nb_env,
            nb_rollout=args.nb_rollout,
            discount=args.discount,
            egreedy_final=args.egreedy_final,
            egreedy_steps=args.egreedy_steps / denom,
            target_copy_steps=args.target_copy_steps,
            double_dqn=args.double_dqn
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @staticmethod
    def output_space(action_space, args=None):
        head_dict = {**action_space}
        return head_dict

    def seq_obs_to_pathways(self, obs, device):
        """
            Converts a dict of sequential observations to a list(of seq len) of dicts
        """
        pathway_dict = self.gpu_preprocessor(obs, device)
        return dlist_to_listd(pathway_dict)

    def act(self, obs):
        self.network.train()
        self._act_count += 1
        return self._act_gym(obs)

    def _act_gym(self, obs):
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals
        )
        batch_size = predictions[self._action_keys[0]].shape[0]

        # reduce feature dim, build action_key dim
        actions = OrderedDict()
        values = []
        compressed_actions = []
        # TODO support multi-dimensional action spaces?
        for key in self._action_keys:
            # possible sample
            if self._act_count < self.egreedy_steps:
                epsilon = 1 - ((1-self.egreedy_final) / self.egreedy_steps) * self._act_count
            else:
                epsilon = self.egreedy_final

            # random action across some environments
            rand_mask = (epsilon > torch.rand(batch_size)).nonzero().squeeze(-1)
            action = predictions[key].argmax(dim=-1, keepdim=True)
            rand_act = torch.randint(self.action_space[key][0], (rand_mask.shape[0], 1), dtype=torch.long).to(self.device)
            action[rand_mask] = rand_act

            actions[key] = action.squeeze(1).cpu().numpy()
            compressed_actions.append(action)

        compressed_actions = torch.cat(compressed_actions, dim=1)

        self.exp_cache.write_forward(sampled_action=compressed_actions)
        self.internals = internals
        return actions

    def act_eval(self, *args, **kwargs):
        pass

    def _compute_returns_advantages(self, estimated_value, rewards, terminals):
        next_value = estimated_value
        # First step of nstep reward target is estimated value of t+1
        target_return = estimated_value
        nstep_target_returns = []
        for i in reversed(range(len(rewards))):
            # unsqueeze over action dim so it isn't broadcasted
            reward = rewards[i].unsqueeze(-1)
            terminal = terminals[i].unsqueeze(-1)

             # Nstep return is always calculated for the critic's target
            target_return = reward + self.discount * target_return * terminal
            nstep_target_returns.append(target_return)

        # reverse lists
        nstep_target_returns = torch.stack(list(reversed(nstep_target_returns))).data
        return nstep_target_returns

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
            qvals = self._predictions_to_qvals_host(
                seq_ind, obs, results, sampled_actions[seq_ind]
            )
            # seq lists
            values.append(qvals)

            # if this state was terminal reset internals
            terminals = np.where(terminal_masks[seq_ind] == 0)[0]
            for batch_ind in terminals:
                reset_internals = self.network.new_internals(self.device)
                for k, v in reset_internals.items():
                    internals[k][batch_ind] = v

        # forward on state t+1
        with torch.no_grad():
            results, _ = self._target_net(next_obs_on_device, internals)
            # if double dqn estimate get target val for current estimated action
            if self.double_dqn:
                current_results, _ = self.network(next_obs_on_device, internals)
                last_actions = [current_results[k].argmax(dim=-1, keepdim=True) for k in self._action_keys]
                last_values = torch.stack([results[k].gather(1, a)[:, 0].data for k, a in zip(self._action_keys, last_actions)], dim=1)
            else:
                last_values = torch.stack([torch.max(results[k], 1)[0].data for k in self._action_keys], dim=1)

        return torch.stack(values), last_values

    def _predictions_to_qvals_host(
        self, seq_ind, obs, predictions, actions_taken
    ):
        return self.__predictions_to_qvals_host_gym(
            predictions, actions_taken
        )

    def __predictions_to_qvals_host_gym(
        self, predictions, actions_taken
    ):
        qvals = []
        # TODO support multi-dimensional action spaces?
        for key_ind, key in enumerate(self._action_keys):
            vals = predictions[key]
            qvals.append(vals.gather(1, actions_taken[:, key_ind].unsqueeze(1)))

        return torch.cat(qvals, dim=1)

    def compute_loss(self, rollouts):
        # rollouts here are a list of [seq, nb_env]
        # cat along the 1 dim gives [seq, batch = nb_env*nb_batches]
        # pull from rollout and convert to tensors of [seq, batch, ...]
        rewards = torch.cat(rollouts['rewards'], 1).to(self.device)
        terminals_mask = torch.cat(rollouts['terminals'], 1)  # cpu
        discount_terminal_mask = (self.discount * terminals_mask).to(self.device)
        states = {
            k.split('-')[-1]: torch.cat(rollouts[k], 1)
            for k, v in rollouts.items() if 'rollout_obs-' in k
        }
        next_states = {
            k.split('-')[-1]:
            torch.cat(rollouts[k],
                      0)  # 0 dim here is batch since next obs has no seq
            for k, v in rollouts.items() if 'next_obs-' in k
        }
        behavior_sampled_action = torch.cat(rollouts['sampled_action'],
                                            1).long().to(self.device)
        # internals are prefixed like internals-
        # they are a list[]
        behavior_starting_internals = {
            # list flattening
            # https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
            k.split('-')[-1]:
            [item.to(self.device) for sublist in v for item in sublist]
            for k, v in rollouts.items() if 'internals' in k
        }

        self._act_count += np.prod(rewards.shape)
        # copy target network
        if self._act_count > self._next_target_copy:
            self._target_net = deepcopy(self.network)
            self._target_net.eval()
            self._next_target_copy += self.target_copy_steps

        # compute current forward
        current_values, estimated_value = self.act_on_host(
            states, next_states, terminals_mask, behavior_sampled_action,
            behavior_starting_internals
        )

        # compute target for current value and advantage
        with torch.no_grad():
            value_target = self._compute_returns_advantages(
                estimated_value, rewards, discount_terminal_mask
            )

        # using torch.no_grad so detach is unnecessary
        value_loss = 0.5 * torch.mean(
            (value_target - current_values).pow(2)
        )

        losses = {'value_loss': value_loss}
        metrics = {}
        return losses, metrics

