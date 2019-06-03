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

import torch
from torch.nn import functional as F
import numpy as np

from adept.expcaches.replay import ExperienceReplay
from adept.agents.agent_module import AgentModule
from adept.utils import listd_to_dlist, dlist_to_listd


class BaseDQN(AgentModule):
    args = {
        'nb_rollout': 20,
        'discount': 0.99,
        'target_copy_steps': 2500,
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
        target_copy_steps,
        double_dqn
    ):
        super(BaseDQN, self).__init__(
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            engine,
            action_space,
            nb_env
        )
        self.nb_rollout = nb_rollout
        self.discount = discount
        self.double_dqn = double_dqn
        self.target_copy_steps = target_copy_steps / nb_env
        self._next_target_copy = self.target_copy_steps
        self._target_net = deepcopy(network)
        self._target_net.eval()
        self._act_count = 0
        self._action_keys = list(sorted(action_space.keys()))

        # use ape-x epsilon
        self.epsilon = 0.4 ** (1+((torch.arange(nb_env, dtype=torch.float, requires_grad=False) / (nb_env - 1)) * 7))

    @classmethod
    def from_args(
        cls, args, network, device, reward_normalizer, gpu_preprocessor, engine,
        action_space, nb_env=None
    ):
        if nb_env is None:
            nb_env = args.nb_env

        # if running in distrib mode, divide by number of processes
        denom = 1
        if hasattr(args, 'nb_proc') and args.nb_proc is not None:
            denom = args.nb_proc

        return cls(
            network, device, reward_normalizer, gpu_preprocessor, engine,
            action_space,
            nb_env=nb_env,
            nb_rollout=args.nb_rollout,
            discount=args.discount,
            target_copy_steps=args.target_copy_steps / denom,
            double_dqn=args.double_dqn
        )

    @property
    def exp_cache(self):
        return self._exp_cache

    @staticmethod
    def output_space(action_space, args=None):
        head_dict = {**action_space}
        return head_dict

    def _prepare_act(self):
        self.network.train()
        self._act_count += 1

    def act(self, obs):
        self._prepare_act()
        return self._act_gym(obs)

    def _act_gym(self, obs):
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals
        )
        q_vals = self._get_qvals_from_pred(predictions)
        batch_size = predictions[self._action_keys[0]].shape[0]

        # reduce feature dim, build action_key dim
        actions = OrderedDict()
        values = []
        # TODO support multi-dimensional action spaces?
        for key in self._action_keys:
            # random action across some environments based on the actors epsilon
            rand_mask = (self.epsilon > torch.rand(batch_size)).nonzero().squeeze(-1)
            action = self._action_from_q_vals(q_vals[key])
            rand_act = torch.randint(self.action_space[key][0], (rand_mask.shape[0], 1), dtype=torch.long).to(self.device)
            action[rand_mask] = rand_act
            actions[key] = action.squeeze(1).cpu().numpy()

            values.append(self._get_rollout_values(q_vals[key], action, batch_size))

        self._write_exp_cache(values, actions)
        self.internals = internals
        return actions

    def act_eval(self, obs):
        self.network.eval()
        return self._act_eval_gym(obs)

    def _act_eval_gym(self, obs):
        with torch.no_grad():
            predictions, internals = self.network(
                self.gpu_preprocessor(obs, self.device), self.internals
            )

            # reduce feature dim, build action_key dim
            actions = OrderedDict()
            for key in self._action_keys:
                # TODO: some amount of egreedy
                action = self._action_from_q_vals(predictions[key])
                actions[key] = action.cpu().numpy()

        self.internals = internals
        return actions

        raise NotImplementedError()

    def _batch_forward(self, obs, sampled_actions, internals, terminals):
        """
        This is the method to recompute the forward pass on the host, it
        must return values. Obs, sampled_actions,
        terminal_masks here are [seq, batch], internals must be reset if
        terminal
        """
        self.network.train()
        values = []

        # numpy to vectorize check for terminals
        terminal_masks = terminals.numpy()

        # if network is modular,
        # trunk can be sped up by combining batch & seq dim
        def get_results_generator():
            torch_obs_dict = {k: torch.stack(v) for k, v in listd_to_dlist(obs).items()}
            obs_on_device = dlist_to_listd(self.gpu_preprocessor(torch_obs_dict, self.device))

            def get_results(seq_ind, internals):
                obs_of_seq_ind = obs_on_device[seq_ind]
                return self.network(obs_of_seq_ind, internals)

            return get_results

        result_fn = get_results_generator()
        for seq_ind in range(terminal_masks.shape[0]):
            results, internals = result_fn(seq_ind, internals)

            qvals = self._get_qvals_from_pred_sampled(results, sampled_actions[seq_ind])
            # seq lists
            values.append(qvals)

            # if this state was terminal reset internals
            terminals = np.where(terminal_masks[seq_ind] == 0)[0]
            for batch_ind in terminals:
                reset_internals = self.network.new_internals(self.device)
                for k, v in reset_internals.items():
                    internals[k][batch_ind] = v

        return torch.stack(values), internals

    def _compute_estimated_values(self, next_obs):
        # estimate value of next state
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            results, _ = self._target_net(next_obs_on_device, self.internals)
            target_q = self._get_qvals_from_pred(results)

            # if double dqn estimate get target val for current estimated action
            if self.double_dqn:
                current_results, _ = self.network(next_obs_on_device, self.internals)
                current_q = self._get_qvals_from_pred(current_results)
                last_actions = [self._action_from_q_vals(current_q[k]) for k in self._action_keys]
                last_values = torch.stack([target_q[k].gather(1, a)[:, 0].data for k, a in zip(self._action_keys, last_actions)], dim=1)
            else:
                last_values = torch.stack([torch.max(target_q[k], 1)[0].data for k in self._action_keys], dim=1)

        return last_values

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
            target_return = self._scale(reward + self.discount * self._inverse_scale(target_return) * terminal)
            nstep_target_returns.append(target_return)

        # reverse lists
        nstep_target_returns = torch.stack(list(reversed(nstep_target_returns))).data
        return nstep_target_returns

    def _possible_update_target(self):
        # copy target network
        if self._act_count > self._next_target_copy:
            self._target_net = deepcopy(self.network)
            self._target_net.eval()
            self._next_target_copy += self.target_copy_steps

    def _action_from_q_vals(self, q_vals):
        return q_vals.argmax(dim=-1, keepdim=True)

    def _get_qvals_from_pred(self, predictions):
        return predictions

    def _get_qvals_from_pred_sampled(self, predictions, actions):
        qvals = []
        # TODO support multi-dimensional action spaces?
        for key in predictions.keys():
            vals = predictions[key]
            qvals.append(vals.gather(1, actions[key].unsqueeze(1)))

        return torch.cat(qvals, dim=1)

    def _loss_fn(self, batch_values, value_targets):
        return 0.5 * (value_targets - batch_values).pow(2)
        # return F.smooth_l1_loss(batch_values, value_targets, reduction='none')

    def _scale(self, x, SCALE=10**-3):
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + SCALE*x

    def _inverse_scale(self, x, SCALE=10**-3):
        return torch.sign(x) * ((((torch.sqrt(1+4*SCALE*(torch.abs(x)+1+SCALE))-1)/(2*SCALE))**2)-1)

