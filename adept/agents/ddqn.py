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

from adept.expcaches.rollout import RolloutCache
from adept.agents.dqn import DQN


class DDQN(DQN):
    @staticmethod
    def output_space(action_space):
        head_dict = {'value': (1, ), **action_space}
        return head_dict

    def _act_gym(self, obs):
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals
        )
        batch_size = predictions[self._action_keys[0]].shape[0]
        q_vals = self._get_qvals_from_pred(predictions)

        # reduce feature dim, build action_key dim
        actions = OrderedDict()
        values = []
        # TODO support multi-dimensional action spaces?
        for key in self._action_keys:
            # possible sample
            if self._act_count < self.egreedy_steps:
                epsilon = 1 - (0.9 / self.egreedy_steps) * self._act_count
            else:
                epsilon = 0.1

            # TODO: if random action, it's random across all envs, make it single
            if epsilon > torch.rand(1):
                action = torch.randint(self.action_space[key][0], (batch_size, 1), dtype=torch.long).to(self.device)
            else:
                action = q_vals[key].argmax(dim=-1, keepdim=True)

            actions[key] = action.squeeze(1).cpu().numpy()
            values.append(q_vals[key].gather(1, action))

        values = torch.cat(values, dim=1)

        self.exp_cache.write_forward(values=values)
        self.internals = internals
        return actions

    def act_eval(self, obs):
        self.network.eval()
        return self._act_eval_gym(obs)

    def _act_eval_gym(self, obs):
        raise NotImplementedError()
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

    def compute_loss(self, rollouts, next_obs):
        # copy target network
        if self._act_count > self._next_target_copy:
            self._target_net = deepcopy(self.network)
            self._target_net.eval()
            self._next_target_copy += self.target_copy_steps

        # estimate value of next state
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            results, _ = self._target_net(next_obs_on_device, self.internals)
            target_q = self._get_qvals_from_pred(results)

        # if double dqn estimate get target val for current estimated action
        if self.double_dqn:
            current_results, _ = self.network(next_obs_on_device, self.internals)
            current_q = self._get_qvals_from_pred(current_results)
            last_actions = [current_q[k].argmax(dim=-1, keepdim=True) for k in self._action_keys]
            last_values = torch.stack([target_q[k].gather(1, a)[:, 0].data for k, a in zip(self._action_keys, last_actions)], dim=1)
        else:
            last_values = torch.stack([torch.max(target_q[k], 1)[0].data for k in self._action_keys], dim=1)

        # compute nstep return and advantage over batch
        batch_values = torch.stack(rollouts.values)
        value_targets = self._compute_returns_advantages(last_values, rollouts.rewards, rollouts.terminals)

        # batched value loss
        value_loss = 0.5 * torch.mean((value_targets - batch_values).pow(2))

        losses = {'value_loss': value_loss}
        metrics = {}
        return losses, metrics

    def _get_qvals_from_pred(self, predictions):
        q = {}
        for k in self._action_keys:
            q[k] = predictions[k] + predictions['value']
        return q
