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
import torch

from adept.expcaches.rollout import RolloutCache
from adept.agents.dqn import OnlineDQN, DQN


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class OnlineIQN(OnlineDQN):
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
        super(OnlineDQN, self).__init__(
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

        # base dqn doesn't set exp cache
        self._exp_cache = RolloutCache(
            nb_rollout, device, reward_normalizer,
            ['values', 'quantiles']
        )

    def _act_gym(self, obs):
        num_samples = 32
        quantiles = torch.FloatTensor(num_samples, self._nb_env).uniform_(0, 1).to(self.device)
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals,
            num_samples=num_samples, quantiles=quantiles
        )
        q_vals = self._get_qvals_from_pred(predictions)
        batch_size = q_vals[self._action_keys[0]].shape[0]

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

            values.append(self._get_rollout_values(q_vals[key], action, batch_size, num_samples))

        self._write_exp_cache(values, quantiles)
        self.internals = internals
        return actions

    def _write_exp_cache(self, values, quantiles):
        values = torch.cat(values, dim=1)
        self.exp_cache.write_forward(values=values, quantiles=quantiles.t())

    def _get_qvals_from_pred(self, pred):
        # put quantiles on last dim
        qvals = {}
        for k, v in pred.items():
            qvals[k] = v.permute(1, 2, 0)
        return qvals

    def _get_rollout_values(self, q_vals, action, batch_size, num_samples):
        action_select = action.unsqueeze(1).expand(batch_size, 1, num_samples)
        return q_vals.gather(1, action_select).squeeze(1)

    def _action_from_q_vals(self, q_vals):
        return q_vals.mean(2).argmax(dim=-1, keepdim=True)

    def compute_loss(self, rollouts, next_obs):
        self._possible_update_target()

        # estimate value of next state
        last_values = self._compute_estimated_values(next_obs, self.internals)

        # compute nstep return and advantage over batch
        batch_values = torch.stack(rollouts.values)
        value_targets = self._compute_returns_advantages(last_values, rollouts.rewards, rollouts.terminals)

        # batched loss
        value_loss = self._loss_fn(batch_values, value_targets, rollouts.quantiles)

        losses = {'value_loss': value_loss.mean()}
        metrics = {}
        return losses, metrics

    def _compute_estimated_values(self, next_obs, internals):
        # estimate value of next state with same number of samples as policy
        num_samples = 32
        with torch.no_grad():
            next_obs_on_device = self.gpu_preprocessor(next_obs, self.device)
            results, _ = self._target_net(next_obs_on_device, internals, num_samples=num_samples)
            target_q = self._get_qvals_from_pred(results)

            last_values = []
            for k in self._action_keys:
                # if double dqn estimate get target val for current estimated action
                if self.double_dqn:
                    current_results, _ = self.network(next_obs_on_device, internals, num_samples=num_samples)
                    current_q = self._get_qvals_from_pred(current_results)
                    action_select = current_q[k].mean(2).argmax(dim=-1, keepdim=True)
                else:
                    action_select = target_q[k].mean(2).argmax(dim=-1, keepdim=True)
                action_select = action_select.unsqueeze(-1).expand(-1, 1, num_samples)
                last_values.append(target_q[k].gather(1, action_select).squeeze(1))

            last_values = torch.cat(last_values, dim=1)
        return last_values

    def _loss_fn(self, batch_values, value_targets, quantiles):
        # Broadcast temporal difference to compare every combination of quantiles
        # This is formula for loss in the Implicit Quantile Networks paper
        diff = value_targets.unsqueeze(3) - batch_values.unsqueeze(2)
        # target quantiles are last dim, broadcast over it
        quantiles = torch.stack(quantiles).unsqueeze(-1)
        dist_mask = torch.abs(quantiles - (diff.detach() < 0).float())
        return (huber(diff) * dist_mask).sum(-1).mean(-1)

