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
import torch

from adept.utils import listd_to_dlist
from collections import OrderedDict
from adept.expcaches.rollout import RolloutCache
from adept.agents.dqn import BaseDQN


class OnlineDQN(BaseDQN):
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
            ['values']
        )
        self._target_internals = listd_to_dlist(
            [self._target_net.new_internals(device) for _ in range(nb_env)]
        )

    def _get_rollout_values(self, q_vals, action, batch_size=0):
        return q_vals.gather(1, action)

    def _write_exp_cache(self, values, actions):
        values = torch.cat(values, dim=1)
        self.exp_cache.write_forward(values=values)

    def _act_gym(self, obs):
        obs_on_device = self.gpu_preprocessor(obs, self.device)
        predictions, internals = self.network(
            obs_on_device, self.internals
        )
        with torch.no_grad():
            _, targ_internals = self._target_net(
                obs_on_device, self._target_internals
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
        self._target_internals = targ_internals
        return actions

    def _reset_internals_at_index(self, env_idx):
        for (k, v), (tk, tv) in zip(self.network.new_internals(self.device).items(),
                                    self._target_net.new_internals(self.device).items()):
            self.internals[k][env_idx] = v
            self._target_internals[k][env_idx] = tv

    def compute_loss(self, rollouts, next_obs):
        self._possible_update_target()

        # estimate value of next state
        last_values = self._compute_estimated_values(next_obs, self._target_internals)

        # compute nstep return and advantage over batch
        batch_values = torch.stack(rollouts.values)
        value_targets = self._compute_returns_advantages(last_values, rollouts.rewards, rollouts.terminals)

        # batched loss
        value_loss = self._loss_fn(batch_values, value_targets)

        losses = {'value_loss': value_loss.mean()}
        metrics = {}
        return losses, metrics

