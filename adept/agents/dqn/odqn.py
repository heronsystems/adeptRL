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

    def _get_rollout_values(self, q_vals, action, batch_size=0):
        return q_vals.gather(1, action)

    def _write_exp_cache(self, values, actions):
        values = torch.cat(values, dim=1)
        self.exp_cache.write_forward(values=values)

    def compute_loss(self, rollouts, next_obs):
        self._possible_update_target()

        # estimate value of next state
        last_values = self._compute_estimated_values(next_obs)

        # compute nstep return and advantage over batch
        batch_values = torch.stack(rollouts.values)
        value_targets = self._compute_returns_advantages(last_values, rollouts.rewards, rollouts.terminals)

        # batched loss
        value_loss = self._loss_fn(batch_values, value_targets)

        losses = {'value_loss': value_loss}
        metrics = {}
        return losses, metrics

