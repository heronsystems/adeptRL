# Copyright (C) 2019 Heron Systems, Inc.
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
from torch.nn import functional as F
import torchvision.utils as vutils


from adept.utils import listd_to_dlist
from adept.agents.dqn import OnlineQRDDQN


class I2A(OnlineQRDDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_cache['actions'] = []
        self.exp_cache['lstm_out'] = []

    def _act_gym(self, obs):
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals, ret_lstm=True
        )
        lstm_output = predictions['lstm_out']
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

        values = torch.cat(values, dim=1)
        one_hot_action = torch.zeros(self._nb_env, self.action_space[key][0], device=self.device)
        one_hot_action = one_hot_action.scatter_(1, action, 1)
        self.exp_cache.write_forward(values=values, actions=one_hot_action, lstm_out=lstm_output)
        self.internals = internals
        return actions

    def compute_loss(self, rollouts, next_obs):
        # forward of upsample
        lstm_outs = torch.stack(rollouts.lstm_out).view(self.nb_rollout * self._nb_env, -1)
        actions = torch.stack(rollouts.actions).view(self.nb_rollout * self._nb_env, -1)
        predicted_next_obs = self.network.pred_next(lstm_outs, actions)
        # autoencoder loss
        # next states as categorical label
        states_list = listd_to_dlist(rollouts.states)[self.network._obs_key]
        next_states = states_list[1:] + [next_obs[self.network._obs_key]]
        next_states = torch.stack(next_states)
        # next_states is nb_rollout * nb_env * 84 * 84
        next_states_flat = next_states.view(-1).to(self.device).long()

        # convert predictions to [nb_rollout * nb_env * 84 * 84, 255]
        predicted_next_obs_flat = predicted_next_obs.permute(0, 2, 3, 1).contiguous()
        predicted_next_obs_flat = predicted_next_obs.view(-1, 255)
        autoencoder_loss = F.cross_entropy(predicted_next_obs_flat, next_states_flat, reduction='none')
        # don't predict next state for terminal 
        terminal_mask = torch.stack(rollouts.terminals).unsqueeze(-1)
        autoencoder_loss = autoencoder_loss.view(self.nb_rollout, self._nb_env, -1) * terminal_mask

        # q value loss
        self._possible_update_target()

        # estimate value of next state
        last_values = self._compute_estimated_values(next_obs, self.internals)

        # compute nstep return and advantage over batch
        batch_values = torch.stack(rollouts.values)
        value_targets = self._compute_returns_advantages(last_values, rollouts.rewards, rollouts.terminals)

        # batched q loss
        value_loss = self._loss_fn(batch_values, value_targets)

        # policy distil loss

        # predicted_next_obs to image
        predicted_next_obs = predicted_next_obs.view(self.nb_rollout, self._nb_env, 255, 84, 84)[:5, 0].argmax(dim=1, keepdim=True)
        autoencoder_img = torch.cat([predicted_next_obs, next_states[:5, 0].long()], 0)
        autoencoder_img = vutils.make_grid(autoencoder_img, nrow=5)
        losses = {'value_loss': value_loss.mean(), 'autoencoder_loss': autoencoder_loss.mean()}
        metrics = {'autoencoder_img': autoencoder_img}
        return losses, metrics

