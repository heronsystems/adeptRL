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
import math
import torch
from torch.nn import functional as F
import torchvision.utils as vutils


from adept.utils import listd_to_dlist
from adept.agents.dqn import OnlineQRDDQN


class I2AEmbed(OnlineQRDDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_cache['actions'] = []
        self.exp_cache['imag_encoded'] = []
        self.exp_cache['imag_qs'] = []
        self.exp_cache['imag_obs'] = []
        self.exp_cache['internals'] = []

    def _act_gym(self, obs):
        predictions, internals = self.network(
            self.gpu_preprocessor(obs, self.device), self.internals, ret_imag=True
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

        values = torch.cat(values, dim=1)
        one_hot_action = torch.zeros(self._nb_env, self.action_space[key][0], device=self.device)
        one_hot_action = one_hot_action.scatter_(1, action, 1)
        self.exp_cache.write_forward(values=values, actions=one_hot_action, imag_encoded=predictions['imag_encoded'],
                                     imag_qs=predictions['imag_qs'], internals=self.internals)
        self.internals = internals
        return actions

    def compute_loss(self, rollouts, next_obs):
        # qvals from policy
        batch_values = torch.stack(rollouts.values)

        # q value loss
        self._possible_update_target()

        # estimate value of next state
        last_values = self._compute_estimated_values(next_obs, self.internals)

        # compute nstep return and advantage over batch
        value_targets = self._compute_returns_advantages(last_values, rollouts.rewards, rollouts.terminals)

        # batched q loss
        value_loss = self._loss_fn(batch_values, value_targets)

        # actions, terminals to tensors
        terminal_mask = torch.stack(rollouts.terminals)
        actions = torch.stack(rollouts.actions)

        # predict next embed only
        imag_encoded = torch.stack(rollouts.imag_encoded)
        imag_encoded = imag_encoded.view(self.nb_rollout * self._nb_env, *imag_encoded.shape[2:])
        actions = torch.stack(rollouts.actions).view(self.nb_rollout * self._nb_env, -1)
        predicted_next_embed, predicted_reward = self.network.pred_next_from_action(imag_encoded, actions)

        # distil policy loss same as qloss but between distil and current policy
        imag_qs = torch.stack(rollouts.imag_qs)
        imag_qs = imag_qs.view(self.nb_rollout * self._nb_env, *imag_qs.shape[2:])
        actions_argmax = actions.argmax(-1, keepdim=True)
        action_select = actions_argmax.unsqueeze(-1).expand(actions_argmax.shape[0], 1, imag_qs.shape[-1]).long()
        imag_qs_a = imag_qs.gather(1, action_select).squeeze(1)
        imag_qs_a = imag_qs_a.view(self.nb_rollout, self._nb_env, -1)
        distil_loss = self._loss_fn(imag_qs_a, batch_values.detach())

        # imagined policy accuracy
        with torch.no_grad():
            imag_action = imag_qs.mean(-1).argmax(-1, keepdim=True)
            imag_policy_accuracy = (imag_action == actions_argmax).view(self.nb_rollout, self._nb_env)[:, -1].cpu()
            imag_policy_accuracy = imag_policy_accuracy.sum() / self.nb_rollout

        # predict next embedding loss
        # mse loss, have to chop off last
        predicted_next_embed_flat = predicted_next_embed[:-1*self._nb_env].view((self.nb_rollout-1)*self._nb_env, -1)
        imag_encoded_targ = imag_encoded[:-1*self._nb_env].detach()
        pred_mse_loss = 0.5 * torch.mean((predicted_next_embed_flat - imag_encoded_targ)**2, dim=1)
        pred_mse_loss = pred_mse_loss.view(-1, self._nb_env) * terminal_mask[:-1]
        # end pred_next_loss

        # reward loss huber TODO: probably classification to see if there is a reward, then another
        # head to predict the value of it
        rewards = torch.stack(rollouts.rewards)
        predicted_reward = predicted_reward.view(-1, self._nb_env)
        reward_loss = 0.5 * torch.mean((predicted_reward - self._scale(rewards)) ** 2)

        losses = {
            'value_loss': value_loss.mean(),
            'embed_pred_loss': pred_mse_loss.mean(),
            'reward_pred_loss': reward_loss.mean(),
            'distil_loss': distil_loss.mean()
        }
        metrics = {'distil_policy_accuracy': imag_policy_accuracy}
        return losses, metrics

