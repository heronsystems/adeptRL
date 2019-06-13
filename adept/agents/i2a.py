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


class I2A(OnlineQRDDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_cache['actions'] = []
        self.exp_cache['imag_lstm'] = []
        self.exp_cache['imag_conv'] = []
        self.exp_cache['imag_qs'] = []
        self.exp_cache['imag_states'] = []
        self.exp_cache['internals'] = []
        self.ssim = SSIM(1, self.device)

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
        self.exp_cache.write_forward(values=values, actions=one_hot_action, imag_conv=predictions['imag_conv'],
                                     imag_states=predictions['imag_states'],
                                     imag_lstm=predictions['imag_lstm'], imag_qs=predictions['imag_qs'], internals=self.internals)
        self.internals = internals
        return actions

    def compute_loss(self, rollouts, next_obs):
        # qvals from policy
        batch_values = torch.stack(rollouts.values)

        # states, actions, terminals to tensors
        states_list = listd_to_dlist(rollouts.states)[self.network._obs_key]
        next_states = states_list[1:] + [next_obs[self.network._obs_key]]
        next_states = torch.stack(next_states).to(self.device).float() / 255.0
        terminal_mask = torch.stack(rollouts.terminals)
        actions = torch.stack(rollouts.actions)

        # predict_sequence
        # first_state = rollouts.states[0][self.network._obs_key].to(self.device).float() / 255.0
        # max_seq = min(math.ceil(self._act_count / (200000 / self._nb_env)), 5)
        # predicted_qvals, predicted_next_obs, predicted_reward = self.network.pred_next(first_state, rollouts.internals[0], actions, terminal_mask, max_seq)
        # next_states = next_states[0:max_seq]
        # terminal_mask = terminal_mask[0:max_seq]

        # predict next state only
        # forward of upsample
        imag_conv = torch.stack(rollouts.imag_conv)
        imag_conv = imag_conv.view(self.nb_rollout * self._nb_env, *imag_conv.shape[2:])
        imag_lstm = torch.stack(rollouts.imag_lstm)
        imag_lstm = imag_lstm.view(self.nb_rollout * self._nb_env, *imag_lstm.shape[2:])
        actions = torch.stack(rollouts.actions).view(self.nb_rollout * self._nb_env, -1)
        predicted_next_obs, predicted_reward = self.network.pred_next_from_action(imag_conv, imag_lstm, actions)
        predicted_next_obs = predicted_next_obs.view(self.nb_rollout, self._nb_env, 1, 84, 84)

        # distil policy loss same as qloss but between distil and current policy
        imag_qs = torch.stack(rollouts.imag_qs)
        imag_qs = imag_qs.view(self.nb_rollout * self._nb_env, *imag_qs.shape[2:])
        action_select = actions.argmax(-1, keepdim=True)
        action_select = action_select.unsqueeze(-1).expand(action_select.shape[0], 1, imag_qs.shape[-1]).long()
        imag_qs_a = imag_qs.gather(1, action_select).squeeze(1)
        imag_qs_a = imag_qs_a.view(self.nb_rollout, self._nb_env, -1)

        distil_loss = self._loss_fn(imag_qs_a, batch_values.detach())

        # autoencoder loss
        # ssim loss
        autoencoder_loss = 1 - self.ssim(predicted_next_obs.view(-1, *predicted_next_obs.shape[2:]),
                                         next_states.view(-1, *next_states.shape[2:]), reduction='none').mean(-1).mean(-1)
        autoencoder_loss = autoencoder_loss.view(-1, self._nb_env) * terminal_mask
        # mae loss
        autoencoder_mse_loss = F.l1_loss(predicted_next_obs.view(-1, *predicted_next_obs.shape[2:]),
                                         next_states.view(-1, *next_states.shape[2:]), reduction='none').mean(-1).mean(-1)
        autoencoder_mse_loss = autoencoder_mse_loss.view(-1, self._nb_env) * terminal_mask
        autoencoder_loss = autoencoder_loss * 0.9 + autoencoder_mse_loss * 0.1
        # end autoencoder_loss

        # reward loss huber TODO: probably classification to see if there is a reward, then another
        # head to predict the value of it
        rewards = torch.stack(rollouts.rewards)
        predicted_reward = self._inverse_scale(predicted_reward.view(-1, self._nb_env))
        reward_loss = F.smooth_l1_loss(predicted_reward, rewards)

        # cross_entropy loss
        # next_states = torch.stack(next_states).to(self.device).long()
        # # next_states is nb_rollout * nb_env * 84 * 84
        # next_states_flat = next_states.view(-1)

        # # convert predictions to [nb_rollout * nb_env * 84 * 84, 255]
        # predicted_next_obs_cont = predicted_next_obs.permute(0, 2, 3, 1).contiguous()
        # predicted_next_obs_flat = predicted_next_obs_cont.view(-1, 255)
        # autoencoder_loss = F.cross_entropy(predicted_next_obs_flat, next_states_flat, reduction='none')
        # # don't predict next state for terminal 
        # terminal_mask = terminal_mask.unsqueeze(-1)
        # autoencoder_loss = autoencoder_loss.view(self.nb_rollout, self._nb_env, -1) * terminal_mask

        # q value loss
        self._possible_update_target()

        # estimate value of next state
        last_values = self._compute_estimated_values(next_obs, self.internals)

        # compute nstep return and advantage over batch
        value_targets = self._compute_returns_advantages(last_values, rollouts.rewards, rollouts.terminals)

        # batched q loss
        value_loss = self._loss_fn(batch_values, value_targets)

        # imagination rollout
        imag_rollout = rollouts.imag_states[0][:, 0]
        imag_rollout_view = torch.cat([next_states[0, 0:1], imag_rollout], dim=0)
        imag_rollout_view = vutils.make_grid(imag_rollout_view, nrow=5)
        # predicted_next_obs to image
        autoencoder_img = torch.cat([predicted_next_obs[:, 0], next_states[:, 0]], 0)
        autoencoder_img = vutils.make_grid(autoencoder_img, nrow=5)
        losses = {
            'value_loss': value_loss.mean(),
            'autoencoder_loss': autoencoder_loss.mean(),
            'reward_pred_loss': reward_loss.mean(),
            'distil_loss': distil_loss.mean()
        }
        metrics = {'autoencoder_img': autoencoder_img, 'imag_rollout': imag_rollout_view}
        return losses, metrics


class SSIM:
    def __init__(self, channels, device, kernel_size=11, kernel_sigma=1.5):
        self.channels = channels
        self.window = self.create_window(channels, kernel_size, kernel_sigma).to(device)

    def __call__(self, inputs, targets, reduction='mean'):
        """
        Assumes float in range 0, 1
        """
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        # from https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/image_ops_impl.py#L2609
        # luminance
        mu1 = F.conv2d(inputs, self.window, groups=self.channels)
        mu2 = F.conv2d(targets, self.window, groups=self.channels)
        num0 = mu1 * mu2 * 2.0
        den0 = mu1 ** 2 + mu2 ** 2
        luminance = (num0 + c1) / (den0 + c1)

        # contrast structure
        num1 = F.conv2d(inputs * targets, self.window, groups=self.channels) * 2.0
        den1 = F.conv2d((inputs ** 2) + (targets ** 2), self.window, groups=self.channels)
        cs = (num1 - num0 + c2) / (den1 - den0 + c2)

        loss = luminance * cs

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

    # from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    @staticmethod
    def create_window(channels, kernel_size, sigma, dim=2):
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        return kernel

