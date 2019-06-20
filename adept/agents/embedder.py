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


class Embedder(OnlineQRDDQN):
    args = {**OnlineQRDDQN.args}
    args['autoencoder_loss'] = True
    args['reward_pred_loss'] = False
    args['next_embed_pred_loss'] = True

    def __init__(self, *args, **kwargs):
        self._autoencode_loss = kwargs['autoencoder_loss']
        self._reward_pred_loss = kwargs['reward_pred_loss']
        self._next_embed_pred_loss = kwargs['next_embed_pred_loss']
        del kwargs['autoencoder_loss']
        del kwargs['reward_pred_loss']
        del kwargs['next_embed_pred_loss']

        super().__init__(*args, **kwargs)
        self.ssim = SSIM(1, self.device)

        if self._autoencode_loss:
            self.exp_cache['ae_state_pred'] = []
        if self._reward_pred_loss:
            self.exp_cache['predicted_reward'] = []
        if self._next_embed_pred_loss:
            self.exp_cache['predicted_next_embed'] = []
            self.exp_cache['obs_embed'] = []

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
            double_dqn=args.double_dqn,
            num_atoms=args.num_atoms,
            autoencoder_loss=args.autoencoder_loss,
            reward_pred_loss=args.reward_pred_loss,
            next_embed_pred_loss=args.next_embed_pred_loss
        )

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

        exp_cache = {'values': values}
        if self._autoencode_loss:
            exp_cache['ae_state_pred'] = predictions['ae_state_pred']
        if self._reward_pred_loss or self._next_embed_pred_loss:
            one_hot_action = torch.zeros(self._nb_env, self.action_space[key][0], device=self.device)
            one_hot_action = one_hot_action.scatter_(1, action, 1)
        if self._reward_pred_loss:
            predicted_reward = self.network.predict_reward(predictions['encoded_obs'], one_hot_action)
            exp_cache['predicted_reward'] = predicted_reward.squeeze(-1)
        if self._next_embed_pred_loss:
            predicted_next_embed = self.network.predict_next_embed(predictions['encoded_obs'], one_hot_action)
            exp_cache['predicted_next_embed'] = predicted_next_embed
            exp_cache['obs_embed'] = predictions['encoded_obs']

        self.exp_cache.write_forward(**exp_cache)
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
        losses = {'value_loss': value_loss.mean()}
        metrics = {}

        def view(tensor):
            return tensor.view(-1, *tensor.shape[2:])

        def image_loss(im1, im2):
            autoencoder_loss = 1 - self.ssim(im1, im2).mean(-1).mean(-1)
            # mae loss
            autoencoder_mse_loss = F.l1_loss(im1, im2).mean(-1).mean(-1)
            return autoencoder_loss * 0.9 + autoencoder_mse_loss * 0.1

        if self._autoencode_loss:
            states_list = listd_to_dlist(rollouts.states)[self.network._obs_key]
            states_tensor = view(torch.stack(states_list))
            states_tensor = states_tensor.to(self.device).float() / 255.0
            states_pred = view(torch.stack(rollouts.ae_state_pred))
            losses['ae_loss'] = image_loss(states_pred, states_tensor).mean()

            # show autoencode images
            autoencoder_img = torch.cat([states_pred[-5:], states_tensor[-5:]], 0)
            autoencoder_img = vutils.make_grid(autoencoder_img, nrow=5)
            metrics['ae_state'] = autoencoder_img

        if self._reward_pred_loss:
            # reward loss huber TODO: probably classification to see if there is a reward, then another
            # head to predict the value of it
            rewards = torch.stack(rollouts.rewards)
            predicted_reward = torch.stack(rollouts.predicted_reward)
            reward_loss = 0.5 * torch.mean((predicted_reward - self._scale(rewards)) ** 2)
            losses['reward_pred_loss'] = reward_loss.mean()

        if self._next_embed_pred_loss:
            terminal_mask = view(torch.stack(rollouts.terminals[:-1]))
            predicted_next_embed = view(torch.stack(rollouts.predicted_next_embed[:-1]))
            predicted_next_embed_flat = predicted_next_embed.view(predicted_next_embed.shape[0], -1)
            # this is obs time + 1
            obs_embed = view(torch.stack(rollouts.obs_embed[1:])).detach()
            obs_embed_flat = obs_embed.view(obs_embed.shape[0], -1)
            pred_mse_loss = 0.5 * torch.mean((predicted_next_embed_flat - obs_embed_flat)**2, dim=1)
            losses['next_embed_pred_loss'] = (pred_mse_loss * terminal_mask).mean()

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

