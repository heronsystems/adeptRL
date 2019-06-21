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
import numpy as np


from adept.utils import listd_to_dlist
from adept.networks import NetworkModule
from adept.networks.net3d.four_conv import FourConv
from adept.networks.net3d.convlstm import ConvLSTM

import torch
from torch import nn
from torch.nn import ConvTranspose2d, BatchNorm2d, init, functional as F
from torch.nn import UpsamplingBilinear2d


def flatten(tensor):
    return tensor.view(*tensor.shape[0:-3], -1)


class Embedder(NetworkModule):
    args = {
        'autoencoder': True,
        'vae': False,
        'tanh': True,
        'reward_pred': True,
        'next_embed_pred': True,
        'inv_model': True,
        'additive_embed': False
    }

    def __init__(self, args, obs_space, output_space):
        super().__init__()
        self._autoencoder = args.autoencoder
        self._vae = args.vae
        self._tanh = args.tanh
        self._reward_pred = args.reward_pred
        self._next_embed_pred = args.next_embed_pred
        self._inv_model = args.inv_model
        self._additive_embed = args.additive_embed
        self._nb_action = int(output_space['Discrete'][0] / 51)

        # encode state with recurrence captured
        self._obs_key = list(obs_space.keys())[0]
        self.conv_stack = FourConv(obs_space[self._obs_key], 'fourconv', True)

        if self._vae:
            # reparameterize
            self.lstm = ConvLSTM(self.conv_stack.output_shape(), 'lstm', True, 64, 3)
            embed_shape = list(self.lstm.output_shape())
            # don't include mu & sigma, num channels will be 32
            embed_shape[0] = 32
            embed_shape = tuple(embed_shape)
        else:
            self.lstm = ConvLSTM(self.conv_stack.output_shape(), 'lstm', True, 32, 3)
            embed_shape = self.lstm.output_shape()
        embed_flat_shape = np.prod(embed_shape)

        # policy output
        self.pol_outputs = nn.ModuleDict(
            {k: nn.Linear(embed_flat_shape, v[0]) for k, v in output_space.items()}
        )

        # upsample back to image
        if self._autoencoder:
            self.ae_upsample = PixelShuffleFourConv(embed_shape[0])

        # predict reward negative, zero, positive given encoded_obs and action
        if self._reward_pred:
            self.reward_pred = nn.Sequential(
                nn.Linear(embed_flat_shape+self._nb_action, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 3),
                nn.Softmax(dim=1)
            )

        # predict next encoded_obs
        if self._next_embed_pred:
            if self._additive_embed:
                self.embed_pred = ResEmbed(self._nb_action, embed_shape[0])
            else:
                self.embed_pred = ResEmbed(embed_shape[0]+self._nb_action, embed_shape[0])

        # predict action given encoded_obs, encoded_obs_tp1
        if self._inv_model:
            self.inv_pred = InvModel(embed_shape[0] * 2, self._nb_action)

        # imagined next embedding
        # self.imag_embed_encoder = ResEmbed(self.lstm.output_shape()[0]+self._nb_action, 32)
        # self.imag_encoder = nn.Linear(np.prod(self.lstm.output_shape()) + 1, 128, bias=True)
        # reward prediciton from lstm + action one hot
        # self.reward_pred = nn.Linear(lstm_out_shape+self._nb_action, 1)
        # distil policy from imagination
        # self.distil_pol_outputs = nn.Linear(lstm_out_shape, output_space['Discrete'][0])

    @classmethod
    def from_args(
        cls,
        args,
        observation_space,
        output_space,
        net_reg
    ):
        """
        Construct a Embedder from arguments.

        ArgName = str
        ObsKey = str
        OutputKey = str
        Shape = Tuple[*int]

        :param args: Dict[ArgName, Any]
        :param observation_space: Dict[ObsKey, Shape]
        :param output_space: Dict[OutputKey, Shape]
        :param net_reg: NetworkRegistry
        :return: Embedder
        """
        return Embedder(args, observation_space, output_space)

    def new_internals(self, device):
        """
        Define any initial hidden states here, move them to device if necessary.

        InternalKey=str

        :return: Dict[InternalKey, torch.Tensor (ND)]
        """
        return {
            **self.lstm.new_internals(device),
        }

    def _encode_observation(self, observation, internals):
        conv_out, _ = self.conv_stack(observation, {})
        hx, lstm_internals = self.lstm(conv_out, internals)

        if self._vae:
            mu = hx[:, :32]
            logvar = hx[:, 32:]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps, mu, logvar, lstm_internals
        elif self._tanh:
            # hx is tanh'd add noise
            with torch.no_grad():
                maxes = 1 - hx
                mins = -1 - hx
                noise = 0.5 * torch.randn_like(hx)
                # clamp noise
                noise = torch.max(torch.min(noise, maxes), mins)
            return noise + hx, hx, lstm_internals
        else:
            return hx, lstm_internals

    def forward(self, observation, internals, ret_imag=False):
        """
        Compute forward pass.

        ObsKey = str
        InternalKey = str

        :param observation: Dict[ObsKey, torch.Tensor (1D | 2D | 3D | 4D)]
        :param internals: Dict[InternalKey, torch.Tensor (ND)]
        :return: Dict[str, torch.Tensor (ND)]
        """
        obs = observation[self._obs_key]
        if self._vae:
            encoded_obs, encoded_mu, encoded_logvar, lstm_internals = self._encode_observation(obs, internals)
        elif self._tanh:
            encoded_obs, encoded_obs_nonoise, lstm_internals = self._encode_observation(obs, internals)
        else:
            encoded_obs, lstm_internals = self._encode_observation(obs, internals)
        encoded_obs_flat = flatten(encoded_obs)

        pol_outs = {k: self.pol_outputs[k](encoded_obs_flat) for k in self.pol_outputs.keys()}

        # return cached stuff for training
        if ret_imag:
            if self._autoencoder:
                # upsample back to pixels
                ae_state_pred = self.ae_upsample(encoded_obs)
                pol_outs['ae_state_pred'] = ae_state_pred
            if self._reward_pred or self._next_embed_pred:
                pol_outs['encoded_obs'] = encoded_obs
            if self._vae:
                pol_outs['encoded_mu'] = encoded_mu
                pol_outs['encoded_logvar'] = encoded_logvar
            if self._tanh:
                pol_outs['encoded_obs_nonoise'] = encoded_obs_nonoise

        return pol_outs, lstm_internals

    def predict_reward(self, encoded_obs, action_taken):
        # reward prediction
        return self.reward_pred(torch.cat([flatten(encoded_obs), action_taken], dim=1))

    def predict_next_embed(self, encoded_obs, action_taken):
        # tile actions
        actions_tiled = action_taken.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)

        if self._additive_embed:
            # only the actions are passed
            return self.embed_pred(actions_tiled) + encoded_obs
        else:
            # cat along channel dim
            cat_embed_act = torch.cat([encoded_obs, actions_tiled], dim=1)
            return self.embed_pred(cat_embed_act)

    def predict_inv_action(self, encoded_obs, encoded_obs_tp1):
        # concat along channel
        all_obs = torch.cat([encoded_obs, encoded_obs_tp1], dim=1)
        return self.inv_pred(all_obs)


def pixel_norm(xs):
    return xs * torch.rsqrt(torch.mean(xs ** 2, dim=1, keepdim=True) + 1e-8)


class ResEmbed(nn.Module):
    def __init__(self, in_channel, out_channel, batch_norm=False):
        super().__init__()
        self._out_shape = None
        self.conv1 = nn.Conv2d(in_channel, 32, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, out_channel, 3, padding=1, bias=False)

        if not batch_norm:
            self.n1 = pixel_norm
            self.n2 = pixel_norm
        else:
            self.n1 = BatchNorm2d(32)
            self.n2 = BatchNorm2d(64)

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)

    def forward(self, xs):
        xs = F.relu(self.n1(self.conv1(xs)))
        xs = F.relu(self.n2(self.conv2(xs)))
        xs = self.conv3(xs)
        return xs


class PixelShuffleFourConv(nn.Module):
    def __init__(self, in_channel, batch_norm=False):
        super().__init__()
        self._out_shape = None
        self.conv1 = ConvTranspose2d(in_channel, 32*4, 7, bias=False)
        self.conv2 = ConvTranspose2d(32, 32*4, 4, bias=False)
        self.conv3 = ConvTranspose2d(32, 32*4, 3, padding=1, bias=False)
        self.conv4 = ConvTranspose2d(32, 1, 3, padding=1, bias=True)

        if not batch_norm:
            self.n1 = pixel_norm
            self.n2 = pixel_norm
            self.n3 = pixel_norm
        else:
            self.n1 = BatchNorm2d(32*4)
            self.n2 = BatchNorm2d(32*4)
            self.n3 = BatchNorm2d(32*4)

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)

    def forward(self, xs):
        xs = F.relu(F.pixel_shuffle(self.n1(self.conv1(xs)), 2))
        xs = F.relu(F.pixel_shuffle(self.n2(self.conv2(xs)), 2))
        xs = F.relu(F.pixel_shuffle(self.n3(self.conv3(xs)), 2))
        xs = F.leaky_relu(self.conv4(xs))
        return xs


class InvModel(nn.Module):
    def __init__(self, in_channel, out_channel, batch_norm=False):
        super().__init__()
        self._out_shape = None
        self.conv1 = nn.Conv2d(in_channel, 32, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, out_channel, 3, bias=False)

        if not batch_norm:
            self.n1 = pixel_norm
            self.n2 = pixel_norm
        else:
            self.n1 = BatchNorm2d(32)
            self.n2 = BatchNorm2d(64)

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)

    def forward(self, xs):
        xs = F.relu(self.n1(self.conv1(xs)))
        xs = F.relu(self.n2(self.conv2(xs)))
        xs = F.softmax(self.conv3(xs).squeeze(), dim=-1)
        return xs

