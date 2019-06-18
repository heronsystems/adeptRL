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

class I2AEmbed(NetworkModule):
    args = {
        'nb_imagination_rollout': 4
    }

    def __init__(self, args, obs_space, output_space):
        super().__init__()
        self.nb_imagination_rollout = args.nb_imagination_rollout
        self._nb_action = int(output_space['Discrete'][0] / 51)

        # encode state with recurrence captured
        self._obs_key = list(obs_space.keys())[0]
        self.conv_stack = FourConv(obs_space[self._obs_key], 'fourconv', True)
        self.lstm = ConvLSTM(self.conv_stack.output_shape(), 'lstm', True, 32, 3)
        lstm_out_shape = np.prod(self.lstm.output_shape())

        # policy output
        self.pol_outputs = nn.ModuleDict(
            {k: nn.Linear(lstm_out_shape+512, v[0]) for k, v in output_space.items()}
        )

        # imagined next embedding
        self.imag_embed_encoder = ResEmbed(self.lstm.output_shape()[0]+self._nb_action, 32)
        self.imag_encoder = nn.Linear(np.prod(self.lstm.output_shape()) + 1, 128, bias=True)
        # reward prediciton from lstm + action one hot
        self.reward_pred = nn.Linear(lstm_out_shape+self._nb_action, 1)
        # distil policy from imagination
        self.distil_pol_outputs = nn.Linear(lstm_out_shape, output_space['Discrete'][0])

    @classmethod
    def from_args(
        cls,
        args,
        observation_space,
        output_space,
        net_reg
    ):
        """
        Construct a I2A from arguments.

        ArgName = str
        ObsKey = str
        OutputKey = str
        Shape = Tuple[*int]

        :param args: Dict[ArgName, Any]
        :param observation_space: Dict[ObsKey, Shape]
        :param output_space: Dict[OutputKey, Shape]
        :param net_reg: NetworkRegistry
        :return: I2A
        """
        return I2AEmbed(args, observation_space, output_space)

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
        lstm_flat = hx.view(*hx.shape[0:-3], -1)
        return lstm_flat, lstm_internals

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
        encoded_obs, lstm_internals = self._encode_observation(obs, internals)

        # imagination rollout
        imag_encoded_obs, imag_internals = encoded_obs, lstm_internals
        imag_qs, imag_embed, imag_r = self._imag_forward(encoded_obs)
        imag_encoded = self._encode_imag(imag_embed, imag_r)
        # store for later
        all_imag_encoded = [imag_encoded]
        for i in range(self.nb_imagination_rollout - 1):
            # TODO: how can we incorporate world model sequential errors to train on them?
            with torch.no_grad():
                imag_embed_flat = imag_embed.view(encoded_obs.shape[0], -1)
                _, imag_embed, imag_r = self._imag_forward(imag_embed_flat)
            imag_encoded = self._encode_imag(imag_embed, imag_r)
            all_imag_encoded.append(imag_encoded)

        # concat imagination encodings
        imagination_rollout_encoding = torch.cat(all_imag_encoded, dim=1)
        policy_embedding = torch.cat([imagination_rollout_encoding, encoded_obs], dim=1)
        pol_outs = {k: self.pol_outputs[k](policy_embedding) for k in self.pol_outputs.keys()}

        # return cached stuff for training
        if ret_imag:
            pol_outs['imag_qs'] = imag_qs
            pol_outs['imag_encoded'] = encoded_obs

        return pol_outs, lstm_internals

    def _imag_forward(self, encoded_obs):
        # compute imagined action 
        imag_q = self.distil_pol_outputs(encoded_obs).view(-1, self._nb_action, 51)
        imag_action = imag_q.mean(2).argmax(-1, keepdim=True).detach()
        imag_one_hot_action = torch.zeros(imag_action.shape[0], self._nb_action, device=imag_action.device)
        imag_one_hot_action = imag_one_hot_action.scatter_(1, imag_action, 1)

        # reward prediction
        imag_r = self.reward_pred(torch.cat([encoded_obs, imag_one_hot_action], dim=1))

        # state prediction
        # encoder back to conv
        encoded_obs_conv = encoded_obs.view(-1, 32, 3, 3)

        # tile actions
        actions_tiled = imag_one_hot_action.view(imag_action.shape[0], self._nb_action, 1, 1).repeat(1, 1, 3, 3)
        # cat to upsample
        cat_lstm_act = torch.cat([encoded_obs_conv, actions_tiled], dim=1)

        imag_next_embed = self.imag_embed_encoder(cat_lstm_act)
        return imag_q, imag_next_embed, imag_r

    def _encode_imag(self, imag_state, imag_r):
        imag_flat = imag_state.view(*imag_state.shape[0:-3], -1)
        imag_cat_r = torch.cat([imag_flat, imag_r], dim=1)
        imag_encoded = F.relu(self.imag_encoder(imag_cat_r))
        return imag_encoded

    def pred_next_from_action(self, imag_encoded, actions):
        # reward prediction
        predicted_r = self.reward_pred(torch.cat([imag_encoded, actions], dim=1))

        # state prediction
        imag_conv_lstm = imag_encoded.view(-1, 32, 3, 3)
        # tile actions
        actions_tiled = actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)
        # cat to upsample
        cat_lstm_conv_act = torch.cat([imag_conv_lstm, actions_tiled], dim=1)

        predicted_next_obs = self.imag_embed_encoder(cat_lstm_conv_act)
        return predicted_next_obs, predicted_r

    def _merge_internals(self, internals):
        merged_internals = {}
        for internal in internals:
            for k, v in internal.items():
                merged_internals[k] = v
        return merged_internals


def pixel_norm(xs):
    return xs * torch.rsqrt(torch.mean(xs ** 2, dim=1, keepdim=True) + 1e-8)


class ResEmbed(nn.Module):
    def __init__(self, in_channel, batch_norm=False):
        super().__init__()
        self._out_shape = None
        self.conv1 = nn.Conv2d(in_channel, 32, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1, bias=False)

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

