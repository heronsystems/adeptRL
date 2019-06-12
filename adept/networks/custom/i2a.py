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


from adept.networks import NetworkModule
from adept.networks.net3d.four_conv import FourConv
from adept.networks.net1d.lstm import LSTM

import torch
from torch import nn
from torch.nn import ConvTranspose2d, BatchNorm2d, init, functional as F
from torch.nn import UpsamplingBilinear2d

class I2A(NetworkModule):
    args = {
    }

    def __init__(self, args, obs_space, output_space):
        super().__init__()

        self._obs_key = list(obs_space.keys())[0]
        self.conv_stack = FourConv(obs_space[self._obs_key], 'fourconv', True)
        conv_out_shape = np.prod(self.conv_stack.output_shape())
        self.lstm = LSTM((conv_out_shape, ), 'lstm', True, 512)
        self.auto_lstm = LSTM((conv_out_shape, ), 'autolstm', True, 800)
        self.pol_outputs = nn.ModuleDict(
            {k: nn.Linear(512, v[0]) for k, v in output_space.items()}
        )
        self._nb_action = int(output_space['Discrete'][0] / 51)

        # reward prediciton from lstm + action one hot
        self.reward_pred = nn.Linear(1600+self._nb_action, 1)
        # upsample_stack needs to make a 1x84x84 from 64x5x5
        self.upsample_stack = PixelShuffleFourConv(64+self._nb_action)

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
        return I2A(args, observation_space, output_space)

    def new_internals(self, device):
        """
        Define any initial hidden states here, move them to device if necessary.

        InternalKey=str

        :return: Dict[InternalKey, torch.Tensor (ND)]
        """
        return {
            **self.lstm.new_internals(device),
            **self.auto_lstm.new_internals(device)
        }

    def forward(self, observation, internals, ret_lstm=False):
        """
        Compute forward pass.

        ObsKey = str
        InternalKey = str

        :param observation: Dict[ObsKey, torch.Tensor (1D | 2D | 3D | 4D)]
        :param internals: Dict[InternalKey, torch.Tensor (ND)]
        :return: Dict[str, torch.Tensor (ND)]
        """
        obs = observation[self._obs_key]
        conv_out, _ = self.conv_stack(obs, {})
        conv_flat = conv_out.view(*conv_out.shape[0:-3], -1)
        hx, cx = self.lstm(conv_flat, internals)
        auto_hx, auto_cx = self.auto_lstm(conv_flat, internals)

        pol_outs = {k: self.pol_outputs[k](hx) for k in self.pol_outputs.keys()}
        if ret_lstm:
            pol_outs['lstm_out'] = torch.cat([conv_out, auto_hx.view(-1, 32, 5, 5)], dim=1)

        return pol_outs, self._merge_internals([cx, auto_cx])

    def pred_next(self, encoder, actions):
        # tile actions
        actions_tiled = actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 5, 5)
        # cat to upsample
        cat_lstm_act = torch.cat([encoder, actions_tiled], dim=1)
        # cat to reward pred
        cat_flat_act = torch.cat([encoder.view(-1, 1600), actions], dim=1)

        predicted_next_obs = self.upsample_stack(cat_lstm_act)
        predicted_next_r = self.reward_pred(cat_flat_act)
        return predicted_next_obs, predicted_next_r

    def _merge_internals(self, internals):
        merged_internals = {}
        for internal in internals:
            for k, v in internal.items():
                merged_internals[k] = v
        return merged_internals


def pixel_norm(xs):
    return xs * torch.rsqrt(torch.mean(xs ** 2, dim=1, keepdim=True) + 1e-8)


class PixelShuffleFourConv(nn.Module):
    def __init__(self, in_channel, batch_norm=False):
        super().__init__()
        self._out_shape = None
        self.conv1 = ConvTranspose2d(in_channel, 32*4, 5, bias=False)
        self.conv2 = ConvTranspose2d(32, 32*4, 3, bias=False)
        self.conv3 = ConvTranspose2d(32, 32*4, 3, bias=False)
        self.conv4 = ConvTranspose2d(32, 1, 3, padding=1, bias=True)
        # if cross entropy
        # self.conv4 = ConvTranspose2d(32, 255, 7, bias=True)

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
        xs = self.conv4(xs)
        return xs

