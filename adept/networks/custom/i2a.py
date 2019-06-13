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

class I2A(NetworkModule):
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

        # imagined state encoder
        self.imag_conv_encoder = FourConv(obs_space[self._obs_key], 'imagfourconv', True)
        self.imag_encoder = nn.Linear(np.prod(self.imag_conv_encoder.output_shape()) + 1, 128, bias=True)
        # reward prediciton from lstm + action one hot
        self.reward_pred = nn.Linear(lstm_out_shape+self._nb_action, 1)
        # upsample_stack needs to make a 1x84x84 from 64x5x5
        self.upsample_stack = PixelShuffleFourConv(32+self._nb_action)
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
        return I2A(args, observation_space, output_space)

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
        imag_qs, imag_obs, imag_r = self._imag_forward(encoded_obs)
        imag_encoded = self._encode_imag(imag_obs, imag_r)
        # store for later
        all_imag_obs = [imag_obs]
        all_imag_encoded = [imag_encoded]
        for i in range(self.nb_imagination_rollout - 1):
            # TODO: how can we incorporate world model sequential errors to train on them?
            imag_encoded_obs, imag_internals = self._encode_observation(imag_obs, imag_internals)
            _, imag_obs, imag_r = self._imag_forward(imag_encoded_obs)
            imag_encoded = self._encode_imag(imag_obs, imag_r)
            all_imag_encoded.append(imag_encoded)
            all_imag_obs.append(imag_obs)

        # concat imagination encodings
        imagination_rollout_encoding = torch.cat(all_imag_encoded, dim=1)
        policy_embedding = torch.cat([imagination_rollout_encoding, encoded_obs], dim=1)
        pol_outs = {k: self.pol_outputs[k](policy_embedding) for k in self.pol_outputs.keys()}

        # return cached stuff for training
        if ret_imag:
            pol_outs['imag_qs'] = imag_qs
            pol_outs['imag_encoded'] = encoded_obs
            pol_outs['imag_obs'] = torch.stack(all_imag_obs)

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

        imag_next_state = self.upsample_stack(cat_lstm_act)
        return imag_q, imag_next_state, imag_r

    def _encode_imag(self, imag_state, imag_r):
        imag_conv_encoded, _ = self.imag_conv_encoder(imag_state, {})
        imag_conv_flat = imag_conv_encoded.view(*imag_conv_encoded.shape[0:-3], -1)
        imag_cat_r = torch.cat([imag_conv_flat, imag_r], dim=1)
        imag_encoded = F.relu(self.imag_encoder(imag_cat_r))
        return imag_encoded

#     def _pred_seq_forward(self, state, internals, actions, actions_tiled):
        # conv_out, _ = self.conv_stack(state, {})
        # conv_flat = conv_out.view(*conv_out.shape[0:-3], -1)
        # auto_hx, auto_internals = self.auto_lstm(conv_flat, internals)
        # encoder = torch.cat([conv_out, auto_hx.view(-1, 32, 5, 5)], dim=1)
        # # cat to upsample
        # cat_lstm_act = torch.cat([encoder, actions_tiled], dim=1)
        # # cat to reward pred
        # cat_flat_act = torch.cat([encoder.view(-1, 1600), actions], dim=1)

        # # TODO: this must be sorted in the same way as the agent
        # qvals = self.distil_pol_outputs(auto_hx)
        # qvals = qvals.view(actions.shape[0], self._nb_action, -1)
        # action_select = actions.argmax(-1, keepdim=True)
        # action_select = action_select.unsqueeze(-1).expand(action_select.shape[0], 1, qvals.shape[-1]).long()
        # predicted_qvals = qvals.gather(1, action_select).squeeze(1)

        # predicted_next_obs = self.upsample_stack(cat_lstm_act)
        # predicted_next_r = self.reward_pred(cat_flat_act)
        # return predicted_qvals, predicted_next_obs, predicted_next_r, auto_internals

    # def pred_seq(self, state, internals, actions, terminals, max_seq=1):
        # # tile actions
        # actions_tiled = actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 5, 5)
        # # starting from 0 predict the next sequence of states
        # pred_qs, pred_states, pred_reward, internals = self._pred_seq_forward(state, internals, actions[0], actions_tiled[0])
        # pred_qs, pred_states, pred_reward = [pred_qs], [pred_states], [pred_reward]
        # max_seq = min(max_seq, actions.shape[0])
        # for s_ind in range(1, max_seq):
            # pred_q, pred_s, pred_r, internals = self._pred_seq_forward(pred_states[-1], internals, actions[s_ind], actions_tiled[s_ind])
            # pred_qs.append(pred_q)
            # pred_states.append(pred_s)
            # pred_reward.append(pred_r)
            # # TODO: check for terminal and reset internals
        # return torch.stack(pred_qs), torch.stack(pred_states), torch.stack(pred_reward)

    def pred_next_from_action(self, imag_encoded, actions):
        # reward prediction
        predicted_r = self.reward_pred(torch.cat([imag_encoded, actions], dim=1))

        # state prediction
        imag_conv_lstm = imag_encoded.view(-1, 32, 3, 3)
        # tile actions
        actions_tiled = actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)
        # cat to upsample
        cat_lstm_conv_act = torch.cat([imag_conv_lstm, actions_tiled], dim=1)

        predicted_next_obs = self.upsample_stack(cat_lstm_conv_act)
        return predicted_next_obs, predicted_r

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
        self.conv1 = ConvTranspose2d(in_channel, 32*4, 7, bias=False)
        self.conv2 = ConvTranspose2d(32, 32*4, 4, bias=False)
        self.conv3 = ConvTranspose2d(32, 32*4, 3, padding=1, bias=False)
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
        xs = F.leaky_relu(self.conv4(xs))
        return xs

