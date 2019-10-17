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
from collections import OrderedDict

import torch

from .dqn_rollout import DQNRolloutActor


class DDQNRolloutActor(DQNRolloutActor):
    @staticmethod
    def output_space(action_space, args=None):
        head_dict = {'value': (1, ), **action_space}
        return head_dict

    def _get_qvals_from_pred(self, predictions):
        q = {}
        for k in self.action_keys:
            norm_adv = predictions[k] - predictions[k].mean(-1, keepdim=True)
            q[k] = norm_adv + predictions['value']
        return q

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space):
        act_key_len = len(act_space.keys())

        spec = {
            'values': (exp_len, batch_sz, act_key_len),
            'value': (exp_len, batch_sz)
        }

        return spec


class QRDDQNRolloutActor(DQNRolloutActor):
    @staticmethod
    def output_space(action_space):
        # TODO: This function needs args, to replace hardcoded 51
        head_dict = {}
        for k, shape in action_space.items():
            head_dict[k] = (shape[0] * 51, )
        head_dict['value'] = (51, )
        return head_dict

    def _get_qvals_from_pred(self, predictions):
        pred = {}
        for k in self.action_keys:
            v = predictions[k]
            # batch, num_actions, num_atoms
            adv = v.view(v.shape[0], self._action_space[k][0], -1)
            norm_adv = adv - adv.mean(1, keepdim=True)
            pred[k] = norm_adv + predictions['value'].unsqueeze(1)
        return pred

    def _action_from_q_vals(self, q_vals):
        # mean atoms, argmax over mean
        return q_vals.mean(-1).argmax(dim=-1, keepdim=True)

    def _get_action_values(self, q_vals, action, batch_size):
        # TODO: need to store num atoms in self
        action_select = action.unsqueeze(1).expand(batch_size, 1, 51)
        return q_vals.gather(1, action_select).squeeze(1)

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space):
        # TODO: This function needs args, to replace hardcoded 51
        act_key_len = len(act_space.keys())

        spec = {
            'values': (exp_len, batch_sz, act_key_len * 51),
            'value': (exp_len, batch_sz)
        }

        return spec

