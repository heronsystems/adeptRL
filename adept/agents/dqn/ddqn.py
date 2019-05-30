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
from adept.agents.dqn import DQN
from adept.agents.dqn import OnlineDQN


class DDQN(DQN):
    args = {**DQN.args}
    args['double_dqn'] = False

    @staticmethod
    def output_space(action_space, args=None):
        head_dict = {'value': (1, ), **action_space}
        return head_dict

    def _get_qvals_from_pred(self, predictions):
        q = {}
        for k in self._action_keys:
            norm_adv = predictions[k] - predictions[k].mean(-1, keepdim=True)
            q[k] = norm_adv + predictions['value']
        return q


# TODO: these are the same, must be a fancy way to do this in python
class OnlineDDQN(OnlineDQN):
    args = {**OnlineDQN.args}
    args['double_dqn'] = False

    @staticmethod
    def output_space(action_space, args=None):
        head_dict = {'value': (1, ), **action_space}
        return head_dict

    def _get_qvals_from_pred(self, predictions):
        q = {}
        for k in self._action_keys:
            norm_adv = predictions[k] - predictions[k].mean(-1, keepdim=True)
            q[k] = norm_adv + predictions['value']
        return q
