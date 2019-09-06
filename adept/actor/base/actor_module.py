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
"""
An actor observes the environment and decides actions. It also outputs extra
info necessary for model updates (learning) to occur.
"""
import abc

from adept.exp.base.spec_builder import ExpSpecBuilder
from adept.utils.requires_args import RequiresArgsMixin


class ActorModule(RequiresArgsMixin, metaclass=abc.ABCMeta):

    def __init__(
            self,
            action_space
    ):
        self._action_space = action_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def action_keys(self):
        return list(sorted(self.action_space.keys()))

    @staticmethod
    @abc.abstractmethod
    def output_space(action_space):
        raise NotImplementedError

    @staticmethod
    def exp_spec_builder(obs_space, act_space, internal_space, batch_sz):
        def build_fn(exp_len):
            return ActorModule._exp_spec(
                exp_len, batch_sz, obs_space, act_space, internal_space)
        return ExpSpecBuilder(obs_space, act_space, internal_space, build_fn)

    @staticmethod
    @abc.abstractmethod
    def _exp_spec(exp_len, batch_sz, obs_space, act_space, internal_space):
        raise NotImplementedError

    @abc.abstractmethod
    def from_args(self, args, action_space):
        raise NotImplementedError

    @abc.abstractmethod
    def process_predictions(self, preds, available_actions):
        """
        B = Batch Size

        :param preds: Dict[str, torch.Tensor]
        :return:
            actions: Dict[ActionKey, LongTensor (B)]
            experience: Dict[str, Tensor (B, X)]
        """
        raise NotImplementedError

    def act(self, network, obs, prev_internals):
        """
        :param obs: Dict[str, Tensor]
        :param prev_internals: previous interal states. Dict[str, Tensor]
        :return:
            actions: Dict[ActionKey, LongTensor (B)]
            experience: Dict[str, Tensor (B, X)]
            internal_states: Dict[str, Tensor]
        """

        predictions, internal_states = network(obs, prev_internals)

        if 'available_actions' in obs:
            av_actions = obs['available_actions']
        else:
            av_actions = None

        actions, exp = self.process_predictions(predictions, av_actions)
        return actions, exp, internal_states
