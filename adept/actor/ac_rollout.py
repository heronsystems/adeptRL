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
import abc
from collections import OrderedDict

import torch

from adept.actor._ac_helper import ACActorHelperMixin
from adept.actor._actor import ActorMixin


class ACRolloutActorTrainMixin(
    ActorMixin,
    ACActorHelperMixin,
    metaclass=abc.ABCMeta
):
    @staticmethod
    def output_space(action_space):
        head_dict = {'critic': (1,), **action_space}
        return head_dict

    def process_predictions(self, preds, available_actions):
        values = preds['critic'].squeeze(1)

        actions = OrderedDict()
        log_probs = []
        entropies = []

        for key in self.action_keys:
            logit = self.flatten_to_2d(preds[key])

            log_softmax, softmax = self.log_softmax(logit), self.softmax(logit)
            entropy = self.entropy(log_softmax, softmax)
            action = self.sample_action(softmax)

            entropies.append(entropy)
            log_probs.append(self.log_probability(log_softmax, action))
            actions[key] = action.cpu()

        log_probs = torch.cat(log_probs, dim=1)
        entropies = torch.cat(entropies, dim=1)

        return actions, {
            'log_probs': log_probs,
            'entropies': entropies,
            'values': values
        }


class ACRolloutActorTrain(ACRolloutActorTrainMixin):
    def __init__(self, network, device, gpu_preprocessor, action_space, nb_env):
        self._network = network
        self._device = device
        self._internals = [
            self.network.new_internals(device) for _ in range(nb_env)
        ]
        self._gpu_preprocessor = gpu_preprocessor
        self._action_space = action_space

    @property
    def network(self):
        return self._network

    @property
    def device(self):
        return self._device

    @property
    def internals(self):
        return self._internals

    @internals.setter
    def internals(self, new):
        self.internals = new

    @property
    def is_train(self):
        return True

    @property
    def gpu_preprocessor(self):
        return self._gpu_preprocessor

    @property
    def action_space(self):
        return self._action_space
