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

from adept.actor.base.ac_helper import ACActorHelperMixin
from adept.actor.base.actor_module import ActorModule


class ACRolloutActorTrain(ActorModule, ACActorHelperMixin):
    args = {}

    @classmethod
    def from_args(cls, args, action_space):
        return cls(action_space)

    @staticmethod
    def output_space(action_space):
        head_dict = {"critic": (1,), **action_space}
        return head_dict

    def compute_action_exp(self, preds, internals, obs, available_actions):
        values = preds["critic"].squeeze(1)

        actions = OrderedDict()
        log_probs = []
        entropies = []

        for key in self.action_keys:
            logit = self.flatten_logits(preds[key])

            log_softmax, softmax = self.log_softmax(logit), self.softmax(logit)
            entropy = self.entropy(log_softmax, softmax)
            action = self.sample_action(softmax)

            entropies.append(entropy)
            log_probs.append(self.log_probability(log_softmax, action))
            actions[key] = action.cpu()

        log_probs = torch.cat(log_probs, dim=1)
        entropies = torch.cat(entropies, dim=1)

        return (
            actions,
            {"log_probs": log_probs, "entropies": entropies, "values": values},
        )

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space):
        act_key_len = len(act_space.keys())

        spec = {
            "log_probs": (exp_len, batch_sz, act_key_len),
            "entropies": (exp_len, batch_sz, act_key_len),
            "values": (exp_len, batch_sz),
        }

        return spec
