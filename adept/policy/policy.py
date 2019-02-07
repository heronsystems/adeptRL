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
import torch
from torch.nn import functional as F
from collections import OrderedDict


class Policy:
    def __init__(self, action_space):
        super(Policy, self).__init__()
        self._action_space = action_space
        self._action_keys = list(sorted(action_space.keys()))

    def act(self, logits, available_actions=None):
        """
        :param logits: Dict[ActionKey, torch.Tensor]
        :return:
            actions: Dict[ActionKey, torch.LongTensor], flattened to (N)
            log_probs: torch.Tensor, flattened to (N, X)
            entropies: torch.Tensor, flattened to (N, X)
        """
        actions = OrderedDict()
        log_probs = []
        entropies = []
        for k in self._action_keys:
            logit = logits[k]
            size = logit.size()
            dim = logit.dim()

            if dim == 3:
                n, f, l = size
                logit = logit.view(n, f * l)
            elif dim == 4:
                n, f, h, w = size
                logit = logit.view(n, f * h * w)
            elif dim == 5:
                n, f, d, h, w = size
                logit = logit.view(n, f * d * h * w)

            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            action = prob.multinomial(1)  # (N)
            log_prob = log_prob.gather(1, action)

            actions[k] = action.cpu()
            log_probs.append(log_prob)
            entropies.append(entropy)

        log_probs = torch.cat(log_probs, dim=1)
        entropies = torch.cat(entropies, dim=1)
        return actions, log_probs, entropies

    def act_eval(self, logits, available_actions=None):
        """
        :param logits: Dict[ActionKey, torch.Tensor]
        :return: actions: Dict[ActionKey, torch.LongTensor]
        """
        with torch.no_grad():
            actions = OrderedDict()
            for k in self._action_keys:
                logit = logits[k]
                size = logit.size()
                dim = logit.dim()

                if dim == 3:
                    n, f, l = size
                    logit = logit.view(n, f * l)
                elif dim == 4:
                    n, f, h, w = size
                    logit = logit.view(n, f * h * w)
                elif dim == 5:
                    n, f, d, h, w = size
                    logit = logit.view(n, f * d * h * w)

                prob = F.softmax(logit, dim=1)
                action = prob.multinomial(1)  # (N)
                actions[k] = action.cpu()
            return actions
