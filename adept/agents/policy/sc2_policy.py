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
from adept.environments.deepmind_sc2 import SC2ActionLookup


class SC2Policy:
    def __init__(self, action_space):
        super(SC2Policy, self).__init__()
        self._action_space = action_space
        self._action_keys = list(sorted(action_space.keys()))
        self._func_id_to_headnames = SC2ActionLookup()

    def act(self, predictions, available_actions):
        """
        :param predictions: Dict[ActionKey, torch.Tensor]
        :param available_actions: torch.Tensor (N, NB_ACTION), one hot
        :return:
            actions: Dict[ActionKey, torch.Tensor], flattened to (N)
            log_probs: torch.Tensor, flattened to (N, X)
            entropies: torch.Tensor, flattened to (N, X)
        """
        actions = OrderedDict()
        head_masks = OrderedDict()
        log_probs = []
        entropies = []
        for key in self._action_keys:
            logits = predictions[key]
            size = logits.size()
            dim = logits.dim()

            if dim == 3:
                n, f, l = size
                logits = logits.view(n, f * l)
            elif dim == 4:
                n, f, h, w = size
                logits = logits.view(n, f * h * w)
            elif dim == 5:
                n, f, d, h, w = size
                logits = logits.view(n, f * d * h * w)
            prob = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            action = prob.multinomial(1)
            log_prob = log_prob.gather(1, action)

            actions[key] = action.squeeze(1).cpu()
            log_probs.append(log_prob)
            entropies.append(entropy)

            # Initialize masks
            if key == 'func_id':
                head_masks[key] = torch.ones_like(entropy)
            else:
                head_masks[key] = torch.zeros_like(entropy)

        log_probs = torch.cat(log_probs, dim=1)
        entropies = torch.cat(entropies, dim=1)

        # Mask invalid actions with NOOP and fill masks with ones
        for batch_idx, action in enumerate(actions['func_id']):
            # convert unavailable actions to NOOP
            if available_actions[batch_idx][action] == 0:
                actions['func_id'][batch_idx] = 0

            # build SC2 action masks
            func_id = actions['func_id'][batch_idx]
            # TODO this can be vectorized via gather?
            for headname in self._func_id_to_headnames[func_id].keys():
                head_masks[headname][batch_idx] = 1.

        head_masks = torch.cat(
            [head_mask for head_mask in head_masks.values()], dim=1
        )
        log_probs = log_probs * head_masks
        entropies = entropies * head_masks
        return actions, log_probs, entropies

    def act_eval(self, predictions, available_actions):
        """
        :param predictions:
        :return: actions: Dict[ActionKey, torch.Tensor]
        """
        with torch.no_grad():
            pass
