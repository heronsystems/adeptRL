"""
Copyright (C) 2018 Heron Systems, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import torch
from torch.nn import functional as F

BATCH_AXIS = 0
CHANNEL_AXIS = 1


class CircularDND(torch.nn.Module):
    """
    Does not support batches
    """
    def __init__(self, nb_key_chan, nb_v_chan, delta=1e-3, query_width=50, max_len=1028):
        super(CircularDND, self).__init__()
        self.delta = delta
        self.query_width = query_width
        self.keys = torch.nn.Parameter(torch.zeros(max_len, nb_key_chan, requires_grad=True))
        self.values = torch.nn.Parameter(torch.zeros(max_len, nb_v_chan, requires_grad=True))

    def forward(self, key):
        inds, weights = self._k_nearest(key, self.query_width)
        return torch.sum(self.values[inds, :] * weights, BATCH_AXIS, keepdim=True)

    def _k_nearest(self, key, k):
        lookup_weights = self._kernel(key, self.keys)
        top_ks, top_k_inds = torch.topk(lookup_weights, k)
        weights = (top_ks / torch.sum(lookup_weights)).unsqueeze(CHANNEL_AXIS)
        return top_k_inds, weights

    def _kernel(self, query_key, all_keys):
        return 1. / (torch.pow(query_key - all_keys, 2).sum(CHANNEL_AXIS) + self.delta)

    def sync_from_shared(self, shared_dnd):
        self.load_state_dict(shared_dnd.state_dict())

    def sync_to_shared(self, shared_dnd):
        is_cpu = self.keys.device.type == 'cpu'
        if shared_dnd.keys.grad is not None and is_cpu:
            return
        elif is_cpu:
            shared_dnd.keys._grad = self.keys.grad
            shared_dnd.values._grad = self.values.grad
        else:
            shared_dnd.keys._grad = self.keys.grad.cpu()
            shared_dnd.values._grad = self.values.grad.cpu()

    def detach(self):
        self.keys.detach_()
        self.values.detach_()


class PruningDND(torch.nn.Module):
    """
    Does not support batches.
    """
    def __init__(self, nb_key_chan, nb_v_chan, delta=1e-3, query_width=50, max_len=1024):
        super(PruningDND, self).__init__()
        self.delta = delta
        self.query_width = query_width
        self.keys = torch.nn.Parameter(torch.rand(max_len, nb_key_chan))
        self.values = torch.nn.Parameter(torch.zeros(max_len, nb_v_chan))
        self.register_buffer('weight_buff', torch.zeros(max_len))

    def forward(self, key):
        inds, weights = self._k_nearest(key, self.query_width)
        return torch.sum(self.values[inds, :] * weights.unsqueeze(CHANNEL_AXIS), BATCH_AXIS, keepdim=True), inds, weights

    def _k_nearest(self, key, k):
        lookup_weights = self._kernel(key, self.keys)
        top_ks, top_k_inds = torch.topk(lookup_weights, k)
        weights = (top_ks / torch.sum(lookup_weights))
        return top_k_inds, weights

    def _kernel(self, query_key, all_keys):
        return 1. / (torch.pow(query_key - all_keys, 2).sum(CHANNEL_AXIS) + self.delta)

    def sync_from_shared(self, shared_dnd):
        self.load_state_dict(shared_dnd.state_dict())

    def sync_to_shared(self, shared_dnd):
        is_cpu = self.keys.device.type == 'cpu'
        if shared_dnd.keys.grad is not None and is_cpu:
            return
        elif is_cpu:
            shared_dnd.keys._grad = self.keys.grad
            shared_dnd.values._grad = self.values.grad
        else:
            shared_dnd.keys._grad = self.keys.grad.cpu()
            shared_dnd.values._grad = self.values.grad.cpu()

    def detach(self):
        self.keys.detach_()
        self.values.detach_()

    # only callable on the shared dnd
    def update_buff(self, inds, weights):
        self.weight_buff[inds] = weights

    # only callable on the shared dnd
    def append(self, new_k, new_v):
        min_idx = torch.argmin(self.weight_buff).item()
        self.keys[min_idx, :] = new_k
        self.values[min_idx, :] = new_v
        self.weight_buff[min_idx] = torch.mean(self.weight_buff)


class FreqPruningLTM(torch.nn.Module):
    def __init__(self, nb_key_chan, nb_v_chan, query_breadth=50, max_len=1024):
        super(FreqPruningLTM, self).__init__()
        self.query_breadth = query_breadth
        self.keys = torch.nn.Parameter(torch.randn(max_len, nb_key_chan))
        self.values = torch.nn.Parameter(torch.randn(max_len, nb_v_chan))
        self.register_buffer('weight_buff', torch.zeros(max_len))

    def forward(self, queries):
        """
        :param queries: expecting a [Batch Size, Num Key Channel] matrix
        :return: a [Batch Size, Num Value Channel] matrix
        """
        inds, weights = self._k_nearest(queries, self.query_breadth)

        # indexing headache
        values = self.values.unsqueeze(0)
        values = values.expand(inds.size(0), values.size(1), values.size(2))
        inds_tmp = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), values.size(2))
        selected_values = values.gather(1, inds_tmp)

        weighted_selection = (selected_values * weights.unsqueeze(2)).sum(1)
        return weighted_selection, inds, weights

    def _k_nearest(self, queries, query_width):
        lookup_weights = torch.mm(queries, torch.t(self.keys))
        top_ks, top_k_inds = torch.topk(lookup_weights, query_width)
        weights = F.softmax(top_ks, dim=CHANNEL_AXIS)
        return top_k_inds, weights

    def sync_from_shared(self, shared_dnd):
        self.load_state_dict(shared_dnd.state_dict())

    def sync_to_shared(self, shared_dnd):
        is_cpu = self.keys.device.type == 'cpu'
        if shared_dnd.keys.grad is not None and is_cpu:
            return
        elif is_cpu:
            shared_dnd.keys._grad = self.keys.grad
            shared_dnd.values._grad = self.values.grad
        else:
            shared_dnd.keys._grad = self.keys.grad.cpu()
            shared_dnd.values._grad = self.values.grad.cpu()

    def detach(self):
        self.keys = self.keys.detach_()
        self.values = self.values.detach_()

    # only callable on the shared dnd
    def update_buff(self, inds, weights):
        self.weight_buff[inds] = weights

    # only callable on the shared dnd
    def append(self, new_k, new_v):
        """
        :param new_k: expecting a vector of dimensionality [Num Key Chan]
        :param new_v: expecting a vector of dimensionality [Num Value Chan]
        :return:
        """
        min_idx = torch.argmin(self.weight_buff).item()
        self.keys[min_idx, :] = new_k
        self.values[min_idx, :] = new_v
        self.weight_buff[min_idx] = torch.mean(self.weight_buff)
