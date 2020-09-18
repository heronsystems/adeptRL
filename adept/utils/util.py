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
import json
import heapq
from collections import OrderedDict

import numpy as np
import torch


def listd_to_dlist(list_of_dicts):
    """
    Converts a list of dictionaries to a dictionary of lists. Preserves key
    order.

    K is type of key.
    V is type of value.
    :param list_of_dicts: List[Dict[K, V]]
    :return: Dict[K, List[V]]
    """
    new_dict = OrderedDict()
    for d in list_of_dicts:
        for k, v in d.items():
            if k not in new_dict:
                new_dict[k] = [v]
            else:
                new_dict[k].append(v)
    return new_dict


def dlist_to_listd(dict_of_lists):
    """
    Converts a dictionary of lists to a list of dictionaries. Preserves key
    order.

    K is type of key.
    V is type of value.
    :param dict_of_lists: Dict[K, List[V]]
    :return: List[Dict[K, V]]
    """
    keys = dict_of_lists.keys()
    list_len = len(dict_of_lists[next(iter(keys))])
    new_list = []
    for i in range(list_len):
        temp_d = OrderedDict()
        for k in keys:
            temp_d[k] = dict_of_lists[k][i]
        new_list.append(temp_d)
    return new_list


def dtensor_to_dev(d_tensor, device):
    """
    Move a dictionary of tensors to a device.

    :param d_tensor: Dict[str, Tensor]
    :param device: torch.device
    :return: Dict[str, Tensor] on desired device.
    """
    return {k: v.to(device) for k, v in d_tensor.items()}


def json_to_dict(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, "r"))
    return json_object


_numpy_to_torch_dtype = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
}
_torch_to_numpy_dtype = {v: k for k, v in _numpy_to_torch_dtype.items()}


def numpy_to_torch_dtype(dtype):

    # check if dtype is weird and convert to familiar format
    if type(dtype) == np.dtype:
        dtype = dtype.type

    if dtype not in _numpy_to_torch_dtype:
        raise ValueError(
            "Could not convert numpy dtype {} to a torch dtype.".format(
                dtype
            )
        )

    return _numpy_to_torch_dtype[dtype]


def torch_to_numpy_dtype(dtype):
    if dtype not in _torch_to_numpy_dtype:
        raise ValueError(
            "Could not convert torch dtype {} to a numpy dtype.".format(
                dtype
            )
        )

    return _torch_to_numpy_dtype[dtype]


class CircularBuffer(object):
    def __init__(self, size):
        self.index = 0
        self.size = size
        self._data = []

    def append(self, value):
        if len(self._data) == self.size:
            self._data[self.index] = value
        else:
            self._data.append(value)
        self.index = (self.index + 1) % self.size

    def is_empty(self):
        return self._data == []

    def not_empty(self):
        return not self.is_empty()

    def is_full(self):
        return len(self) == self.size

    def not_full(self):
        return not self.is_full()

    def __getitem__(self, key):
        """get element by index like a regular array"""
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __repr__(self):
        """return string representation"""
        return self._data.__repr__() + " (" + str(len(self._data)) + " items)"

    def __len__(self):
        return len(self._data)


class HeapQueue:
    def __init__(self, maxlen):
        self.q = []
        self.maxlen = maxlen

    def push(self, item):
        if len(self.q) < self.maxlen:
            heapq.heappush(self.q, item)
        else:
            heapq.heappushpop(self.q, item)

    def flush(self):
        q = self.q
        self.q = []
        return q

    def __len__(self):
        return len(self.q)


class DotDict(dict):
    """
    Dictionary to access attributes
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # Support pickling
    def __getstate__(obj):
        return dict(obj.items())

    def __setstate__(cls, attributes):
        return DotDict(**attributes)
