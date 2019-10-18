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
from adept.exp.base.exp_module import ExpModule
from adept.utils import listd_to_dlist
from collections import namedtuple
import torch
import numpy as np

# Some code is taken from OpenAI. MIT license
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import random
from operator import itemgetter


class ExperienceReplay(ExpModule):
    args = {
        'exp_size': 15625,
        'exp_min_size': 1000,
        'rollout_len': 20,
        'exp_update_rate': 1
    }

    def __init__(self, spec_builder, size, min_size, rollout_len, update_rate):
        """
        Size and min_size are not dependent on nb of envs so if desired size is 
        1mil samples with 64 envs size should be 1mil/64 = 15625
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        super(ExperienceReplay, self).__init__()
        assert type(size == int)
        assert type(rollout_len == int)

        self.spec = spec_builder(rollout_len)
        self.obs_keys = spec_builder.obs_keys
        self.exp_keys = spec_builder.exp_keys
        self.key_types = spec_builder.key_types
        self.keys = self.exp_keys

        self._storage = []
        self._full = False
        self._maxsize = size
        self._update_rate = update_rate
        self._minsize = min_size
        self._next_idx = 0
        self._keys = ['observations', 'rewards', 'terminals'] + self.keys

        self.rollout_len = rollout_len
        self.device = 'cpu'
        self.target_device = self.device

    @classmethod
    def from_args(cls, args, spec_builder):
        return cls(
            spec_builder, args.exp_size,
            args.exp_min_size, args.rollout_len,
            args.exp_update_rate
        )

    def __len__(self):
        if not self._full:
            return len(self._storage)
        else:
            return self._maxsize

    def write_actor(self, experience):
        # convert to cpu
        exp_storage_dev = self._exp_to_dev(experience, self.device)
        # write forward occurs before write env so append here
        if not self._full and self._next_idx >= len(self._storage):
            self._storage.append(exp_storage_dev)
        else:
            self._storage[self._next_idx] = exp_storage_dev

    def _exp_to_dev(self, experience, device):
        # TODO: this should be a generic function somewhere
        exp = {}
        for k, v in experience.items():
            if isinstance(v, dict):
                on_d = {d_key: d_v.to(device) for d_key, d_v in v.items()}
            # tensor
            else:
                on_d = v.to(device)
            exp[k] = on_d
        return exp

    def write_env(self, obs, rewards, terminals, infos):
        # forward already written, add env info then increment
        dict_at_ind = self._storage[self._next_idx]
        self._next_idx = int((self._next_idx + 1) % self._maxsize)
        # when index wraps exp is full
        if self._next_idx == 0:
            self._full = True

        dict_at_ind['observations'] = {k: v.cpu() for k, v in obs.items()}
        dict_at_ind['rewards'] = rewards.cpu()
        dict_at_ind['terminals'] = terminals.cpu()

    def read(self):
        exp_list, last_obs, is_weights = self._sample()
        exp_dev_list = [self._exp_to_dev(e, self.target_device) for e in exp_list]
        # will be list of dicts, convert to dict of lists
        dict_of_list = listd_to_dlist(exp_dev_list)
        # get the next observation
        dict_of_list['next_observation'] = last_obs
        # importance sampling weights
        dict_of_list['importance_sample_weights'] = is_weights

        # return named tuple
        return namedtuple(self.__class__.__name__, ['importance_sample_weights', 'next_observation'] + self._keys)(**dict_of_list)

    def _sample(self):
        # TODO: support burn in
        # if full indexes may wrap 
        if self._full:
            # wrap index starting from current index to full size
            min_ind = self._next_idx
            max_ind = min_ind + (self._maxsize - (self.rollout_len + 1))
            index = random.randint(min_ind, max_ind)
            # range is exclusive of end so last_index == end_index
            end_index = index + self.rollout_len
            last_index = int((end_index) % self._maxsize)
            indexes = (np.arange(index, end_index) % self._maxsize).astype(int)
        else:
            # sample an index and get the next sequential samples of len rollout_len
            index = random.randint(0, len(self._storage) - (self.rollout_len + 1))
            end_index = index + self.rollout_len
            indexes = list(range(index, end_index))
            # range is exclusive of end so last_index == end_index
            last_index = end_index
        # assert (self._maxsize + (self._next_idx - 1)) % self._maxsize not in indexes

        weights = np.ones(self.rollout_len)
        return itemgetter(*indexes)(self._storage), self._storage[last_index]['observations'], weights

    def is_ready(self):
        # plus 2 to include next observations
        if len(self) > self._minsize and len(self) > self.rollout_len + 2:
            return self._next_idx % self._update_rate == 0
        return False

    def clear(self):
        pass

    def update_priorities(self, *args, **kwargs):
        pass

    def to(self, device):
        self.target_device = device


class PrioritizedExperienceReplay(ExpModule):
    def write_actor(self, items):
        pass

    def write_env(self, obs, rewards, terminals, infos):
        pass

    def read(self):
        pass

    def is_ready(self):
        pass

