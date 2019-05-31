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
from adept.expcaches._base import BaseExperience
from adept.expcaches.segmenttree import MinSegmentTree, SumSegmentTree
from adept.utils import listd_to_dlist
from collections import namedtuple
import torch
import numpy as np

# Some code is taken from OpenAI. MIT license
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import random
from operator import itemgetter


class ExperienceReplay(BaseExperience):
    def __init__(self, size, min_size, nb_rollout, update_rate, reward_normalizer, keys):
        """
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        super(ExperienceReplay, self).__init__()
        assert type(size == int)
        assert type(nb_rollout == int)

        self._storage = []
        self._full = False
        self._maxsize = size
        self._update_rate = update_rate
        self._minsize = min_size
        self._next_idx = 0
        self._keys = ['states', 'rewards', 'terminals'] + keys

        self.nb_rollout = nb_rollout
        self.reward_normalizer = reward_normalizer

    def __len__(self):
        if not self._full:
            return len(self._storage)
        else:
            return self._maxsize

    def write_forward(self, **kwargs):
        # write forward occurs before write env so append here
        if not self._full and self._next_idx >= len(self._storage):
            self._storage.append(kwargs)
        else:
            self._storage[self._next_idx] = kwargs

    def write_env(self, obs, rewards, terminals, infos):
        # forward already written, add env info then increment
        dict_at_ind = self._storage[self._next_idx]
        self._next_idx = int((self._next_idx + 1) % self._maxsize)
        # when index wraps exp is full
        if self._next_idx == 0:
            self._full = True

        rewards = torch.tensor(
            [self.reward_normalizer(reward) for reward in rewards]
        )
        terminals = (
            1. - torch.from_numpy(np.array(terminals, dtype=np.float32))
        )

        dict_at_ind['states'] = obs
        dict_at_ind['rewards'] = rewards
        dict_at_ind['terminals'] = terminals

    def read(self):
        exp_list, last_obs, is_weights = self._sample()
        # will be list of dicts, convert to dict of lists
        dict_of_list = listd_to_dlist(exp_list)
        # get the next observation
        dict_of_list['next_obs'] = last_obs
        # importance sampling weights
        dict_of_list['importance_sample_weights'] = is_weights
        # return named tuple
        return namedtuple(self.__class__.__name__, ['importance_sample_weights', 'next_obs'] + self._keys)(**dict_of_list)

    def _sample(self):
        # TODO: support burn in
        # if full indexes may wrap 
        if self._full:
            # wrap index starting from current index to full size
            min_ind = self._next_idx
            max_ind = min_ind + (self._maxsize - (self.nb_rollout + 1))
            index = random.randint(min_ind, max_ind)
            # range is exclusive of end so last_index == end_index
            end_index = index + self.nb_rollout
            last_index = int((end_index) % self._maxsize)
            indexes = (np.arange(index, end_index) % self._maxsize).astype(int)
        else:
            # sample an index and get the next sequential samples of len nb_rollout
            index = random.randint(0, len(self._storage) - (self.nb_rollout + 1))
            end_index = index + self.nb_rollout
            indexes = list(range(index, end_index))
            # range is exclusive of end so last_index == end_index
            last_index = end_index
        # assert (self._maxsize + (self._next_idx - 1)) % self._maxsize not in indexes

        weights = np.ones(self.nb_rollout)
        return itemgetter(*indexes)(self._storage), self._storage[last_index]['states'], weights

    def is_ready(self):
        # plus 2 to include next states
        if len(self) > self._minsize and len(self) > self.nb_rollout + 2:
            return self._next_idx % self._update_rate == 0
        return False

    def clear(self):
        pass

    def update_priorities(self, *args, **kwargs):
        pass


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, alpha, size, min_size, nb_rollout, update_rate, reward_normalizer, keys):
        """
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        super(PrioritizedExperienceReplay, self).__init__(size, min_size, nb_rollout, update_rate, reward_normalizer, keys)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._last_sample_idx = 0

    def write_env(self, *args, **kwargs):
        idx = self._next_idx
        super().write_env(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self):
        res = []
        if self._full:
            p_total = self._it_sum.sum(0, self._maxsize - 1)
        else:
            p_total = self._it_sum.sum(0, len(self._storage) - 1)
        mass = random.random() * p_total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def _sample(self, beta=0.6):
        # TODO: beta from agent, support burn in
        assert beta > 0

        index = self._sample_proportional()

        # try to fit a sequence to this index, with the index as early as possible
        if self._full:
            end_index = index + self.nb_rollout
            end_index_wrap = end_index % self._maxsize

            # if index is wrapped
            if self._next_idx - 1 < index:
                compare_index = self._maxsize + (self._next_idx - 1)
            else:
                compare_index = self._next_idx - 1

            # if in the sample rollout adjust starting index
            if compare_index >= index and compare_index <= end_index:
                end_index = compare_index
                index = end_index - self.nb_rollout
                last_index = int((end_index) % self._maxsize)
                indexes = (np.arange(index, end_index) % self._maxsize).astype(int)
            else:
                # wrap index starting from current index to full size
                end_index = index + self.nb_rollout
                last_index = int((end_index) % self._maxsize)
                indexes = (np.arange(index, end_index) % self._maxsize).astype(int)
        else:
            max_index = (self._next_idx - 1) - self.nb_rollout
            # if index is too far forward
            if index > max_index:
                index = max_index
                end_index = self._next_idx - 1
            else:
                end_index = index + self.nb_rollout
            indexes = range(index, end_index)
            # range is exclusive of end so last_index == end_index
            last_index = end_index
        # assert (self._maxsize + (self._next_idx - 1)) % self._maxsize not in indexes

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in indexes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.asarray(weights)

        self._last_sample_idx = index
        return itemgetter(*indexes)(self._storage), self._storage[last_index]['states'], weights

    def update_priorities(self, priorities):
        """Update priority of sampled transitions.
        Sets priority of transition of last sample
        Parameters
        ----------
        priorities: float
            Priority corresponding to
            transitions at the sampled index.
        """
        rollout_inds = self._rollout_inds_from_ind(self._last_sample_idx)
        for ind, p in zip(rollout_inds, priorities):
            self._it_sum[ind] = p ** self._alpha
            self._it_min[ind] = p ** self._alpha
            self._max_priority = max(self._max_priority, p)

    def _rollout_inds_from_ind(self, index):
        end_index = index + self.nb_rollout
        indexes = (np.arange(index, end_index) % self._maxsize).astype(int)
        return indexes

