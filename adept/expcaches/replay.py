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
from adept.utils import listd_to_dlist
from collections import namedtuple
import torch
import numpy as np

# Some code is taken from OpenAI MIT license
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import random


class ExperienceReplay(BaseExperience):
    def __init__(self, size, nb_rollout, reward_normalizer, keys):
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
        self._maxsize = size
        self._next_idx = 0
        self._keys = ['states', 'rewards', 'terminals'] + keys

        self.nb_rollout = nb_rollout
        self.reward_normalizer = reward_normalizer

    def __len__(self):
        return len(self._storage)

    def write_forward(self, **kwargs):
        # write forward occurs before write env so append here
        if self._next_idx >= len(self._storage):
            self._storage.append(kwargs)
        else:
            self._storage[self._next_idx] = kwargs

    def write_env(self, obs, rewards, terminals, infos):
        # forward already written, add env info then increment
        dict_at_ind = self._storage[self._next_idx]
        self._next_idx = int((self._next_idx + 1) % self._maxsize)

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
        # TODO: support burn in
        # sample an index and get the next sequential samples of len nb_rollout
        # minus two so last state fits
        index = random.randint(0, len(self._storage) - (self.nb_rollout + 2))
        end_index = index + self.nb_rollout
        exp_list = self._storage[index:end_index]
        # will be list of dicts, convert to dict of lists
        # TODO: should be dict of tensors?
        dict_of_list = listd_to_dlist(exp_list)
        # get the next observation
        dict_of_list['next_obs'] = self._storage[end_index + 1]['states']
        # return named tuple
        return namedtuple(self.__class__.__name__, ['next_obs'] + self._keys)(**dict_of_list)

    def is_ready(self):
        # plus 2 to include next states
        return len(self) > self.nb_rollout + 2

    def clear(self):
        pass


class PrioritizedExperienceReplay(BaseExperience):
    def write_forward(self, items):
        pass

    def write_env(self, obs, rewards, terminals, infos):
        pass

    def read(self):
        pass

    def is_ready(self):
        pass
