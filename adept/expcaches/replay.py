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
from ._base import BaseExperience
import numpy as np
from adept.utils import listd_to_dlist
from collections import namedtuple


class ExperienceReplay(dict, BaseExperience):
    def __init__(self, nb_env, batch_size, rollout_len, max_len, reward_normalizer, keys):
        super().__init__()
        for k in keys:
            self[k] = []
        self['obs'] = []
        self['rewards'] = []
        self['terminals'] = []
        self.nb_env = nb_env
        self.max_len = max_len
        self.batch_size = batch_size
        self.rollout_len = rollout_len
        self.min_length = self.batch_size * 10
        self.reward_normalizer = reward_normalizer

    def write_forward(self, **kwargs):
        for k, v in kwargs.items():
            self[k].append(v)
            if len(self[k]) > self.max_len:
                # TODO: this is really slow according to https://wiki.python.org/moin/TimeComplexity
                del self[k][0]

    def write_env(self, obs, rewards, terminals, infos):
        self['obs'].append(obs)
        if len(self['obs']) > self.max_len:
            # TODO: this is really slow according 
            del self['obs'][0]
        self['rewards'].append(rewards)
        if len(self['rewards']) > self.max_len:
            # TODO: this is really slow according 
            del self['rewards'][0]
        self['terminals'].append(terminals)
        if len(self['terminals']) > self.max_len:
            # TODO: this is really slow according 
            del self['terminals'][0]

    def read(self):
        max_ind = len(self) - 2 - self.rollout_len
        start_indexes = np.random.randint(0, max_ind, size=self.batch_size)
        end_indexes = start_indexes + self.rollout_len
        env_ind = np.random.randint(0, self.nb_env, size=self.batch_size)
        rollout = {}
        for key in self.keys():
            data = self.take(self[key], start_indexes, end_indexes, env_ind)

        rollout = {k: self.take(v, start_indexes, end_indexes, env_ind) for k, v in self.items()}
        rollout['last_obs'] = self.take_single(self['obs'], end_indexes + 1, env_ind)
        rollout['last_internals'] = self.take_single(self['internals'], end_indexes + 1, env_ind)
        # returns rollout as a named tuple
        rollout = namedtuple(self.__class__.__name__, rollout.keys())(**rollout)
        return rollout

    def is_ready(self):
        return len(self['rewards']) > self.min_length

    def __len__(self):
        return len(self['rewards'])

    @staticmethod
    def take(list, inds, end_inds, worker_inds):
        lslice = [list[i:end_i] for i, end_i in zip(inds, end_inds)]
        if isinstance(lslice[0], dict):
            slice_dict = listd_to_dlist(lslice)
            new_dict = {}
            for k, v in slice_dict.items():
                new_dict[k] = [v[i][w_ind] for i, w_ind in zip(range(len(lslice)), worker_inds)]
            return new_dict
        else:
            return [lslice[i][w_ind] for i, w_ind in zip(range(len(lslice)), worker_inds)]

    @staticmethod
    def take_single(list, inds, worker_inds):
        if isinstance(list[0], dict):
            slice_dict = listd_to_dlist(list)
            new_dict = {}
            for k, v in slice_dict.items():
                new_dict[k] = [v[i][w_ind] for i, w_ind in zip(inds, worker_inds)]
            return new_dict
        else:
            return [list[i][worker_ind] for i, worker_ind in zip(inds, worker_inds)]

class PrioritizedExperienceReplay(BaseExperience):

    def write_forward(self, items):
        pass

    def write_env(self, obs, rewards, terminals, infos):
        pass

    def read(self):
        pass

    def is_ready(self):
        pass
