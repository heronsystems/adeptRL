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
from collections import namedtuple


class ACRollout(dict, ExpModule):
    args = {'rollout_len': 20}

    def __init__(self, reward_normalizer, rollout_len):
        super(ACRollout, self).__init__()
        assert type(rollout_len == int)
        self['states'] = []
        self['rewards'] = []
        self['terminals'] = []
        self['values'] = []
        self['log_probs'] = []
        self['entropies'] = []
        self.rollout_len = rollout_len
        self.reward_normalizer = reward_normalizer

    @classmethod
    def from_args(cls, args, reward_normalizer):
        return cls(reward_normalizer, args.rollout_len)

    def write_actor(self, experience):
        for k, v in experience.items():
            if k not in self:
                raise KeyError(f'Incompatible rollout key: {k}')
            self[k].append(v)

    def write_env(self, obs, rewards, terminals, infos):
        rewards = self.reward_normalizer(rewards)
        self['states'].append(obs)
        self['rewards'].append(rewards)
        # TODO: rename as terminals_mask or don't mask here
        self['terminals'].append(terminals)

    def read(self):
        # returns rollout as a named tuple
        return namedtuple(self.__class__.__name__, self.keys())(**self)

    def clear(self):
        for k in self.keys():
            self[k] = []

    def is_ready(self):
        return len(self) == self.rollout_len

    def __len__(self):
        return len(self['rewards'])
