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


class ExperienceReplay(BaseExperience):
    def __init__(self, batch_size, rollout_len, max_len, reward_normalizer, keys):
        super(RolloutCache, self).__init__()
        for k in keys:
            self[k] = []
        self['states'] = []
        self['rewards'] = []
        self['terminals'] = []
        self.max_len = max_len
        self.batch_size = batch_size
        self.rollout_len = rollout_len
        self.reward_normalizer = reward_normalizer

    def write_forward(self, **kwargs):
        for k, v in kwargs.items():
            self[k].append(v)
            if len(self[k]) > self.max_len:
                # TODO: this is really slow according to https://wiki.python.org/moin/TimeComplexity
                del self[k][0]

    def write_env(self, obs, rewards, terminals, infos):
        self['states'].append(obs)
        if len(self['states']) > self.max_len:
            # TODO: this is really slow according 
            del self['states'][0]
        self['rewards'].append(rewards)
        if len(self['rewards']) > self.max_len:
            # TODO: this is really slow according 
            del self['rewards'][0]
        self['terminals'].append(terminals)
        if len(self['terminals']) > self.max_len:
            # TODO: this is really slow according 
            del self['terminals'][0]

    def read(self):
        max_ind = len(self) - 2
        pass

    def is_ready(self):
        pass

    def __len__(self):
        return len(self['rewards'])

class PrioritizedExperienceReplay(BaseExperience):

    def write_forward(self, items):
        pass

    def write_env(self, obs, rewards, terminals, infos):
        pass

    def read(self):
        pass

    def is_ready(self):
        pass
