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
from collections import namedtuple

import torch
from adept.exp.base.exp_module import ExpModule


class Rollout(dict, ExpModule):
    args = {'rollout_len': 20}

    def __init__(self, reward_normalizer, spec_builder, rollout_len):
        super(Rollout, self).__init__()
        self.spec = spec_builder(rollout_len)
        self.obs_keys = spec_builder.obs_keys
        self.action_keys = spec_builder.action_keys
        self.internal_keys = spec_builder.internal_keys
        self.exp_keys = spec_builder.exp_keys
        self.reward_normalizer = reward_normalizer
        self.rollout_len = rollout_len

        self.has_obs = all([
            obs_key in self.spec for obs_key in self.obs_keys
        ])
        self.has_actions = all([
            act_key in self.spec for act_key in self.action_keys
        ])
        self.has_internals = all([
            internal_key in self.spec for internal_key in self.internal_keys
        ])
        self.cur_idx = 0

        for k in self.spec.keys():
            self[k] = self._init_key(k)

    @classmethod
    def from_args(cls, args, reward_normalizer, spec_builder):
        return cls(
            reward_normalizer, spec_builder, args.rollout_len
        )

    def write_actor(self, experience, no_env=False):
        for k in self.exp_keys:
            self[k][self.cur_idx] = experience[k]

        if no_env:
            self.cur_idx += 1

    def write_env(self, obs, rewards, terminals, infos):
        rewards = self.reward_normalizer(rewards)
        if self.has_obs:
            for k in self.obs_keys:
                self[k] = obs[k]
        self['rewards'][self.cur_idx] = rewards
        self['terminals'][self.cur_idx] = terminals

        self.cur_idx += 1

    def write_next_obs(self, next_obs):
        if self.has_obs:
            for k in self.obs_keys:
                self[k][self.rollout_len] = next_obs[k]
        else:
            raise Exception(
                'This rollout does not store observations. Select a valid actor'
            )

    def read(self):
        tmp = {}
        if self.has_obs:
            tmp['observations'] = {k: self[k][:-1] for k in self.obs_keys}
            tmp['next_observation'] = {k: self[k][-1] for k in self.obs_keys}
        if self.has_actions:
            tmp['actions'] = {k: self[k] for k in self.action_keys}
        if self.has_internals:
            tmp['internals'] = {k: self[k] for k in self.internal_keys}
        for k in self.exp_keys:
            tmp[k] = self[k]
        tmp['rewards'] = self['rewards']
        tmp['terminals'] = self['terminals']
        return namedtuple(self.__class__.__name__, tmp.keys())(**tmp)

    def clear(self):
        for k, tensor_list in self.items():
            for i in range(len(tensor_list)):
                self[k][i] = self[k][i].detach()
        self.cur_idx = 0

    def is_ready(self):
        return self.cur_idx == self.rollout_len

    def __len__(self):
        return self.rollout_len

    def to(self, device):
        for k, tensor_list in self.items():
            for tensor_idx, tensor in enumerate(tensor_list):
                self[k][tensor_idx] = tensor.to(device)
        return self

    def _init_key(self, key):
        return [
            torch.zeros(*self.spec[key][1:])
            for _ in range(self.spec[key][0])
        ]
