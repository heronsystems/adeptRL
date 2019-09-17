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
from adept.utils import dlist_to_listd
from torch import distributed as dist
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
        self.key_types = spec_builder.key_types
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

        self.sorted_keys = sorted(self.keys())

    @classmethod
    def from_args(cls, args, reward_normalizer, spec_builder):
        return cls(
            reward_normalizer, spec_builder, args.rollout_len
        )

    def write_actor(self, experience, no_env=False):
        for k in experience.keys() & self.keys():
            # exp_shape = self[k][self.cur_idx].shape
            # write_shape = experience[k].shape
            # if exp_shape != write_shape:
            #     print(f'actor shape mismatch {k} {exp_shape} {write_shape}')
            self[k][self.cur_idx] = experience[k]

        if no_env:
            self.cur_idx += 1

    def write_env(self, obs, rewards, terminals, infos):
        rewards = self.reward_normalizer(rewards)
        if self.has_obs:
            for k in self.obs_keys:
                # exp_shape = self[k][self.cur_idx].shape
                # write_shape = obs[k].shape
                # if exp_shape != write_shape:
                #     print(f'obs {k} {exp_shape} {write_shape}')
                self[k][self.cur_idx] = obs[k]
        # exp_shape = self['rewards'][self.cur_idx].shape
        # write_shape = rewards.shape
        # if exp_shape != write_shape:
        #     print(f'rewards shape mismatch {exp_shape} {write_shape}')
        self['rewards'][self.cur_idx] = rewards
        # exp_shape = self['terminals'][self.cur_idx].shape
        # write_shape = terminals.shape
        # if exp_shape != write_shape:
        #     print(f'terminals shape mismatch')
        self['terminals'][self.cur_idx] = terminals

        self.cur_idx += 1

    def write_exps(self, exps):
        other_exp_keys = exps[0].keys()
        for rollout_idx in range(self.rollout_len):
            for k in self.sorted_keys:
                if k in other_exp_keys:
                    tensors_to_cat = []
                    for exp in exps:
                        tensors_to_cat.append(exp[k][rollout_idx])
                    cat = torch.cat(tensors_to_cat)
                    # exp_shape = self[k][rollout_idx].shape
                    # write_shape = cat.shape
                    # if self[k][rollout_idx].shape != cat.shape:
                    #     print(f'write_exps shape mismatch {k} {exp_shape} {write_shape}')
                    self[k][rollout_idx] = cat

    def write_next_obs(self, obs):
        if self.has_obs:
            for k in self.obs_keys:
                self[k][-1] = obs[k]

    def read(self):
        tmp = {}
        if self.has_obs:
            tmp['observations'] = dlist_to_listd({k: self[k][:-1] for k in self.obs_keys})
            tmp['next_observation'] = {k: self[k][-1] for k in self.obs_keys}
        if self.has_actions:
            tmp['actions'] = dlist_to_listd({k: self[k] for k in self.action_keys})
        if self.has_internals:
            tmp['internals'] = dlist_to_listd({k: self[k] for k in self.internal_keys})
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
        zs = torch.zeros(*self.spec[key][1:])
        if self.key_types[key] == 'long':
            zs = zs.long()
        elif self.key_types[key] == 'float':
            pass
        else:
            raise Exception(f'Unrecognized key_type: {self.key_types[key]}')
        return [zs for _ in range(self.spec[key][0])]

    def sync(self, src, grp, async_op=False):
        handles = []
        for k in self.sorted_keys:
            for t in self[k]:
                if t.dtype == torch.bool:
                    t = t.float()  # cast terminals
                handles.append(
                    dist.broadcast(t, src=src, group=grp, async_op=True)
                )

        if not async_op:
            [handle.wait() for handle in handles]

        return handles
