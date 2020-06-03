# Copyright (C) 2020 Heron Systems, Inc.
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
import random
from collections import namedtuple

import torch
from adept.utils import dlist_to_listd
from adept.exp.rollout import Rollout
from adept.exp.base.exp_module import ExpModule


class MultiRollout(object):
    args = {"max_cache_size": 3}

    def __init__(self, spec_builder, rollout_len, max_cache_size):
        super(MultiRollout, self).__init__()
        self.spec_builder = spec_builder
        self.rollout_len = rollout_len
        self.max_cache_size = max_cache_size

        self.cur_rollout_idx = -1
        self.rollouts = []

    # TODO: move max_cache_size to args
    @classmethod
    def from_args(cls, args, spec_builder, max_cache_size):
        return cls(spec_builder, args.rollout_len, max_cache_size)

    def _create_rollout(self):
        return {
            "rollout": Rollout(self.spec_builder, self.rollout_len),
            "metas": {},
            "valid": False,
        }

    def sample(self):
        assert self.rollouts[self.cur_rollout_idx]["valid"]
        ix = random.randint(0, len(self) - 1)
        return self.rollouts[ix]

    def current_rollout(self):
        return self.rollouts[self.cur_rollout_idx]["rollout"]

    def start_next_rollout(self):
        self.cur_rollout_idx = (self.cur_rollout_idx + 1) % self.max_cache_size

        if self.cur_rollout_idx >= len(self.rollouts):
            self.rollouts.append(self._create_rollout())

        self.clear()

    def write_exps(self, exps):
        self.current_rollout().write_exps(exps)
        self.rollouts[self.cur_rollout_idx]["valid"] = True

    def write_meta(self, meta):
        self.rollouts[self.cur_rollout_idx]["metas"] = meta
        self.rollouts[self.cur_rollout_idx]["valid"] = True

    def clear(self):
        self.current_rollout().clear()
        self.rollouts[self.cur_rollout_idx]["metas"] = {}
        self.rollouts[self.cur_rollout_idx]["valid"] = False

    def is_ready(self):
        return self.current_rollout().is_ready()

    def __len__(self):
        return len(self.rollouts)

    def to(self, device):
        self.current_rollout().to(device)
        return self
