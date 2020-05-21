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


class ExpSpecBuilder:
    def __init__(
        self, obs_keys, act_keys, internal_keys, key_types, exp_keys, build_fn
    ):
        self.obs_keys = sorted(obs_keys.keys())
        self.action_keys = sorted(act_keys.keys())
        self.internal_keys = sorted(internal_keys.keys())
        self.key_types = key_types
        self.exp_keys = sorted(exp_keys)
        self.build_fn = build_fn

    def __call__(self, rollout_len):
        return self.build_fn(rollout_len)
