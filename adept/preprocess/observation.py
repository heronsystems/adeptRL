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
from copy import deepcopy


class ObsPreprocessor:
    def __init__(self, ops, observation_space, observation_dtypes=None):
        """
        :param ops: List[Operation]
        :param observation_space: Dict[ObsKey, Shape]
        :param observation_dtypes: Dict[ObsKey, dtype_str]
        """
        cur_space = deepcopy(observation_space)
        cur_dtypes = deepcopy(observation_dtypes)

        rank_to_names = self._bld_rank_to_names(observation_space)

        for op in ops:
            if op.name_filters:
                names = op.name_filters
            elif op.rank_filters:
                names = []
                for rank in op.rank_filters:
                    names += rank_to_names[rank]
            else:
                names = list(cur_space.keys())

            cur_space = self._update(names, cur_space, op.update_shape)
            if observation_dtypes:
                cur_dtypes = self._update(names, cur_dtypes, op.update_dtype)
            rank_to_names = self._bld_rank_to_names(observation_space)

        self.ops = ops
        self.observation_space = cur_space
        self.observation_dtypes = cur_dtypes
        self.rank_to_names = rank_to_names

    def __call__(self, obs):
        for op in self.ops:
            obs = op.update_obs(obs)
        return obs

    def reset(self):
        for o in self.ops:
            o.reset()

    def _bld_rank_to_names(self, obs_space):
        d = {1: [], 2: [], 3: [], 4: []}
        for name, shape in obs_space.items():
            d[len(shape)].append(name)
        return d

    def _update(self, names, prev, fn):
        cur = {}
        for name in names:
            cur[name] = prev[name]
            del prev[name]
        update = fn(cur)
        return {**prev, **update}
