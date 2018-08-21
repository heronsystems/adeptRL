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
from copy import deepcopy


class ObsPreprocessor:
    def __init__(self, ops, observation_space):
        updated_obs_space = deepcopy(observation_space)

        nbr = updated_obs_space.names_by_rank
        ebn = updated_obs_space.entries_by_name

        for op in ops:
            for rank, names in nbr.items():
                for name in names:
                    if op.filter(name, rank):
                        ebn[name] = op.update_space(ebn[name])

        self.ops = ops
        self.observation_space = updated_obs_space

    def __call__(self, obs, device=None):
        nbr = self.observation_space.names_by_rank
        for op in self.ops:
            for rank, names in nbr.items():
                for name in names:
                    if device is not None:
                        new_obs = obs[name].to(device)
                    else:
                        new_obs = obs[name]

                    if op.filter(name, rank):
                        obs[name] = op.update_obs(new_obs)
        return obs
