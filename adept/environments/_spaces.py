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
from gym import spaces


class Space(dict):
    def __init__(self, entries_by_name):
        super(Space, self).__init__(entries_by_name)

    @classmethod
    def from_gym(cls, gym_space):
        entries_by_name = Space._detect_gym_spaces(gym_space)
        return cls(entries_by_name)

    @staticmethod
    def _detect_gym_spaces(space):
        if isinstance(space, spaces.Discrete):
            return {'Discrete': (space.n, )}
        elif isinstance(space, spaces.MultiDiscrete):
            raise NotImplementedError
        elif isinstance(space, spaces.MultiBinary):
            return {'MultiBinary': (space.n, )}
        elif isinstance(space, spaces.Box):
            return {
                'Box': space.shape
            }
        elif isinstance(space, spaces.Dict):
            return {
                name: list(Space._detect_gym_spaces(s).values())[0]
                for name, s in space.spaces.items()
            }
        elif isinstance(space, spaces.Tuple):
            return {
                idx: list(Space._detect_gym_spaces(s).values())[0]
                for idx, s in enumerate(space.spaces)
            }
