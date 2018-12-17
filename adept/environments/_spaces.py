from collections.__init__ import namedtuple

import numpy as np
from gym import spaces

Space = namedtuple('Space', ['shape', 'low', 'high', 'dtype'])


class Spaces:
    def __init__(self, entries_by_name):
        self.entries_by_name = entries_by_name
        self.names_by_rank = {
            1: [],
            2: [],
            3: [],
            4: []
        }
        for name, entry in entries_by_name.items():
            self.names_by_rank[len(entry.shape)].append(name)

    @classmethod
    def from_gym(cls, gym_space):
        entries_by_name = Spaces._detect_gym_spaces(gym_space)
        return cls(entries_by_name)

    @staticmethod
    def _detect_gym_spaces(space):
        if isinstance(space, spaces.Discrete):
            return {'Discrete': Space([space.n], 0, 1, np.float32)}
        elif isinstance(space, spaces.MultiDiscrete):
            raise NotImplementedError
        elif isinstance(space, spaces.MultiBinary):
            return {'MultiBinary': Space([space.n], 0, 1, space.dtype)}
        elif isinstance(space, spaces.Box):
            return {'Box': Space(space.shape, 0., 255., space.dtype)}  # TODO, is it okay to hardcode 0, 255
        elif isinstance(space, spaces.Dict):
            return {name: list(Spaces._detect_gym_spaces(s).values())[0] for name, s in space.spaces.items()}
        elif isinstance(space, spaces.Tuple):
            return {idx: list(Spaces._detect_gym_spaces(s).values())[0] for idx, s in enumerate(space.spaces)}