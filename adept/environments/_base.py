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
import abc
from collections import namedtuple

import numpy as np
from gym import spaces

Space = namedtuple('Space', ['shape', 'low', 'high', 'dtype'])


# agent gets an obs_space and action_space, which both have a process method
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
            return {name: Spaces._detect_gym_spaces(s) for name, s in space.spaces.items()}
        elif isinstance(space, spaces.Tuple):
            return {idx: Spaces._detect_gym_spaces(s) for idx, s in enumerate(space.spaces)}


class BaseEnvironment(abc.ABC):
    @property
    @abc.abstractmethod
    def observation_space(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_space(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cpu_preprocessor(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gpu_preprocessor(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError


class ObservationWrapper(metaclass=abc.ABCMeta):
    def __init__(self, env):
        self.env = env

    def __getattr__(self, item):
        return getattr(self.env, item)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.wrap_observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.wrap_observation(observation)

    @abc.abstractmethod
    def wrap_observation(self, obs):
        raise NotImplementedError
