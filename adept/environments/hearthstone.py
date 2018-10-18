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
import gym
import gym_hearthstone
import torch
from gym import spaces

from adept.preprocess.ops import CastToFloat, Divide255
from ._base import BaseEnvironment, Spaces
import numpy as np

def make_hearthstone_env():
    def _f():
        env = hearthstone_env()
        return env
    return _f


def hearthstone_env():
    env = gym.make('Hearthstone-v0')
    env = AdeptGymEnv(env)
    return env


class AdeptHearthstoneEnv(BaseEnvironment):
    def __init__(self, env):
        self.gym_env = env
        cpu_ops = []
        self._cpu_preprocessor = ObsPreprocessor(cpu_ops, Spaces.from_gym(env.observation_space))
        self._gpu_preprocessor = ObsPreprocessor(
            [CastToFloat(), Divide255()],
            self._cpu_preprocessor.observation_space
        )
        self._observation_space = self._gpu_preprocessor.observation_space
        self._gym_obs_space = env.observation_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return Spaces.from_gym(env.get_possible_actions())

    @property
    def cpu_preprocessor(self):
        return self._cpu_preprocessor

    @property
    def gpu_preprocessor(self):
        return self._gpu_preprocessor

    def step(self, action):
        obs, reward, done, info = self.gym_env.step(action)
        return self._wrap_observation(obs), reward, done, info

    def reset(self, **kwargs):
        self.cpu_preprocessor.reset()
        obs = self.gym_env.reset(**kwargs)
        return self._wrap_observation(obs)

    def close(self):
        self.gym_env.close()

    def __getattr__(self, item):
        return getattr(self.gym_env, item)

    def _wrap_observation(self, observation):
        space = self._gym_obs_space
        if isinstance(space, spaces.Box):
            return self.cpu_preprocessor({'Box': torch.from_numpy(observation)})
        elif isinstance(space, spaces.Discrete):
            # one hot encode discrete inputs
            longs = torch.from_numpy(observation)
            if longs.dim() > 2:
                raise ValueError('observation is not discrete, too many dims: ' + str(longs.dim()))
            elif len(longs.dim()) == 1:
                longs = longs.unsqueeze(1)
            one_hot = torch.zeros(observation.size(0), space.n)
            one_hot.scatter_(1, longs, 1)
            return self.cpu_preprocessor({'Discrete': one_hot})
        elif isinstance(space, spaces.MultiBinary):
            return self.cpu_preprocessor({'MultiBinary': torch.from_numpy(observation)})
        elif isinstance(space, spaces.Dict):
            return self.cpu_preprocessor({name: torch.from_numpy(obs) for name, obs in observation.items()})
        elif isinstance(space, spaces.Tuple):
            return self.cpu_preprocessor({idx: torch.from_numpy(obs) for idx, obs in enumerate(observation)})
        else:
            raise NotImplementedError
