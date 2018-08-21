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
import torch
from gym import spaces

from adept.preprocess.ops import CastToFloat, GrayScaleAndMoveChannel, ResizeTo84x84, Divide255, FrameStack
from adept.preprocess.observation import ObsPreprocessor
from ._wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv
)
from ._base import BaseEnvironment, Spaces


def make_atari_env(env_id, skip_rate, max_ep_length, do_zscore_norm, do_frame_stack, seed):
    def _f():
        env = atari_env(env_id, skip_rate, max_ep_length, do_zscore_norm, do_frame_stack, seed)
        return env
    return _f


def atari_env(env_id, skip_rate, max_ep_length, do_zscore_norm, do_frame_stack, seed):
    env = gym.make(env_id)
    if 'NoFrameskip' in env_id:
        assert 'NoFrameskip' in env.spec.id
        env._max_episode_steps = max_ep_length * skip_rate
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=skip_rate)
    else:
        env._max_episode_steps = max_ep_length
    if hasattr(env.unwrapped, 'ale'):
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
    env.seed(seed)
    env = AdeptGymEnv(env, do_frame_stack)
    return env


class AdeptGymEnv(BaseEnvironment):
    def __init__(self, env, do_frame_stack):
        self.gym_env = env
        cpu_ops = [GrayScaleAndMoveChannel(), ResizeTo84x84()]
        if do_frame_stack:
            cpu_ops.append(FrameStack(4))
        self._cpu_preprocessor = ObsPreprocessor(cpu_ops, Spaces.from_gym(env.observation_space))
        self._gpu_preprocessor = ObsPreprocessor(
            [CastToFloat(), Divide255()],
            self._cpu_preprocessor.observation_space
        )
        self._observation_space = self._gpu_preprocessor.observation_space
        self._action_space = Spaces.from_gym(env.action_space)
        self._gym_obs_space = env.observation_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

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
        obs = self.gym_env.reset(**kwargs)
        return self._wrap_observation(obs)

    def close(self):
        self.gym_env.close()

    def __getattr__(self, item):
        return self.gym_env.__dict__[item]

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
