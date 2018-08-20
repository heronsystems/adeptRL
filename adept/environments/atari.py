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

from ._wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, FrameStack, WarpFrame, Divide255,
    ZScoreEnv
)


def make_atari_env(env_id, skip_rate, max_ep_length, do_zscore_norm, do_frame_stack, seed):
    def _f():
        env = atari_env(env_id, skip_rate, max_ep_length, do_zscore_norm, do_frame_stack)
        env.seed(seed)
        return env
    return _f


def atari_env(env_id, skip_rate, max_ep_length, do_zscore_norm, do_frame_stack):
    env = gym.make(env_id)
    if isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 3:
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
        env = WarpFrame(env)
        if do_zscore_norm:
            env = ZScoreEnv(env)
        else:
            env = Divide255(env)
        if do_frame_stack:
            env = FrameStack(env, 4)
    env = DictTensorObs(env)
    return env


class DictTensorObs(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        return {'obs': torch.from_numpy(observation)}
