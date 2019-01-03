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
import gym
import torch
from gym import spaces

from adept.environments._spaces import Spaces
from adept.preprocess.observation import ObsPreprocessor
from adept.preprocess.ops import (
    CastToFloat, GrayScaleAndMoveChannel, ResizeTo84x84, Divide255, FrameStack
)
from adept.environments.env_plugin import EnvPlugin
from ._gym_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv
)


class AdeptGymEnv(EnvPlugin):
    """
    Converts gym observations to dictionaries and reads actions from
    dictionaries instead of numpy arrays. This allows the Gym Env to
    communicate properly with an EnvManager.
    """
    args = {
        "max_episode_length": 10000,
        "skip_rate": 4,
        "noop_max": 30
    }

    def __init__(self, env, do_frame_stack):
        # Define the preprocessing operations to be performed on observations
        # CPU Ops
        cpu_ops = [GrayScaleAndMoveChannel(), ResizeTo84x84()]
        if do_frame_stack:
            cpu_ops.append(FrameStack(4))
        cpu_preprocessor = ObsPreprocessor(
            cpu_ops, Spaces.from_gym(env.observation_space)
        )

        # GPU Ops
        gpu_preprocessor = ObsPreprocessor(
            [CastToFloat(), Divide255()], cpu_preprocessor.observation_space
        )

        action_space = Spaces.from_gym(env.action_space)

        super(AdeptGymEnv, self).__init__(
            action_space, cpu_preprocessor, gpu_preprocessor
        )

        self.gym_env = env
        self._gym_obs_space = env.observation_space

    @classmethod
    def from_args(cls, args, seed, **kwargs):
        # TODO fix this hack
        do_frame_stack = 'Linear' in args.netbody
        env = gym.make(args.env)
        if hasattr(env.unwrapped, 'ale'):
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = NoopResetEnv(env, noop_max=args.noop_max)
            env = EpisodicLifeEnv(env)
        if 'NoFrameskip' in args.env:
            env._max_episode_steps = args.max_episode_length * \
                                     args.skip_rate
            env = MaxAndSkipEnv(env, skip=args.skip_rate)
        else:
            env._max_episode_steps = args.max_episode_length
        env.seed(seed)
        return cls(env, do_frame_stack)

    def step(self, action):
        obs, reward, done, info = self.gym_env.step(self._wrap_action(action))
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
            # one hot encode net1d inputs
            longs = torch.from_numpy(observation)
            if longs.dim() > 2:
                raise ValueError(
                    'observation is not net1d, too many dims: ' +
                    str(longs.dim())
                )
            elif len(longs.dim()) == 1:
                longs = longs.unsqueeze(1)
            one_hot = torch.zeros(observation.size(0), space.n)
            one_hot.scatter_(1, longs, 1)
            return self.cpu_preprocessor({'Discrete': one_hot})
        elif isinstance(space, spaces.MultiBinary):
            return self.cpu_preprocessor(
                {
                    'MultiBinary': torch.from_numpy(observation)
                }
            )
        elif isinstance(space, spaces.Dict):
            return self.cpu_preprocessor(
                {
                    name: torch.from_numpy(obs)
                    for name, obs in observation.items()
                }
            )
        elif isinstance(space, spaces.Tuple):
            return self.cpu_preprocessor(
                {
                    idx: torch.from_numpy(obs)
                    for idx, obs in enumerate(observation)
                }
            )
        else:
            raise NotImplementedError

    def _wrap_action(self, action):
        return action['Discrete']
