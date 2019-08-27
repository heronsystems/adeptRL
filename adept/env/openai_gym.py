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

from adept.env._spaces import Space
from adept.preprocess.observation import ObsPreprocessor
from adept.preprocess.ops import (
    CastToFloat, GrayScaleAndMoveChannel, ResizeTo84x84, Divide255, FrameStack,
    FromNumpy)
from adept.env.base.env_module import EnvModule
from ._gym_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, FireResetEnv
)


ATARI_ENVS = [
    # 'AirRaidNoFrameskip-v4',
    'AlienNoFrameskip-v4',
    'AmidarNoFrameskip-v4',
    'AssaultNoFrameskip-v4',
    'AsterixNoFrameskip-v4',
    'AsteroidsNoFrameskip-v4',
    'AtlantisNoFrameskip-v4',
    'BankHeistNoFrameskip-v4',
    'BattleZoneNoFrameskip-v4',
    'BeamRiderNoFrameskip-v4',
    'BerzerkNoFrameskip-v4',
    'BowlingNoFrameskip-v4',
    'BoxingNoFrameskip-v4',
    'BreakoutNoFrameskip-v4',
    # 'CarnivalNoFrameskip-v4',
    'CentipedeNoFrameskip-v4',
    'ChopperCommandNoFrameskip-v4',
    'CrazyClimberNoFrameskip-v4',
    'DemonAttackNoFrameskip-v4',
    'DoubleDunkNoFrameskip-v4',
    # 'ElevatorActionNoFrameskip-v4',
    'EnduroNoFrameskip-v4',
    'FishingDerbyNoFrameskip-v4',
    'FreewayNoFrameskip-v4',
    'FrostbiteNoFrameskip-v4',
    'GopherNoFrameskip-v4',
    'GravitarNoFrameskip-v4',
    'HeroNoFrameskip-v4',
    'IceHockeyNoFrameskip-v4',
    'JamesbondNoFrameskip-v4',
    # 'JourneyEscapeNoFrameskip-v4',
    'KangarooNoFrameskip-v4',
    'KrullNoFrameskip-v4',
    'KungFuMasterNoFrameskip-v4',
    'MontezumaRevengeNoFrameskip-v4',
    'MsPacmanNoFrameskip-v4',
    'NameThisGameNoFrameskip-v4',
    'PhoenixNoFrameskip-v4',
    'PitfallNoFrameskip-v4',
    'PongNoFrameskip-v4',
    # 'PooyanNoFrameskip-v4',
    'PrivateEyeNoFrameskip-v4',
    'QbertNoFrameskip-v4',
    'RiverraidNoFrameskip-v4',
    'RoadRunnerNoFrameskip-v4',
    'RobotankNoFrameskip-v4',
    'SeaquestNoFrameskip-v4',
    'SkiingNoFrameskip-v4',
    'SolarisNoFrameskip-v4',
    'SpaceInvadersNoFrameskip-v4',
    'StarGunnerNoFrameskip-v4',
    'TennisNoFrameskip-v4',
    'TimePilotNoFrameskip-v4',
    'TutankhamNoFrameskip-v4',
    'UpNDownNoFrameskip-v4',
    'VentureNoFrameskip-v4',
    'VideoPinballNoFrameskip-v4',
    'WizardOfWorNoFrameskip-v4',
    'YarsRevengeNoFrameskip-v4',
    'ZaxxonNoFrameskip-v4'
]


class AdeptGymEnv(EnvModule):
    """
    Converts gym observations to dictionaries and reads actions from
    dictionaries instead of numpy arrays. This allows the Gym Env to
    communicate properly with an EnvManager.
    """
    args = {
        "max_episode_length": 10000,
        "skip_rate": 4,
        "noop_max": 30,
        "frame_stack": False
    }

    ids = ATARI_ENVS

    def __init__(self, env, do_frame_stack):
        # Define the preprocessing operations to be performed on observations
        # CPU Ops
        cpu_ops = [FromNumpy(), GrayScaleAndMoveChannel(), ResizeTo84x84()]
        if do_frame_stack:
            cpu_ops.append(FrameStack(4))
        cpu_preprocessor = ObsPreprocessor(
            cpu_ops,
            Space.from_gym(env.observation_space),
            Space.dtypes_from_gym(env.observation_space)
        )

        # GPU Ops
        gpu_preprocessor = ObsPreprocessor(
            [CastToFloat(), Divide255()],
            cpu_preprocessor.observation_space,
            cpu_preprocessor.observation_dtypes
        )

        action_space = Space.from_gym(env.action_space)

        super(AdeptGymEnv, self).__init__(
            action_space, cpu_preprocessor, gpu_preprocessor
        )

        self.gym_env = env
        self._gym_obs_space = env.observation_space

    @classmethod
    def from_args(cls, args, seed, **kwargs):
        # TODO fix this hack
        env = gym.make(args.env)
        if hasattr(env.unwrapped, 'ale'):
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = NoopResetEnv(env, noop_max=args.noop_max)
            # env = EpisodicLifeEnv(env)
        if 'NoFrameskip' in args.env:
            env._max_episode_steps = args.max_episode_length * \
                                     args.skip_rate
            env = MaxAndSkipEnv(env, skip=args.skip_rate)
        else:
            env._max_episode_steps = args.max_episode_length
        env.seed(seed)
        return cls(env, args.frame_stack)

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
            return self.cpu_preprocessor({'Box': observation})
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
            return self.cpu_preprocessor({'Discrete': one_hot.numpy()})
        elif isinstance(space, spaces.MultiBinary):
            return self.cpu_preprocessor({
                'MultiBinary': observation
            })
        elif isinstance(space, spaces.Dict):
            return self.cpu_preprocessor({
                name: obs
                for name, obs in observation.items()
            })
        elif isinstance(space, spaces.Tuple):
            return self.cpu_preprocessor({
                idx: obs
                for idx, obs in enumerate(observation)
            })
        else:
            raise NotImplementedError

    def _wrap_action(self, action):
        return action['Discrete']
