# from multiprocessing import Pipe, Process
from torch import multiprocessing as mp

from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from baselines.common.vec_env.subproc_vec_env import worker
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.features import parse_agent_interface_format, SCREEN_FEATURES, MINIMAP_FEATURES
from pysc2.lib import features
from pysc2.lib.actions import FUNCTIONS, FUNCTION_TYPES, FunctionCall
from pysc2.env import environment
from pysc2.lib.static_data import UNIT_TYPES
import numpy as np
import torch

import gym


def make_sc2_env(env_id, seed):
    def _f():
        env = sc2_env(env_id, seed)
        return env
    return _f


agent_interface_format = parse_agent_interface_format(
    feature_screen=80,
    feature_minimap=80,
    action_space='FEATURES'
)


def sc2_env(env_id, seed):
    env = SC2EnvWrapper(
        map_name=env_id,
        step_mul=8,
        game_steps_per_episode=0,
        discount=0.99,
        agent_interface_format=agent_interface_format,
        random_seed=seed
    )
    return env


class SC2EnvWrapper(SC2Env):
    def __init__(self, **kwargs):
        super(SC2EnvWrapper, self).__init__(**kwargs)

    def step(self, action_by_name):
        """
        :param action_by_name: Dict{.}[headname: action/arg]
        :return:
        """
        func_id = action_by_name['func_id']
        args = [v for k, v in action_by_name.items() if k != 'func_id']
        timesteps = super(SC2EnvWrapper, self).step([FunctionCall(func_id, args)])
        assert len(timesteps) == 1
        # pysc2 returns a tuple of timesteps, with one timestep inside
        # get first timestep
        pysc2_step = timesteps[0]
        pysc2_obs = pysc2_step.observation
        feature_screen = pysc2_obs['feature_screen']
        feature_minimap = pysc2_obs['feature_minimap']
        control_groups = pysc2_obs['control_groups']
        available_actions = pysc2_obs['available_actions']
        scalar_state, binary_state = self._process_features(feature_screen, feature_minimap)
        scalar_state = scalar_state.astype(np.float32)
        binary_state = binary_state.astype(np.uint8)
        obs = {
            'float_state': torch.from_numpy(scalar_state),
            'binary_state': torch.from_numpy(binary_state),
            'available_actions': available_actions,
            'control_groups': control_groups
        }
        reward = pysc2_step.reward
        done = info = pysc2_step.step_type == environment.StepType.LAST
        return obs, reward, done, info

    def reset(self):
        timesteps = super(SC2EnvWrapper, self).reset()
        assert len(timesteps) == 1
        # pysc2 returns a tuple of timesteps, with one timestep inside
        # get first timestep
        pysc2_step = timesteps[0]
        pysc2_obs = pysc2_step.observation
        feature_screen = pysc2_obs['feature_screen']
        feature_minimap = pysc2_obs['feature_minimap']
        control_groups = pysc2_obs['control_groups']
        available_actions = pysc2_obs['available_actions']
        scalar_state, binary_state = self._process_features(feature_screen, feature_minimap)
        scalar_state = scalar_state.astype(np.float32)
        binary_state = binary_state.astype(np.uint8)
        obs = {
            'float_state': torch.from_numpy(scalar_state),
            'binary_state': torch.from_numpy(binary_state),
            'available_actions': available_actions,
            'control_groups': control_groups
        }
        return obs

    def _process_features(self, screen, minimap):
        screen_scalar, screen_binary = self._process_screen(screen)
        minimap_scalar, minimap_binary = self._process_minimap(minimap)
        return np.concatenate([screen_scalar, minimap_scalar]), np.concatenate([screen_binary, minimap_binary])

    def _process_screen(self, feature_screen):
        """
        Split categorical features into their own channels.
        :param feature_screen:
        :return:
        """
        scalar_features = []
        binary_features = []
        for feat, channel in zip(SCREEN_FEATURES, feature_screen):
            if feat.type == features.FeatureType.SCALAR:
                scalar_features.append(self._scale(channel, feat.scale))
            if feat.type == features.FeatureType.CATEGORICAL:
                if feat.name == 'player_id':
                    # not a useful feature, only care about 1v1
                    continue
                elif feat.name == 'unit_type':
                    for unit_type in UNIT_TYPES:
                        binary_features.append(channel == unit_type)
                elif feat.name == 'visibility':
                    length = 3  # hack to get around bug
                    for i in range(length):
                        binary_features.append(channel == i)
                elif feat.name == 'effects':
                    length = 12  # hack to get around bug
                    for i in range(length):
                        binary_features.append(channel == (i + 1))
                elif feat.scale == 2:
                    binary_features.append(channel == 1)
                else:
                    for i in range(feat.scale):
                        binary_features.append(channel == i)
        return np.array(scalar_features), np.array(binary_features)

    def _process_minimap(self, minimap_screen):
        """
        Split categorical features into their own channels.
        :param minimap:
        :return:
        """
        scalar_features = []
        binary_features = []
        for feat, channel in zip(MINIMAP_FEATURES, minimap_screen):
            if feat.type == features.FeatureType.SCALAR:
                scalar_features.append(self._scale(channel, feat.scale))
            if feat.type == features.FeatureType.CATEGORICAL:
                if feat.name == 'player_id':
                    continue
                elif feat.name == 'visibility':
                    # hack to get around bug
                    for i in range(3):
                        binary_features.append(channel == i)
                else:
                    for i in range(feat.scale):
                        binary_features.append(channel == i)
        return np.array(scalar_features), np.array(binary_features)

    def _scale(self, nd_array, max_val):
        return nd_array / max_val

    def render(self, mode='human'):
        raise NotImplementedError


class SC2SubprocEnv:
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        mp.set_start_method('spawn')
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(nenvs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # self.remotes[0].send(('get_spaces', None))
        # observation_space, action_space = self.remotes[0].recv()
        # VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return obs, [float(rew) for rew in rews], dones, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return obs

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class SC2SingleEnv:
    def __init__(self, env_fns):
        self.env = env_fns[0]()

    def step(self, actions):
        results = [self.env.step(actions[0])]
        obs, rews, dones, infos = zip(*results)
        return obs, [float(rew) for rew in rews], dones, infos

    def reset(self):
        return [self.env.reset()]

    def close(self):
        self.env.close()


if __name__ == '__main__':
    # from absl import flags
    # import sys
    # FLAGS = flags.FLAGS
    # FLAGS(sys.argv)

    env = sc2_env('MoveToBeacon', 1)
    env.reset()
    print()
