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
from collections import OrderedDict
from functools import reduce
from itertools import chain

import gym
import numpy as np
import torch
from pysc2.env import environment
from pysc2.env.sc2_env import SC2Env
from pysc2.lib import features
from pysc2.lib.actions import FUNCTIONS, FUNCTION_TYPES
from pysc2.lib.actions import FunctionCall
from pysc2.lib.features import parse_agent_interface_format, SCREEN_FEATURES, MINIMAP_FEATURES
from pysc2.lib.static_data import UNIT_TYPES

from adept.environments._base import BaseEnvironment, Spaces, Space
from adept.preprocess.observation import ObsPreprocessor
from adept.preprocess.ops import BaseOp, CastToFloat, FlattenSpace


def make_sc2_env(env_id, seed, replay_dir=None, render=False):
    def _f():
        env = sc2_feature_env(env_id, seed, replay_dir, render)
        return env
    return _f


def sc2_feature_env(env_id, seed, replay_dir, render):
    agent_interface_format = parse_agent_interface_format(
        feature_screen=84,
        feature_minimap=84,
        action_space='FEATURES'
    )
    env = SC2Env(
        map_name=env_id,
        step_mul=8,
        game_steps_per_episode=0,
        discount=0.99,
        agent_interface_format=agent_interface_format,
        random_seed=seed,
        save_replay_episodes=1 if replay_dir is not None else 0,
        replay_dir=replay_dir,
        visualize=render
    )
    env = AdeptSC2Env(env)
    return env


class AdeptSC2Env(BaseEnvironment):
    def __init__(self, env):
        self.sc2_env = env
        obs_entries_by_name = {
            # 'single_select': Space((1, 7), 0., 1., np.float32),
            # 'multi_select': Space(),
            # 'build_queue': Space(),
            # 'cargo': Space(),
            # 'cargo_slots_available': Space((1,), None, None, None),
            'vision': Space((24, 84, 84), None, None, None),
            # 'player': Space((11,), None, None, None),
            'control_groups': Space((10, 2), None, None, None),
            'available_actions': Space((None,), None, None, None)
        }
        act_entries_by_name = {
            'func_id': Space((524,), 0., 1., np.float32),
            'screen_x': Space((80,), 0., 1., np.float32),
            'screen_y': Space((80,), 0., 1., np.float32),
            'minimap_x': Space((80,), 0., 1., np.float32),
            'minimap_y': Space((80,), 0., 1., np.float32),
            'screen2_x': Space((80,), 0., 1., np.float32),
            'screen2_y': Space((80,), 0., 1., np.float32),
            'queued': Space((2,), 0., 1., np.float32),
            'control_group_act': Space((4,), 0., 1., np.float32),
            'control_group_id': Space((10,), 0., 1., np.float32),
            'select_point_act': Space((4,), 0., 1., np.float32),
            'select_add': Space((2,), 0., 1., np.float32),
            'select_unit_act': Space((4,), 0., 1., np.float32),
            'select_unit_id': Space((500,), 0., 1., np.float32),
            'select_worker': Space((4,), 0., 1., np.float32),
            'unload_id': Space((500,), 0., 1., np.float32),
            'build_queue_id': Space((10,), 0., 1., np.float32),
        }
        # remove_feat_op = SC2RemoveFeatures({'player_id'})
        self._cpu_preprocessor = ObsPreprocessor(
            [FlattenSpace({'control_groups'})],
            Spaces(obs_entries_by_name)
        )

        self._gpu_preprocessor = SC2RemoveAvailableActions(
            [CastToFloat(), SC2ScaleChannels(24)],
            self._cpu_preprocessor.observation_space
        )
        self._observation_space = self._gpu_preprocessor.observation_space
        self._action_space = Spaces(act_entries_by_name)
        self._func_id_to_headnames = SC2ActionLookup()

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
        """
        :param action_by_name: Dict{.}[headname: action/arg]
        :return:
        """
        timesteps = self.sc2_env.step(self._wrap_action(action))
        assert len(timesteps) == 1
        # pysc2 returns a tuple of timesteps, with one timestep inside
        # get first timestep
        pysc2_step = timesteps[0]
        reward = float(pysc2_step.reward)
        done = info = pysc2_step.step_type == environment.StepType.LAST
        return self.cpu_preprocessor(self._wrap_observation(pysc2_step.observation)), reward, done, info

    def reset(self):
        timesteps = self.sc2_env.reset()
        assert len(timesteps) == 1
        # pysc2 returns a tuple of timesteps, with one timestep inside
        # get first timestep
        pysc2_step = timesteps[0]
        return self.cpu_preprocessor(self._wrap_observation(pysc2_step.observation))

    def close(self):
        self.sc2_env.close()

    def _wrap_observation(self, observation):
        obs = OrderedDict()
        obs['vision'] = torch.cat([
            torch.from_numpy(observation['feature_screen']),
            torch.from_numpy(observation['feature_minimap'])
        ])
        obs['control_groups'] = torch.from_numpy(observation['control_groups'])
        obs['available_actions'] = frozenset(observation['available_actions'])
        return obs

    def _wrap_action(self, action):
        func_id = action['func_id']
        required_heads = self._func_id_to_headnames[func_id]
        args = []

        for headname in required_heads.keys():
            if '_y' in headname:
                continue
            elif '_x' in headname:
                args.append([action[headname], action[headname[:-2] + '_y']])
            else:
                args.append([action[headname]])

        return [FunctionCall(func_id, args)]


class SC2RemoveFeatures(BaseOp):
    def __init__(self, feats_to_remove, feats=SCREEN_FEATURES + MINIMAP_FEATURES):
        super(SC2RemoveFeatures, self).__init__({'vision'})

        self.idxs = []
        self.features = []

        for i, feat in enumerate(feats):
            if feat.name not in feats_to_remove:
                self.idxs.append(i)
                self.features.append(feat)

    def update_space(self, old_space):
        new_shape = (len(self.idxs),) + old_space.shape[1:]
        return Space(new_shape, old_space.low, old_space.high, old_space.dtype)

    def update_obs(self, obs):
        if obs.dim() == 3:
            return obs[self.idxs]
        elif obs.dim() == 4:
            return obs.index_select(1, torch.LongTensor(self.idxs, device=obs.device))
        else:
            raise ValueError('Cannot remove SC2 features from a {}-dimensional tensor'.format(obs.dim()))


class SC2OneHot(BaseOp):
    def __init__(self, feats=SCREEN_FEATURES + MINIMAP_FEATURES):
        super(SC2OneHot, self).__init__({'vision'})

        self.features = []
        self._ranges_by_feature_idx = {}
        self._scalar_idxs = []

        for i, feat in enumerate(feats):
            if feat.type == features.FeatureType.SCALAR:
                self._scalar_idxs.append(i)
                self.features.append(feat)
            if feat.type == features.FeatureType.CATEGORICAL:
                if feat.name == 'unit_type':
                    self._ranges_by_feature_idx[i] = UNIT_TYPES
                elif feat.name == 'visibility':
                    self._ranges_by_feature_idx[i] = list(range(3))
                elif feat.name == 'effects':
                    self._ranges_by_feature_idx[i] = list(range(1, 13))
                elif feat.scale == 2:
                    self._ranges_by_feature_idx[i] = [1]
                else:
                    self._ranges_by_feature_idx[i] = list(range(feat.scale))

    def update_space(self, old_space):
        new_shape = (len(self._scalar_idxs) + len(reduce(lambda prev, cur: prev + cur, self._ranges_by_feature_idx.values())),) + old_space.shape[1:]
        return Space(new_shape, old_space.low, old_space.high, old_space.dtype)

    def update_obs(self, obs):
        # warning, this is really slow
        if obs.dim() == 3:
            one_hot_channels = []
            for i, rngs in self._ranges_by_feature_idx.items():
                for rng in rngs:
                    one_hot_channels.append(obs[i] == rng)
            obs = obs[self._scalar_idxs]
            one_hot_channels = torch.stack(one_hot_channels)
            result = torch.cat([obs, one_hot_channels.to(one_hot_channels.device, dtype=torch.int32)])
            return result
        elif obs.dim() == 4:
            # TODO
            raise NotImplementedError
        else:
            raise ValueError('Cannot convert {}-dimensional tensor to one-hot'.format(obs.dim()))


class SC2ScaleChannels(BaseOp):
    def __init__(self, nb_channel, feats=SCREEN_FEATURES + MINIMAP_FEATURES, mode='all'):
        """

        :param nb_channel:
        :param feats:
        :param mode: 'all' or 'scalar' to decide which type of features to scale
        """
        super(SC2ScaleChannels, self).__init__({'vision'})
        scales = torch.ones(nb_channel)
        for i, feat in enumerate(feats):
            if mode == 'all':
                scales[i] = feat.scale
            elif mode == 'scalar':
                if feat.type == features.FeatureType.SCALAR:
                    scales[i] = feat.scale
        self.scales = 1. / torch.tensor(scales).float()

    def update_space(self, old_space):
        return old_space

    def update_obs(self, obs):
        if self.scales.device != obs.device:
            self.scales = self.scales.to(obs.device)

        if obs.dim() == 3:
            obs *= self.scales.view(-1, 1, 1)
            return obs
        elif obs.dim() == 4:
            obs *= self.scales.view(1, -1, 1, 1)
            return obs
        else:
            raise ValueError('Unsupported dimensionality ' + str(obs.dim()))


class SC2RemoveAvailableActions(ObsPreprocessor):
    def __init__(self, ops, observation_space):
        super().__init__(ops, observation_space)
        ebn = self.observation_space.entries_by_name
        ebn = {k: v for k, v in ebn.items() if k != 'available_actions'}
        self.observation_space = Spaces(ebn)

    def __call__(self, obs, device=None):
        result = super().__call__(obs, device)
        return {k: v for k, v in result.items() if k != 'available_actions'}


class SC2ActionLookup(dict):
    def __init__(self):
        super().__init__()
        for func in FUNCTIONS:
            func_id = func.id
            arg_names = [arg.name for arg in FUNCTION_TYPES[func.function_type]]
            self[func_id] = self._arg_names_to_head_names(arg_names)

    def _arg_names_to_head_names(self, arg_names):
        headnames = []
        for argname in arg_names:
            if argname == 'screen':
                headnames.extend(['screen_x', 'screen_y'])
            elif argname == 'minimap':
                headnames.extend(['minimap_x', 'minimap_y'])
            elif argname == 'screen2':
                headnames.extend(['screen2_x', 'screen2_y'])
            else:
                headnames.append(argname)
        # OrderedDict for constant time membership test while preserving order
        # TODO make an OrderedSet in utils
        return OrderedDict.fromkeys(headnames)


if __name__ == '__main__':
    from absl import flags
    # import sys
    # FLAGS = flags.FLAGS
    # FLAGS(sys.argv)
    FLAGS = flags.FLAGS
    FLAGS(['sc2.py'])

    env = sc2_feature_env('MoveToBeacon', 1, None)
    env.reset()
    print()
