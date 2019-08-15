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
from collections import OrderedDict
from functools import reduce

import numpy as np
import torch
from pysc2.env import environment
from pysc2.env.sc2_env import SC2Env
from pysc2.lib import features
from pysc2.lib.actions import FUNCTIONS, FUNCTION_TYPES, FunctionCall
from pysc2.lib.features import (
    parse_agent_interface_format, SCREEN_FEATURES, MINIMAP_FEATURES
)
from pysc2.lib.static_data import UNIT_TYPES

from adept.environments.env_module import EnvModule
from adept.environments._spaces import Space
from adept.preprocess.observation import ObsPreprocessor
from adept.preprocess.ops import Operation, FlattenSpace, CastToFloat


class AdeptSC2Env(EnvModule):
    args = {}

    def __init__(self, env):
        self.sc2_env = env
        self._max_num_actions = len(FUNCTIONS)
        observation_space = {
            # 'single_select': (1, 7),
            # 'multi_select': ,
            # 'build_queue': ,
            # 'cargo': ,
            # 'cargo_slots_available': (1,),
            'screen': (24, 84, 84),
            # 'player': (11,),
            'control_groups': (10, 2),
            'available_actions': (self._max_num_actions, )
        }
        observation_dtypes = {
            'screen': torch.int16,
            'control_groups': torch.int32,
            'available_actions': torch.int32
        }
        action_space = {
            'func_id': (524, ),
            'screen_x': (84, ),
            'screen_y': (84, ),
            'minimap_x': (84, ),
            'minimap_y': (84, ),
            'screen2_x': (84, ),
            'screen2_y': (84, ),
            'queued': (2, ),
            'control_group_act': (4, ),
            'control_group_id': (10, ),
            'select_point_act': (4, ),
            'select_add': (2, ),
            'select_unit_act': (4, ),
            'select_unit_id': (500, ),
            'select_worker': (4, ),
            'unload_id': (500, ),
            'build_queue_id': (10, )
        }
        # remove_feat_op = SC2RemoveFeatures({'player_id'})
        cpu_preprocessor = ObsPreprocessor(
            [FlattenSpace({'control_groups'})],
            observation_space,
            observation_dtypes
        )

        gpu_preprocessor = SC2RemoveAvailableActions(
            [CastToFloat(), SC2ScaleChannels(24)],
            # [CastToFloat({'control_groups'}), SC2OneHot()],
            cpu_preprocessor.observation_space,
            cpu_preprocessor.observation_dtypes
        )
        self._func_id_to_headnames = SC2ActionLookup()
        super(AdeptSC2Env, self).__init__(
            action_space, cpu_preprocessor, gpu_preprocessor
        )

    @classmethod
    def from_args(
        cls,
        args,
        seed,
        sc2_replay_dir=None,
        sc2_render=False,
    ):
        agent_interface_format = parse_agent_interface_format(
            feature_screen=84, feature_minimap=84, action_space='FEATURES'
        )
        env = SC2Env(
            map_name=args.env,
            step_mul=8,
            game_steps_per_episode=0,
            discount=0.99,
            agent_interface_format=agent_interface_format,
            random_seed=seed,
            save_replay_episodes=1 if sc2_replay_dir is not None else 0,
            replay_dir=sc2_replay_dir,
            visualize=sc2_render
        )
        env = AdeptSC2Env(env)
        return env

    def step(self, action):
        timesteps = self.sc2_env.step(self._wrap_action(action))
        # pysc2 returns a tuple of timesteps, with one timestep inside
        # get first timestep
        pysc2_step = timesteps[0]
        reward = float(pysc2_step.reward)
        done = info = pysc2_step.step_type == environment.StepType.LAST
        return self.cpu_preprocessor(
            self._wrap_observation(pysc2_step.observation)
        ), reward, done, info

    def reset(self):
        timesteps = self.sc2_env.reset()
        assert len(timesteps) == 1
        # pysc2 returns a tuple of timesteps, with one timestep inside
        # get first timestep
        pysc2_step = timesteps[0]
        return self.cpu_preprocessor(
            self._wrap_observation(pysc2_step.observation)
        )

    def close(self):
        self.sc2_env.close()

    def _wrap_observation(self, observation):
        obs = OrderedDict()
        obs['screen'] = torch.cat(
            [
                torch.from_numpy(observation['feature_screen']),
                torch.from_numpy(observation['feature_minimap'])
            ]
        )
        obs['control_groups'] = torch.from_numpy(observation['control_groups'])
        avail_actions_one_hot = np.zeros(self._max_num_actions, dtype=np.int64)
        avail_actions_one_hot[observation['available_actions']] = 1
        obs['available_actions'] = torch.from_numpy(avail_actions_one_hot)
        return obs

    def _wrap_action(self, action):
        func_id = action['func_id'].item()
        required_heads = self._func_id_to_headnames[func_id]
        args = []

        for headname in required_heads.keys():
            if '_y' in headname:
                continue
            elif '_x' in headname:
                args.append([
                    action[headname].item(),
                    action[headname[:-2] + '_y'].item()])
            else:
                args.append([action[headname].item()])

        return [FunctionCall(func_id, args)]


class SC2RemoveFeatures(Operation):
    def __init__(
        self, feats_to_remove, feats=SCREEN_FEATURES + MINIMAP_FEATURES
    ):
        super(SC2RemoveFeatures, self).__init__({'screen'})

        self.idxs = []
        self.features = []

        for i, feat in enumerate(feats):
            if feat.name not in feats_to_remove:
                self.idxs.append(i)
                self.features.append(feat)

    def update_shape(self, old_shape):
        return (len(self.idxs), ) + old_shape[1:]

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        if obs.dim() == 3:
            return obs[self.idxs]
        elif obs.dim() == 4:
            return obs.index_select(
                1, torch.LongTensor(self.idxs, device=obs.device)
            )
        else:
            raise ValueError(
                'Cannot remove SC2 features from a {}-dimensional tensor'.
                format(obs.dim())
            )


class SC2OneHot(Operation):
    def __init__(self, feats=SCREEN_FEATURES + MINIMAP_FEATURES):
        super(SC2OneHot, self).__init__({'screen'})

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

        scales = []
        for i, feat in enumerate(feats):
            if feat.type == features.FeatureType.SCALAR:
                scales.append(feat.scale)
        self._scales = 1. / torch.tensor(scales).float()

    def update_dtype(self, old_dtype):
        return torch.float32

    def update_shape(self, old_shape):
        new_shape = (
            len(self._scalar_idxs) + len(
                reduce(
                    lambda prev, cur: prev + cur,
                    self._ranges_by_feature_idx.values()
                )
            ),
        ) + old_shape[1:]
        return new_shape

    def update_obs(self, obs):
        if self._scales.device != obs.device:
            self._scales = self._scales.to(obs.device)
        # TODO: warning, this is really slow
        if obs.dim() == 3:
            one_hot_channels = []
            for i, rngs in self._ranges_by_feature_idx.items():
                for rng in rngs:
                    one_hot_channels.append(obs[i] == rng)
            obs = obs[self._scalar_idxs]
            one_hot_channels = torch.stack(one_hot_channels)
            result = torch.cat(
                [
                    obs,
                    one_hot_channels.to(
                        one_hot_channels.device, dtype=torch.int32
                    )
                ]
            )
            return result
        elif obs.dim() == 4:
            one_hot_channels = []
            for i, rngs in self._ranges_by_feature_idx.items():
                for rng in rngs:
                    one_hot_channels.append(obs[:, i, :, :] == rng)
            one_hot_channels = torch.stack(one_hot_channels, dim=1)
            return torch.cat(
                [
                    obs[:, self._scalar_idxs, :, :].float() *
                    self._scales.view(1, -1, 1, 1),
                    one_hot_channels.float()
                ],
                dim=1
            )
        elif obs.dim() == 5:  # seq, batch, channel, x, y
            one_hot_channels = []
            for i, rngs in self._ranges_by_feature_idx.items():
                for rng in rngs:
                    one_hot_channels.append(obs[:, :, i, :, :] == rng)
            one_hot_channels = torch.stack(one_hot_channels, dim=2)
            return torch.cat(
                [
                    obs[:, :, self._scalar_idxs, :, :].float() *
                    self._scales.view(1, 1, -1, 1, 1),
                    one_hot_channels.float()
                ],
                dim=2
            )
        else:
            raise ValueError(
                'Cannot convert {}-dimensional tensor to one-hot'.format(
                    obs.dim()
                )
            )


class SC2ScaleChannels(Operation):
    def __init__(
        self, nb_channel, feats=SCREEN_FEATURES + MINIMAP_FEATURES, mode='all'
    ):
        """
        :param nb_channel:
        :param feats:
        :param mode: 'all' or 'scalar' to decide which type of features to scale
        """
        super(SC2ScaleChannels, self).__init__({'screen'})
        scales = torch.ones(nb_channel)
        for i, feat in enumerate(feats):
            if mode == 'all':
                scales[i] = feat.scale
            elif mode == 'scalar':
                if feat.type == features.FeatureType.SCALAR:
                    scales[i] = feat.scale
        self.scales = 1. / scales.float()

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return torch.float32

    def update_obs(self, obs):
        obs = obs.float()

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
    def __init__(self, ops, observation_space, observation_dtypes=None):
        super().__init__(ops, observation_space, observation_dtypes)
        filtered_space = {
            k: v
            for k, v in observation_space.items()
            if k != 'available_actions'
        }
        self.observation_space = filtered_space

    def __call__(self, obs):
        result = super().__call__(obs)
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
