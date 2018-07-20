from collections import OrderedDict

import gym
import numpy as np
import torch
from gym import spaces
from pysc2.env import environment
from pysc2.env.sc2_env import SC2Env
from pysc2.lib import features
from pysc2.lib.actions import FUNCTIONS, FUNCTION_TYPES
from pysc2.lib.actions import FunctionCall
from pysc2.lib.features import parse_agent_interface_format, SCREEN_FEATURES, MINIMAP_FEATURES
from pysc2.lib.static_data import UNIT_TYPES

from adept.agents import EnvBase


def make_sc2_env(env_id, seed, replay_dir=None):
    def _f():
        env = sc2_env(env_id, seed, replay_dir)
        return env
    return _f


agent_interface_format = parse_agent_interface_format(
    feature_screen=80,
    feature_minimap=80,
    action_space='FEATURES'
)


def sc2_env(env_id, seed, replay_dir):
    env = SC2GymEnv(
        map_name=env_id,
        step_mul=8,
        game_steps_per_episode=0,
        discount=0.99,
        agent_interface_format=agent_interface_format,
        random_seed=seed,
        save_replay_episodes=1 if replay_dir is not None else 0,
        replay_dir=replay_dir
    )
    env = FilterObsKeys(env, ['feature_screen', 'feature_minimap', 'control_groups', 'available_actions'])
    env = FilterAndScaleScreenChannels(env)
    return env


class SC2GymEnv(SC2Env, gym.Env):
    action_space = spaces.Dict({
        'func_id': spaces.Discrete(524),
        'screen_x': spaces.Discrete(80),
        'screen_y': spaces.Discrete(80),
        'minimap_x': spaces.Discrete(80),
        'minimap_y': spaces.Discrete(80),
        'screen2_x': spaces.Discrete(80),
        'screen2_y': spaces.Discrete(80),
        'queued': spaces.Discrete(2),
        'control_group_act': spaces.Discrete(4),
        'control_group_id': spaces.Discrete(10),
        'select_point_act': spaces.Discrete(4),
        'select_add': spaces.Discrete(2),
        'select_unit_act': spaces.Discrete(4),
        'select_unit_id': spaces.Discrete(500),
        'select_worker': spaces.Discrete(4),
        'unload_id': spaces.Discrete(500),
        'build_queue_id': spaces.Discrete(10)
    })
    observation_space = spaces.Dict({
        'float_state': spaces.Box(low=0, high=1, shape=(10, 80, 80), dtype=np.float32),
        'binary_state': spaces.Box(low=0, high=1, shape=(256, 80, 80), dtype=np.uint8)
    })

    def __init__(self, **kwargs):
        super(SC2GymEnv, self).__init__(**kwargs)

    def step(self, action):
        """
        :param action_by_name: Dict{.}[headname: action/arg]
        :return:
        """
        timesteps = super(SC2GymEnv, self).step([action])
        assert len(timesteps) == 1
        # pysc2 returns a tuple of timesteps, with one timestep inside
        # get first timestep
        pysc2_step = timesteps[0]
        reward = float(pysc2_step.reward)
        done = info = pysc2_step.step_type == environment.StepType.LAST
        return pysc2_step.observation, reward, done, info

    def reset(self):
        timesteps = super(SC2GymEnv, self).reset()
        assert len(timesteps) == 1
        # pysc2 returns a tuple of timesteps, with one timestep inside
        # get first timestep
        pysc2_step = timesteps[0]
        return pysc2_step.observation

    def render(self, mode='human'):
        raise NotImplementedError


class FilterObsKeys(gym.ObservationWrapper):
    def __init__(self, env, filter_fields):
        gym.Wrapper.__init__(self, env)
        self.filter_fields = filter_fields

    def observation(self, observation):
        new_obs = {}
        for filter in self.filter_fields:
            new_obs[filter] = observation[filter]
        return new_obs


class FilterAndScaleScreenChannels(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        feature_screen = observation['feature_screen']
        feature_minimap = observation['feature_minimap']
        control_groups = observation['control_groups']
        available_actions = observation['available_actions']
        scalar_state, binary_state = self._process_features(feature_screen, feature_minimap)
        scalar_state = scalar_state.astype(np.float32)
        binary_state = binary_state.astype(np.uint8)
        obs = OrderedDict(
            float_state=torch.from_numpy(scalar_state),
            binary_state=torch.from_numpy(binary_state),
            available_actions=frozenset(available_actions),
            control_groups=control_groups
        )
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


def lookup_headnames_by_id(func_id):
    argnames = lookup_argnames_by_id(func_id)
    headnames = []
    for argname in argnames:
        if argname == 'screen':
            headnames.extend(['screen_x', 'screen_y'])
        elif argname == 'minimap':
            headnames.extend(['minimap_x', 'minimap_y'])
        elif argname == 'screen2':
            headnames.extend(['screen2_x', 'screen2_y'])
        else:
            headnames.append(argname)
    # OrderedDict for constant time membership test while preserving order
    return OrderedDict.fromkeys(headnames)


def lookup_argnames_by_id(func_id):
    """
    :param func_id: int
    :return: argument_names: List{.}[arg_name]
    """
    func = FUNCTIONS[func_id]
    required_args = FUNCTION_TYPES[func.function_type]
    argument_names = []
    for arg in required_args:
        argument_names.append(arg.name)
    return argument_names


class SC2AgentOverrides(EnvBase):
    def preprocess_logits(self, logits):
        return logits

    def process_logits(self, logits, obs, deterministic):
        """
        The SC2 environment requires special logic to mask out unused network outputs.
        :param logits:
        :return:
        """
        available_actions = obs['available_actions']
        actions, log_probs, entropies, head_masks = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        headnames = logits.keys()

        for headname in headnames:
            actions[headname], log_probs[headname], entropies[headname] = super().process_logits(logits[headname], obs, deterministic)
            if headname == 'func_id':
                head_masks[headname] = torch.ones_like(entropies[headname])
            else:
                head_masks[headname] = torch.zeros_like(entropies[headname])

        function_calls = []
        # iterate over batch dimension
        for i in range(actions['func_id'].shape[0]):
            # force a no op if action is unavailable
            if actions['func_id'][i] not in available_actions[i]:
                function_calls.append(FunctionCall(0, []))
                continue

            # build the masks and the FunctionCall's in the same loop
            args = []
            func_id = actions['func_id'][i]
            required_heads = lookup_headnames_by_id(func_id)
            for headname in required_heads.keys():
                # toggle mask to 1 if the head is required
                head_masks[headname][i] = 1.

                # skip y's
                if '_y' in headname:
                    continue
                # if x, build the argument
                elif '_x' in headname:
                    args.append([actions[headname][i], actions[headname[:-2] + '_y'][i]])
                else:
                    args.append([actions[headname][i]])
            function_calls.append(FunctionCall(func_id, args))

        # apply masks to log_probs and entropies
        for headname in headnames:
            log_probs[headname] = log_probs[headname] * head_masks[headname]
            entropies[headname] = entropies[headname] * head_masks[headname]

        log_probs = torch.stack(tuple(v for v in log_probs.values()), dim=1)
        entropies = torch.stack(tuple(v for v in entropies.values()), dim=1)
        return function_calls, log_probs, entropies


if __name__ == '__main__':
    from absl import flags
    # import sys
    # FLAGS = flags.FLAGS
    # FLAGS(sys.argv)
    FLAGS = flags.FLAGS
    FLAGS(['sc2.py'])

    env = sc2_env('MoveToBeacon', 1)
    env.reset()
    print()
