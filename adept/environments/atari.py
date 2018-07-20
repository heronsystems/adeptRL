import cv2
import gym
import numpy as np
import torch
from cv2.cv2 import resize
from gym import spaces

from ._wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, NormalizedEnv, FrameStack

cv2.ocl.setUseOpenCL(False)


def make_atari_env(env_id, env_conf, skip_rate, max_ep_length, seed, frame_stack=False):
    def _f():
        env = atari_env(env_id, env_conf, skip_rate, max_ep_length, frame_stack)
        env.seed(seed)
        return env
    return _f


def atari_env(env_id, env_conf, skip_rate, max_ep_length, frame_stack=False):
    env = gym.make(env_id)
    if 'NoFrameskip' in env_id:
        assert 'NoFrameskip' in env.spec.id
        env._max_episode_steps = max_ep_length * skip_rate
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=skip_rate)
    else:
        env._max_episode_steps = max_ep_length
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env._max_episode_steps = max_ep_length
    env = AtariRescale(env, env_conf)
    env = NormalizedEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    env = DictTensorObs(env)
    return env


def process_frame(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = resize(frame, (80, conf["dimension2"]))
    frame = resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(0.0, 1.0, [1, 80, 80], dtype=np.float32)
        self.conf = env_conf

    def observation(self, observation):
        return process_frame(observation, self.conf)


class DictTensorObs(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        return {'obs': torch.from_numpy(observation)}
