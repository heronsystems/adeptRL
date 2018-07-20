from adept.expcaches._base import BaseExperience
from collections import namedtuple
import torch
import numpy as np


class RolloutCache(dict, BaseExperience):
    def __init__(self, nb_rollout, device, reward_normalizer, keys):
        super(RolloutCache, self).__init__()
        for k in keys:
            self[k] = []
        self['states'] = []
        self['rewards'] = []
        # TODO: rename as terminals_mask
        self['terminals'] = []
        self.nb_rollout = nb_rollout
        self.device = device
        self.reward_normalizer = reward_normalizer

    def write_forward(self, **kwargs):
        for k, v in kwargs.items():
            self[k].append(v)

    def write_env(self, obs, rewards, terminals, infos):
        rewards = torch.tensor([self.reward_normalizer(reward) for reward in rewards]).to(self.device)
        terminals = (1. - torch.from_numpy(np.array(terminals, dtype=np.float32))).to(self.device)
        self['states'].append(obs)
        self['rewards'].append(rewards)
        # TODO: rename as terminals_mask or don't mask here
        self['terminals'].append(terminals)

    def read(self):
        # returns rollout as a named tuple
        return namedtuple(self.__class__.__name__, self.keys())(**self)

    def clear(self):
        for k in self.keys():
            self[k] = []

    def is_ready(self):
        return len(self) == (self.nb_rollout)

    def __len__(self):
        return len(self['rewards'])
