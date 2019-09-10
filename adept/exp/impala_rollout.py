from adept.exp import ExpModule
from collections import namedtuple


class ImpalaRollout(dict, ExpModule):
    args = {}

    def __init__(self, reward_normalizer):
        super(ImpalaRollout, self).__init__()
        self['states'] = []
        self['rewards'] = []
        self['terminals'] = []
        self['actions'] = []
        self['log_probs'] = []
        self['next_obs'] = []
        self['internals'] = []
        self.reward_normalizer = reward_normalizer

    @classmethod
    def from_args(cls, args, reward_normalizer):
        return cls(reward_normalizer)

    def write_actor(self, experience):
        for k, v in experience.items():
            if k not in self:
                raise KeyError(f'Incompatible rollout key: {k}')
            self[k].append(v)

    def write_env(self, obs, rewards, terminals, infos):
        rewards = self.reward_normalizer(rewards)
        self['states'].append(obs)
        self['rewards'].append(rewards)
        self['terminals'].append(terminals)

    def write_next_obs(self, obs):
        # must be a list, but only has 1 element
        self['next_obs'].append(obs)

    def write_internals(self, internals):
        # must be a list, but only has 1 element
        self['internals'].append(internals)

    def read(self):
        # returns rollout as a named tuple
        return namedtuple(self.__class__.__name__, self.keys())(**self)

    def clear(self):
        for k in self.keys():
            self[k] = []

    def is_ready(self):
        return True  # TODO

    def __len__(self):
        return len(self['rewards'])

