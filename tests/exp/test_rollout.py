import unittest

import torch
from adept.exp import Rollout, ExpSpecBuilder

obs_space = {
    'obs_a': (2, 2),
    'obs_b': (3, 3)
}
act_space = {
    'act_a': (5, ),
    'act_b': (6, )
}
internal_space = {
    'internal_a': (2, ),
    'internal_b': (3, )
}
obs_keys = ['obs_a', 'obs_b']
act_keys = ['act_a', 'act_b']
internal_keys = ['internal_a', 'internal_b']
batch_size = 16
exp_len = 20


def build_fn(exp_len):
    return {
        'obs_a': (exp_len + 1, batch_size, 2, 2),
        'obs_b': (exp_len + 1, batch_size, 3, 3),
        'act_a': (exp_len, batch_size),
        'act_b': (exp_len, batch_size),
        'internal_a': (exp_len, batch_size, 2),
        'internal_b': (exp_len, batch_size, 3),
        'rewards': (exp_len, batch_size),
        'terminals': (exp_len, batch_size)
    }


spec_builder = ExpSpecBuilder(
    obs_keys=obs_space,
    act_keys=act_space,
    internal_keys=internal_space,
    key_types={
        'obs_a': 'long',
        'obs_b': 'long',
        'act_a': 'long',
        'act_b': 'long',
        'internal_a': 'float',
        'internal_b': 'float',
        'rewards': 'float',
        'terminals': 'float'
    },
    exp_keys=obs_keys + act_keys + internal_keys + ['rewards', 'terminals'],
    build_fn=build_fn
)


class TestRollout(unittest.TestCase):
    def test_next_obs(self):
        r = Rollout(spec_builder, 20)
        next_obs = {
            'obs_a': torch.ones(batch_size, 2, 2),
            'obs_b': torch.ones(batch_size, 3, 3)
        }
        r.write_next_obs(next_obs)
        next_obs = r.read().next_observation
        print(next_obs['obs_a'].shape)
        print(next_obs['obs_b'].shape)
        # print(next_obs)
        self.assertEqual(next_obs['obs_a'][0][0][0].item(), 1)
        self.assertEqual(next_obs['obs_b'][0][0][0].item(), 1)
