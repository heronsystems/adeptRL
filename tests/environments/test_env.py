import io
import sys
import unittest

from adept.environments._env import EnvBase


class StubEnv(EnvBase):
    def __init__(self, defaults):
        self._defaults = defaults

    @property
    def defaults(self):
        return self._defaults

    def step(self, action):
        pass

    def reset(self, **kwargs):
        pass

    def close(self):
        pass

    @property
    def observation_space(self):
        return None

    @property
    def action_space(self):
        return None

    @property
    def cpu_preprocessor(self):
        return None

    @property
    def gpu_preprocessor(self):
        return None


class EnvBase(unittest.TestCase):
    defaults = {
        'k1': 0,
        'k2': False,
        'k3': 1.5,
        'k4': 'hello'
    }
    stub = StubEnv(defaults)

    def test_prompt_no_changes(self):
        sys.stdin = io.StringIO('\n')
        new_conf = self.stub.prompt()
        assert(new_conf == self.defaults)

    def test_prompt_modify(self):
        sys.stdin = io.StringIO('{"k1": 5}')
        new_conf = self.stub.prompt()
        assert(new_conf['k1'] == 5)
        assert(new_conf['k2'] == self.defaults['k2'])
        assert(new_conf['k3'] == self.defaults['k3'])
        assert(new_conf['k4'] == self.defaults['k4'])


if __name__ == '__main__':
    unittest.main(verbosity=1)
