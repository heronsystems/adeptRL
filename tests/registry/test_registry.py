import unittest
import torch

from adept.agent import AgentModule
from adept.env import EnvModule
from adept.preprocess.base.preprocessor import ObsPreprocessor
from adept.preprocess import CPUPreprocessor, GPUPreprocessor
from adept.registry import Registry


class NotSubclass:
    pass


class ArgsNotImplemented(AgentModule):
    pass


class DummyEnv(EnvModule):

    args = {}
    ids = ["dummy"]

    def __init__(self):
        obs_space = {"screen": (3, 84, 84)}
        action_space = {"action": (8,)}
        cpu_preprocessor = CPUPreprocessor([], obs_space)
        gpu_preprocessor = GPUPreprocessor(
            [], cpu_preprocessor.observation_space
        )
        super(DummyEnv, self).__init__(
            action_space, cpu_preprocessor, gpu_preprocessor
        )

    @classmethod
    def from_args(cls, args, seed, **kwargs):
        return cls()

    def step(self, action):
        return {"screen": torch.rand((3, 84, 84))}, 1, False, {}

    def reset(self, **kwargs):
        return torch.rand((3, 84, 84))

    def close(self):
        return None


class TestRegistry(unittest.TestCase):
    def test_register_invalid_class(self):
        registry = Registry()
        with self.assertRaises(AssertionError):
            registry.register_agent(NotSubclass)

    def test_register_args_not_implemented(self):
        registry = Registry()
        with self.assertRaises(NotImplementedError):
            registry.register_agent(ArgsNotImplemented)

    def test_save_classes(self):
        dummy_log_id_dir = "/tmp/adept_test/test_save_classes"
        registry = Registry()
        registry.register_env(DummyEnv)
        registry.save_extern_classes(dummy_log_id_dir)

        other = Registry()
        other.load_extern_classes(dummy_log_id_dir)
        env_cls = other.lookup_env("dummy")
        env = env_cls()
        env.reset()


if __name__ == "__main__":
    unittest.main(verbosity=1)
