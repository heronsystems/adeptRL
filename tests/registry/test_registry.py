import unittest
from adept.agent import AgentModule
from adept.registry import Registry


class NotSubclass:
    pass


class ArgsNotImplemented(AgentModule):
    pass


class TestRegistry(unittest.TestCase):
    def test_register_invalid_class(self):
        registry = Registry()
        with self.assertRaises(AssertionError):
            registry.register_agent(NotSubclass)

    def test_register_args_not_implemented(self):
        registry = Registry()
        with self.assertRaises(NotImplementedError):
            registry.register_agent(ArgsNotImplemented)


if __name__ == '__main__':
    unittest.main(verbosity=1)
