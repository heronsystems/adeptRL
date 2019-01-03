import unittest
from adept.agents.agent_plugin import AgentPlugin
from adept.agents.agent_registry import AgentRegistry


class NotSubclass:
    pass


class ArgsNotImplemented(AgentPlugin):
    pass


class TestAgentRegistry(unittest.TestCase):

    def test_register_invalid_class(self):
        registry = AgentRegistry()
        with self.assertRaises(AssertionError):
            registry.register_agent('not_subclass', NotSubclass)

    def test_register_args_not_implemented(self):
        registry = AgentRegistry()
        with self.assertRaises(NotImplementedError):
            registry.register_agent('args_not_implemented', ArgsNotImplemented)


if __name__ == '__main__':
    unittest.main(verbosity=1)
