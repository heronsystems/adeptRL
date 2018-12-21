from adept.environments._env import HasEnvMetaData
from adept.registries.environment import EnvPluginRegistry


class EnvMetaData(HasEnvMetaData):
    """
    Used to provide environment metadata without spawning multiple processes.

    Networks need an action_space and observation_space
    Agents need an gpu_preprocessor, engine, and action_space
    """
    def __init__(self, env_plugin_class, args):
        dummy_env = env_plugin_class.from_args(args, 0)
        dummy_env.close()

        self._action_space = dummy_env.action_space
        self._observation_space = dummy_env.observation_space
        self._cpu_preprocessor = dummy_env.cpu_preprocessor
        self._gpu_preprocessor = dummy_env.gpu_preprocessor

    @classmethod
    def from_args(cls, args, registry=EnvPluginRegistry()):
        """
        Mimic the AdeptEnvPlugin.from_args API to simplify interface.

        :param args: Arguments object
        :param registry: Optionally provide to avoid recreating.
        :return: EnvMetaData
        """
        plugin_class = registry.lookup_env_class(args.env_id)
        return cls(plugin_class, args)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def cpu_preprocessor(self):
        return self._cpu_preprocessor

    @property
    def gpu_preprocessor(self):
        return self._gpu_preprocessor
