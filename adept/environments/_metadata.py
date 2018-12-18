class EnvMetaData:
    """
    Used to provide environment metadata without spawning multiple processes.

    Networks need an action_space and observation_space
    Agents need an gpu_preprocessor, engine, and action_space
    """
    def __init__(self, env_plugin_class, args):
        dummy_env = env_plugin_class.from_args(args, 0)
        dummy_env.close()

        self.action_space = dummy_env.action_space
        self.observation_space = dummy_env.observation_space
        self.cpu_preprocessor = dummy_env.cpu_preprocessor
        self.gpu_preprocessor = dummy_env.gpu_preprocessor

