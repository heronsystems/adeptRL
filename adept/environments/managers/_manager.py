import abc

from adept.environments._env import EnvBase
from adept.environments.registry import EnvPluginRegistry


class AdeptEnvManager(EnvBase, metaclass=abc.ABCMeta):
    def __init__(self, env_fns, engine):
        self._env_fns = env_fns
        self._engine = engine

    @property
    def env_fns(self):
        return self._env_fns

    @property
    def engine(self):
        return self._engine

    @property
    def nb_env(self):
        return len(self._env_fns)

    @classmethod
    def from_args(
        cls, args, seed_start=None, registry=EnvPluginRegistry(), **kwargs
    ):
        if seed_start is None:
            seed_start = int(args.seed)

        engine = registry.lookup_engine(args.env_id)
        env_class = registry.lookup_env_class(args.env_id)

        env_fns = []
        for i in range(args.nb_env):
            env_fns.append(
                env_class.from_args_curry(args, seed_start + i, **kwargs)
            )
        return cls(env_fns, engine)
