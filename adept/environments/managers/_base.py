import abc

from adept.environments._base import AdeptEnv
from adept.environments.registry import EnvPluginRegistry


class AdeptEnvManager(AdeptEnv, metaclass=abc.ABCMeta):
    def __init__(self, env_fns, engine):
        self._env_fns = env_fns
        self._engine = engine

    @property
    def env_fns(self):
        return self._env_fns

    @property
    def engine(self):
        return self._engine

    @classmethod
    def from_args(cls, args, registry=EnvPluginRegistry(), **kwargs):
        engine = registry.lookup_engine(args.env_id)
        env_class = registry.lookup_env_class(args.env_id)

        def build_env_fn(seed):
            def _f():
                return env_class.from_args(args, seed, **kwargs)
            return _f

        env_fns = []
        for i in range(args.nb_env):
            env_fns.append(build_env_fn(args.seed + i))
        return cls(env_fns, engine)
