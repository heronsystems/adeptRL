import abc

from adept.environments._env import EnvBase


class EnvPlugin(EnvBase, metaclass=abc.ABCMeta):
    """
    Implement this class to add your custom environment.
    """
    def __init__(self, action_space, cpu_preprocessor,
                 gpu_preprocessor):
        """
        :param observation_space: ._spaces.Spaces
        :param action_space: ._spaces.Spaces
        :param cpu_preprocessor: adept.preprocess.observation.ObsPreprocessor
        :param gpu_preprocessor: adept.preprocess.observation.ObsPreprocessor
        """
        self._action_space = action_space
        self._cpu_preprocessor = cpu_preprocessor
        self._gpu_preprocessor = gpu_preprocessor

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args, seed, **kwargs):
        """
        :param args: Arguments object
        :param seed: Integer used to seed this environment.
        :param kwargs: Any custom arguments are passed through kwargs.
        :return: EnvPlugin instance.
        """
        raise NotImplementedError

    @classmethod
    def from_args_curry(cls, args, seed, **kwargs):
        def _f():
            return cls.from_args(args, seed, **kwargs)
        return _f

    @property
    def observation_space(self):
        return self._gpu_preprocessor.observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def cpu_preprocessor(self):
        return self._cpu_preprocessor

    @property
    def gpu_preprocessor(self):
        return self._gpu_preprocessor
