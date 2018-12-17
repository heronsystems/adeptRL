import abc

from adept.environments._base import AdeptEnv


class AdeptEnvPlugin(AdeptEnv, metaclass=abc.ABCMeta):
    """
    Implement this class to add your custom environment.
    """
    def __init__(self, observation_space, action_space, cpu_preprocessor,
                 gpu_preprocessor):
        """
        :param observation_space: ._spaces.Spaces
        :param action_space: ._spaces.Spaces
        :param cpu_preprocessor: adept.preprocess.observation.ObsPreprocessor
        :param gpu_preprocessor: adept.preprocess.observation.ObsPreprocessor
        """
        self._observation_space = observation_space
        self._action_space = action_space
        self._cpu_preprocessor = cpu_preprocessor
        self._gpu_preprocessor = gpu_preprocessor

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args, seed):
        raise NotImplementedError

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
