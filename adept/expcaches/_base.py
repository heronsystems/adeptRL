import abc


class BaseExperience(abc.ABC):
    @abc.abstractmethod
    def write_forward(self, items):
        raise NotImplementedError

    @abc.abstractmethod
    def write_env(self, obs, rewards, terminals, infos):
        raise NotImplementedError

    @abc.abstractmethod
    def read(self):
        raise NotImplementedError

    @abc.abstractmethod
    def is_ready(self):
        raise NotImplementedError
