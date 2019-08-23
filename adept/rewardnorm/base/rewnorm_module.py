import abc

from adept.utils.requires_args import RequiresArgsMixin


class RewardNormModule(RequiresArgsMixin, metaclass=abc.ABCMeta):
    def __call__(self, reward):
        """
        Normalizes a reward tensor.

        :param reward: torch.Tensor (1D)
        :return:
        """
        raise NotImplementedError
