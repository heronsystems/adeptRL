import abc
import torch
from torch.nn import functional as F


class ACActorHelperMixin(metaclass=abc.ABCMeta):
    """
    A helper class for actor critic actors.
    """

    @staticmethod
    def flatten_logits(logit):
        """
        :param logits: Tensor of arbitrary dim
        :return: logits flattened to (N, X)
        """
        size = logit.size()
        dim = logit.dim()

        if dim == 3:
            n, f, l = size
            logit = logit.view(n, f * l)
        elif dim == 4:
            n, f, h, w = size
            logit = logit.view(n, f * h * w)
        elif dim == 5:
            n, f, d, h, w = size
            logit = logit.view(n, f * d * h * w)
        return logit

    @staticmethod
    def softmax(logit):
        """
        :param logit: torch.Tensor (N, X)
        :return: torch.Tensor (N, X)
        """
        return F.softmax(logit, dim=1)

    @staticmethod
    def log_softmax(logit):
        """
        :param logit: torch.Tensor (N, X)
        :return: torch.Tensor (N, X)
        """
        return F.log_softmax(logit, dim=1)

    @staticmethod
    def log_probability(log_softmax, action):
        """
        :param log_softmax: Tensor (N, X)
        :param action: LongTensor (N)
        :return: Tensor (N, 1)
        """
        return log_softmax.gather(1, action.unsqueeze(1))

    @staticmethod
    def entropy(log_softmax, softmax):
        """
        :param log_softmax: Tensor (N, X)
        :param softmax: Tensor (N, X)
        :return: Tensor (N, 1)
        """
        return -(log_softmax * softmax).sum(1, keepdim=True)

    @staticmethod
    def sample_action(softmax):
        """
        Samples an action from a softmax distribution.

        :param softmax: torch.Tensor (N, X)
        :return: torch.Tensor (N)
        """
        return softmax.multinomial(1).squeeze(1)

    @staticmethod
    def select_action(softmax):
        """
        Selects the action with the highest probability.

        :param softmax:
        :return:
        """
        return torch.argmax(softmax, dim=1)
