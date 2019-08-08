import abc
from collections import OrderedDict

import torch
from torch.nn import functional as F


class _BasePolicy(metaclass=abc.ABCMeta):
    """
    An abstract policy. A policy converts logits into actions and extra fields
    fields necessary for loss computation.
    """

    def __init__(self, action_space):
        self._action_sapce = action_space
        self._action_keys = list(sorted(action_space.keys()))

    def act(self, logits, available_actions=None):
        """
        ActionKey = str
        Action = Tensor (cpu)
        Extras = Dict[str, Tensor], extra fields with tensor's needed for loss
        computation. e.g.:
            log_probs: torch.Tensor
            entropies: torch.Tensor

        logits: Dict[ActionKey, torch.Tensor]
        :param available_actions:
            None if not needed
            torch.Tensor (N, NB_ACTION), one hot
        :return: Tuple[Action, Extras]
        """
        raise NotImplementedError


class ActorCriticHelper(_BasePolicy, metaclass=abc.ABCMeta):
    """
    A helper class for actor critic policies. Uses a cache to prevent
    recomputation.
    """

    def __init__(self, action_space):
        """
        :param action_space: Space
        """
        super(ActorCriticHelper, self).__init__(action_space)

        # Initialize cache
        self._cache = {
            'log_probs': None,
            'probs': None
        }

    def act(self, logits, available_actions=None):
        """
        :param logits: Dict[ActionKey, torch.Tensor]
        :param available_actions:
            None if not needed
            torch.Tensor (N, NB_ACTION), one hot
        :return:
        """
        raise NotImplementedError

    def clear_cache(self):
        self._cache = {k: None for k in self._cache.keys()}

    def flatten_logit(self, logit):
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

    def softmax(self, logit):
        """
        :param logit: torch.Tensor (N, X)
        :return:
        """
        if self._cache['probs']:
            return self._cache['probs']

        probs = F.softmax(logit, dim=1)
        self._cache['probs'] = probs
        return probs

    def log_softmax(self, logit):
        if self._cache['log_probs']:
            return self._cache['log_probs']

        log_probs = F.log_softmax(logit, dim=1)
        self._cache['log_probs'] = log_probs
        return log_probs

    def log_probability(self, logit, action):
        return self.log_softmax(logit).gather(1, action.unsqueeze(1))

    def entropy(self, logit):
        return -(
            self.log_softmax(logit) * self.softmax(logit)
        ).sum(1, keepdim=True)

    def sample_action(self, logit):
        return self.softmax(logit).multinomial(1).squeeze(1)


class ActorCriticPolicy(ActorCriticHelper):
    def act(self, logits, available_actions=None):
        actions = OrderedDict()
        log_probs = []
        entropies = []
        for key in self._action_keys:
            logit = self.flatten_logit(logits[key])
            action = self.sample_action(logit)

            log_probs.append(self.log_probability(logit, action))
            entropies.append(self.entropy(logit))
            actions[key] = action.cpu()
            self.clear_cache()

        log_probs = torch.cat(log_probs, dim=1)
        entropies = torch.cat(entropies, dim=1)
        return actions, {
            'log_probs': log_probs,
            'entropies': entropies
        }
