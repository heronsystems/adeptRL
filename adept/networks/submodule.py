import abc

import torch

from adept.utils.requires_args import RequiresArgs


class SubModule(torch.nn.Module, RequiresArgs, metaclass=abc.ABCMeta):
    """
    SubModule of a ModularNetwork.
    """
    dim = None

    def __init__(self, input_shape, id):
        super(SubModule, self).__init__()
        self._input_shape = input_shape
        self._id = id

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args, input_shape):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _output_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self, dim=None):
        raise NotImplementedError

    @abc.abstractmethod
    def _forward(self, *input):
        """
        :param input: torch.Tensor (1D | 2D | 3D | 4D)
        :return: Tuple[Result, Internals]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _to_1d(self, submodule_output):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_2d(self, submodule_output):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_3d(self, submodule_output):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_4d(self, submodule_output):
        raise NotImplementedError

    @property
    def id(self):
        return self._id

    @property
    def input_shape(self):
        return self._input_shape

    def to_dim(self, submodule_output, dim):
        """
        :param submodule_output: torch.Tensor (1D | 2D | 3D | 4D)
        Output of a forward pass to be converted.
        :param dim: int Desired dimensionality
        :return:
        """
        if dim <= 0 or dim > 4:
            raise ValueError('Invalid dim: {}'.format(dim))
        elif dim == 1:
            return self._to_1d(submodule_output)
        elif dim == 2:
            return self._to_2d(submodule_output)
        elif dim == 3:
            return self._to_3d(submodule_output)
        elif dim == 4:
            return self._to_4d(submodule_output)

    def forward(self, *input, dim=None):
        submodule_output, internals = self._forward(*input)
        if dim is None:
            return submodule_output, internals
        if dim == 1:
            return self._to_1d(submodule_output), internals
        elif dim == 2:
            return self._to_2d(submodule_output), internals
        elif dim == 3:
            return self._to_3d(submodule_output), internals
        elif dim == 4:
            return self._to_4d(submodule_output), internals
        else:
            raise ValueError('Invalid dim: {}'.format(dim))
