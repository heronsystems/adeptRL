import abc

import torch

from adept.utils.requires_args import RequiresArgsMixin


class SubModule(torch.nn.Module, RequiresArgsMixin, metaclass=abc.ABCMeta):
    """
    SubModule of a ModularNetwork.
    """
    dim = None

    def __init__(self, input_shape, id):
        """
        :param input_shape: Tuple[*Dim] Input shape excluding batch dimension
        :param id: str Unique identifier for this instance
        """
        super(SubModule, self).__init__()
        self._input_shape = input_shape
        self._id = id

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args, input_shape, id):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _output_shape(self):
        """
        :return: Tuple[*Dim] Output shape excluding batch dimension
        """
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self, dim=None):
        raise NotImplementedError

    @abc.abstractmethod
    def _forward(self, input, internals, **kwargs):
        """
        :param input: torch.Tensor (B+1D | B+2D | B+3D | B+4D)
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

    @abc.abstractmethod
    def _new_internals(self):
        """
        :return: Dict[InternalKey, List[torch.Tensor (ND)]]
        """
        raise NotImplementedError

    @property
    def id(self):
        return self._id

    @property
    def input_shape(self):
        return self._input_shape

    def new_internals(self, device):
        return {
            self.id + k: v.to(device) for k, v in self._new_internals().items()
        }

    def stacked_internals(self, key, internals):
        return torch.stack(internals[self.id + key])

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
            return submodule_output, self._id_internals(internals)
        if dim == 1:
            return self._to_1d(submodule_output), self._id_internals(internals)
        elif dim == 2:
            return self._to_2d(submodule_output), self._id_internals(internals)
        elif dim == 3:
            return self._to_3d(submodule_output), self._id_internals(internals)
        elif dim == 4:
            return self._to_4d(submodule_output), self._id_internals(internals)
        else:
            raise ValueError('Invalid dim: {}'.format(dim))

    def _id_internals(self, internals):
        return {self.id + k: v for k, v in internals.items()}
