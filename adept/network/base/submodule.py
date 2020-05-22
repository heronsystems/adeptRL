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
        Parameters
        ----------
        input_shape : tuple[int]
            Input shape excluding batch dimension
        id : str
            Unique identifier for this instance
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
        """Output shape excluding batch dimension

        Returns
        -------
        tuple[int]
            Output shape exlcuding batch dimension
        """
        raise NotImplementedError

    def output_shape(self, dim=None):
        """Output shape casted to requested dimension

        Parameters
        ----------
        dim : int, optional
            Desired dimensionality, defaults to native

        Returns
        -------
        tuple[int]
            Output shape
        """
        if dim is None:
            dim = len(self._output_shape)
        if dim == 1:
            return self._to_1d_shape()
        elif dim == 2 or dim is None:
            return self._to_2d_shape()
        elif dim == 3:
            return self._to_3d_shape()
        elif dim == 4:
            return self._to_4d_shape()
        else:
            raise ValueError("Invalid dim: {}".format(dim))

    @abc.abstractmethod
    def _forward(self, input, internals, **kwargs):
        """
        :param input: torch.Tensor (B+1D | B+2D | B+3D | B+4D)
        :return: Tuple[Result, Internals]
        """
        raise NotImplementedError

    def _to_1d(self, submodule_output):
        """Convert to Batch + 1D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor

        Returns
        -------
        torch.Tensor
            Batch + 1D Tensor
        """
        b = submodule_output.size()[0]
        return submodule_output.view(b, *self._to_1d_shape())

    def _to_2d(self, submodule_output):
        """Convert to Batch + 2D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor (B, S, F)

        Returns
        -------
        torch.Tensor
            Batch + 2D Tensor (B, S, F)
        """
        b = submodule_output.size()[0]
        return submodule_output.view(b, *self._to_2d_shape())

    def _to_3d(self, submodule_output):
        """Convert to Batch + 3D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor (B, S, F)

        Returns
        -------
        torch.Tensor
            Batch + 3D Tensor
        """
        b = submodule_output.size()[0]
        return submodule_output.view(b, *self._to_3d_shape())

    def _to_4d(self, submodule_output):
        """Convert to Batch + 4D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor (B, S, F)

        Returns
        -------
        torch.Tensor
            Batch + 4D Tensor (B, F, S, H, W)
        """
        b = submodule_output.size()[0]
        return submodule_output.view(b, *self._to_4d_shape())

    @abc.abstractmethod
    def _to_1d_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_2d_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_3d_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_4d_shape(self):
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
            raise ValueError("Invalid dim: {}".format(dim))
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
            raise ValueError("Invalid dim: {}".format(dim))

    def _id_internals(self, internals):
        return {self.id + k: v for k, v in internals.items()}
