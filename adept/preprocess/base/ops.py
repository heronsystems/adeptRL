import abc


class Operation(abc.ABC):
    @abc.abstractmethod
    def update_shape(self, old_shape):
        raise NotImplementedError

    @abc.abstractmethod
    def update_dtype(self, old_dtype):
        raise NotImplementedError

    def to(self, device):
        return self


class MultiOperation(Operation, metaclass=abc.ABCMeta):
    """Modofies multiple keys of an observation dictionary."""

    def __init__(self, input_fields, output_fields):
        self.input_fields = input_fields
        self.output_fields = output_fields

    @abc.abstractmethod
    def preprocess_cpu(self, tensors):
        """Preprocess multiple observation fields on the CPU.

        Parameters
        ----------
        tensors : list[torch.Tensor]

        Returns
        -------
        list[torch.Tensor]
        """
        raise NotImplemented

    @abc.abstractmethod
    def preprocess_gpu(self, tensors):
        """Preprocess multiple observation fields on the GPU.

        Parameters
        ----------
        tensors : list[torch.Tensor]

        Returns
        -------
        list[torch.Tensor]
        """
        raise NotImplemented


class SimpleOperation(Operation, metaclass=abc.ABCMeta):
    """Modifies a single key in the observation dictionary."""

    def __init__(self, input_field, output_field):
        self.input_field = input_field
        self.output_field = output_field

    @abc.abstractmethod
    def preprocess_cpu(self, tensor):
        """Preprocess a specific field of an observation on the CPU.

        Parameters
        ----------
        tensor : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        raise NotImplemented

    @abc.abstractmethod
    def preprocess_gpu(self, tensor):
        """Preprocess a specific field of an observation on the GPU

        Parameters
        ----------
        tensor : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        raise NotImplemented
