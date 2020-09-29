import abc
import cv2
import torch

from adept.utils.util import numpy_to_torch_dtype

cv2.ocl.setUseOpenCL(False)


class MultiOperation(abc.ABC):
    def __init__(
        self,
        input_fields,
        output_fields,
        in_dim=None,
        out_dim=None,
        in_dtype=None,
        out_dtype=None,
    ):
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    @abc.abstractmethod
    def preprocess_cpu(self, tensors):
        raise NotImplemented

    @abc.abstractmethod
    def preprocess_gpu(self, tensors):
        raise NotImplemented

    @abc.abstractmethod
    def update_shape(self, old_shapes):
        raise NotImplementedError

    def update_dtype(self, old_dtypes):
        raise NotImplementedError

    def to(self, device):
        return self


class Operation(abc.ABC):
    def __init__(
        self,
        input_field,
        output_field,
        in_dim=None,
        out_dim=None,
        in_dtype=None,
        out_dtype=None,
    ):
        self.input_field = input_field
        self.output_field = output_field
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    @abc.abstractmethod
    def preprocess_cpu(self, tensor):
        raise NotImplemented

    @abc.abstractmethod
    def preprocess_gpu(self, tensor):
        raise NotImplemented

    @abc.abstractmethod
    def update_shape(self, old_shape):
        raise NotImplementedError

    def update_dtype(self, old_dtype):
        raise NotImplementedError

    def to(self, device):
        return self


class CastToFloat(Operation):
    def __init__(
        self,
        input_field,
        output_field,
        in_dim=None,
        out_dim=None,
        in_dtype=None,
        out_dtype=None,
    ):
        super().__init__(
            input_field, output_field, in_dim, out_dim, in_dtype, out_dtype
        )

    def preprocess_cpu(self, tensor):
        return tensor.float()

    def preprocess_gpu(self, tensor):
        return tensor.float()

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        old_dtype[self.output_field] = torch.float32
        return old_dtype


class CastToDouble(Operation):
    def __init__(
        self,
        input_field,
        output_field,
        in_dim=None,
        out_dim=None,
        in_dtype=None,
        out_dtype=None,
    ):
        super().__init__(
            input_field, output_field, in_dim, out_dim, in_dtype, out_dtype
        )

    def preprocess_cpu(self, tensor):
        return tensor.double()

    def preprocess_gpu(self, tensor):
        return tensor.double()

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        old_dtype[self.output_field] = torch.float64
        return old_dtype


class CastToHalf(Operation):
    def __init__(
        self,
        input_field,
        output_field,
        in_dim=None,
        out_dim=None,
        in_dtype=None,
        out_dtype=None,
    ):
        super().__init__(
            input_field, output_field, in_dim, out_dim, in_dtype, out_dtype
        )

    def preprocess_cpu(self, tensor):
        return tensor.half()

    def preprocess_gpu(self, tensor):
        return tensor.half()

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        old_dtype[self.output_field] = torch.float16
        return old_dtype


class GrayScaleAndMoveChannel(Operation):
    def __init__(
        self,
        input_field,
        output_field,
        in_dim=None,
        out_dim=None,
        in_dtype=None,
        out_dtype=None,
    ):
        super().__init__(
            input_field, output_field, in_dim, out_dim, in_dtype, out_dtype
        )

    def update_shape(self, old_shape):
        return (1,) + old_shape[:-1]

    def update_dtype(self, old_dtype):
        return old_dtype

    def preprocess_cpu(self, tensor):
        if tensor.dim() == 3:
            return torch.from_numpy(
                cv2.cvtColor(tensor.numpy(), cv2.COLOR_RGB2GRAY)
            ).unsqueeze(0)
        elif tensor.dim() == 4:
            return tensor.mean(dim=3).unsqueeze(1)
        else:
            raise ValueError(
                "can't grayscale a rank" + str(tensor.dim()) + "tensor"
            )

    def preprocess_gpu(self, tensor):
        if tensor.dim() == 3:
            return torch.from_numpy(
                cv2.cvtColor(tensor.numpy(), cv2.COLOR_RGB2GRAY)
            ).unsqueeze(0)
        elif tensor.dim() == 4:
            return tensor.mean(dim=3).unsqueeze(1)
        else:
            raise ValueError(
                "can't grayscale a rank" + str(tensor.dim()) + "tensor"
            )


class FromNumpy(Operation):
    def __init__(
        self,
        input_field,
        output_field,
        in_dim=None,
        out_dim=None,
        in_dtype=None,
        out_dtype=None,
    ):
        super().__init__(
            input_field, output_field, in_dim, out_dim, in_dtype, out_dtype
        )

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        old_dtype[self.output_field] = numpy_to_torch_dtype(
            old_dtype[self.output_field]
        )
        return old_dtype

    def preprocess_cpu(self, tensor):
        return torch.from_numpy(tensor)

    def preprocess_gpu(self, tensor):
        return torch.from_numpy(tensor)
