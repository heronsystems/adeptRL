import cv2
import torch

from adept.preprocess.base import SimpleOperation
from adept.utils.util import numpy_to_torch_dtype

cv2.ocl.setUseOpenCL(False)


class CastToFloat(SimpleOperation):

    def preprocess_cpu(self, tensor):
        return tensor.float()

    def preprocess_gpu(self, tensor):
        return tensor.float()

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        old_dtype[self.output_field] = torch.float32
        return old_dtype


class CastToDouble(SimpleOperation):

    def preprocess_cpu(self, tensor):
        return tensor.double()

    def preprocess_gpu(self, tensor):
        return tensor.double()

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        old_dtype[self.output_field] = torch.float64
        return old_dtype


class CastToHalf(SimpleOperation):

    def preprocess_cpu(self, tensor):
        return tensor.half()

    def preprocess_gpu(self, tensor):
        return tensor.half()

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        old_dtype[self.output_field] = torch.float16
        return old_dtype


class GrayScaleAndMoveChannel(SimpleOperation):

    def update_shape(self, old_shape):
        return (1,) + old_shape[:-1]

    def update_dtype(self, old_dtype):
        return old_dtype

    def preprocess_cpu(self, tensor):
        if tensor.dim() == 3:
            return torch.from_numpy(
                cv2.cvtColor(tensor.numpy(), cv2.COLOR_RGB2GRAY)
            ).unsqueeze(0)
        else:
            raise ValueError(
                "can't grayscale a rank" + str(tensor.dim()) + "tensor"
            )

    def preprocess_gpu(self, tensor):
        if tensor.dim() == 4:
            return tensor.mean(dim=3).unsqueeze(1)
        else:
            raise ValueError(
                "can't grayscale a rank" + str(tensor.dim()) + "tensor"
            )


class FromNumpy(SimpleOperation):

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
