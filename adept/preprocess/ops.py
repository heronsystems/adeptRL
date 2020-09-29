from collections import deque
from functools import reduce

import cv2
import torch
from torch.nn import functional as F

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
        return torch.float32


class CastToDouble(SimpleOperation):
    def preprocess_cpu(self, tensor):
        return tensor.double()

    def preprocess_gpu(self, tensor):
        return tensor.double()

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return torch.float64


class CastToHalf(SimpleOperation):
    def preprocess_cpu(self, tensor):
        return tensor.half()

    def preprocess_gpu(self, tensor):
        return tensor.half()

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return torch.float16


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


class ResizeToNxM(SimpleOperation):
    def __init__(self, input_field, output_field, n, m):
        super().__init__(input_field, output_field)
        self.n = n
        self.m = m

    def update_shape(self, old_shape):
        return 1, self.n, self.m

    def update_dtype(self, old_dtype):
        return old_dtype

    def preprocess_cpu(self, tensor):
        if tensor.dim() == 3:
            temp = cv2.resize(
                tensor.squeeze(0).numpy(),
                (self.n, self.m),
                interpolation=cv2.INTER_AREA,
            )
            return torch.from_numpy(temp).unsqueeze(0)
        else:
            raise ValueError(
                "cant resize a rank"
                + str(tensor.dim())
                + " tensor to {}x{}".format(self.n, self.m)
            )

    def preprocess_gpu(self, tensor):
        if tensor.dim() == 4:
            return F.interpolate(tensor, (self.n, self.m), mode="area")
        else:
            raise ValueError(
                "cant resize a rank"
                + str(tensor.dim())
                + " tensor to {}x{}".format(self.n, self.m)
            )


class Divide(SimpleOperation):
    def __init__(self, input_field, output_field, n):
        super().__init__(input_field, output_field)
        self.n = n

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        if old_dtype in {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            return torch.float32
        else:
            return old_dtype

    def preprocess_cpu(self, tensor):
        return tensor * (1.0 / self.n)

    def preprocess_gpu(self, tensor):
        return tensor * (1.0 / self.n)


class FrameStackCPU(SimpleOperation):
    def __init__(self, input_field, output_field, nb_frame):
        super().__init__(input_field, output_field)
        self.nb_frame = nb_frame
        self.frames = None
        self.obs_space = None

    def update_shape(self, old_shape):
        if self.obs_space is None:
            self.obs_space = old_shape
            self.reset()
        return (old_shape[0] * self.nb_frame,) + old_shape[1:]

    def update_dtype(self, old_dtype):
        return old_dtype

    def preprocess_cpu(self, tensor):
        self.frames.append(tensor)
        if tensor.dim() == 3:
            if len(self.frames) == self.nb_frame:
                return torch.cat(list(self.frames))
        else:
            raise NotImplementedError(
                f"Dimensionality not supported: {tensor.dim()}"
            )

    def preprocess_gpu(self, tensor):
        raise NotImplementedError(f"GPU preprocessing not supported")

    def reset(self):
        self.frames = deque(
            [torch.zeros(self.obs_space)] * self.nb_frame, maxlen=self.nb_frame
        )


class FrameStackGPU(FrameStackCPU):
    def preprocess_cpu(self, tensor):
        raise NotImplementedError(f"CPU preprocessing not supported")

    def preprocess_gpu(self, tensor):
        if tensor.dim() == 4:
            if len(self.frames) == self.nb_frame:
                return torch.cat(list(self.frames), dim=1)
        else:
            raise NotImplementedError(
                f"Dimensionality not supported: {tensor.dim()}"
            )

    def reset(self):
        self.frames = deque(
            [torch.zeros((1,) + self.obs_space)] * self.nb_frame,
            maxlen=self.nb_frame,
        )


class FlattenSpace(SimpleOperation):
    def update_shape(self, old_shape):
        return (reduce(lambda prev, cur: prev * cur, old_shape),)

    def update_dtype(self, old_dtype):
        return old_dtype

    def preprocess_cpu(self, tensor):
        return tensor.view(-1)

    def preprocess_gpu(self, tensor):
        return tensor.view(-1)


class FromNumpy(SimpleOperation):
    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return numpy_to_torch_dtype(old_dtype)

    def preprocess_cpu(self, tensor):
        return torch.from_numpy(tensor)

    def preprocess_gpu(self, tensor):
        return torch.from_numpy(tensor)
