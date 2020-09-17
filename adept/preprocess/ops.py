# Copyright (C) 2018 Heron Systems, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import abc
from collections import deque
from functools import reduce

import cv2
import torch
from torch.nn import functional as F
import numpy as np

from adept.utils.util import numpy_to_torch_dtype

cv2.ocl.setUseOpenCL(False)


class Operation(abc.ABC):
    def __init__(self, name_filters, rank_filters):
        self.name_filters = (
            frozenset(name_filters) if name_filters else frozenset()
        )
        self.rank_filters = (
            frozenset(rank_filters)
            if (rank_filters and not name_filters)
            else frozenset()
        )

    def reset(self):
        pass

    @abc.abstractmethod
    def update_shape(self, old_shape):
        raise NotImplementedError

    def update_dtype(self, old_dtype):
        raise NotImplementedError

    @abc.abstractmethod
    def update_obs(self, obs):
        raise NotImplementedError


class CastToFloat(Operation):
    def __init__(self, name_filters=None, rank_filters=None):
        super(CastToFloat, self).__init__(name_filters, rank_filters)

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return {k: torch.float32 for k in old_dtype.keys()}

    def update_obs(self, obs):
        return {k: ob.float() for k, ob in obs.items()}


class CastToDouble(Operation):
    def __init__(self, name_filters=None, rank_filters=None):
        super().__init__(name_filters, rank_filters)

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return {k: torch.float64 for k in old_dtype.keys()}

    def update_obs(self, obs):
        return {k: ob.float() for k, ob in obs.items()}


class CastToHalf(Operation):
    def __init__(self, name_filters=None, rank_filters=None):
        super(CastToHalf, self).__init__(name_filters, rank_filters)

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return {k: torch.float16 for k in old_dtype.keys()}

    def update_obs(self, obs):
        return {k: ob.half() for k, ob in obs.items()}


class GrayScaleAndMoveChannel(Operation):
    def __init__(self, name_filters=None, rank_filters=frozenset([3])):
        super(GrayScaleAndMoveChannel, self).__init__(
            name_filters, rank_filters
        )

    def update_shape(self, old_shape):
        return {k: (1,) + v[:-1] for k, v in old_shape.items()}

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        updated = {}
        for k, v in obs.items():
            if v.dim() == 3:
                result = torch.from_numpy(
                    cv2.cvtColor(v.numpy(), cv2.COLOR_RGB2GRAY)
                ).unsqueeze(0)
            elif v.dim() == 4:
                result = v.mean(dim=3).unsqueeze(1)
            else:
                raise ValueError(
                    "cant grayscale a rank" + str(obs.dim()) + " tensor"
                )
            updated[k] = result
        return updated


class ResizeTo84x84(Operation):
    def __init__(self, name_filters=None, rank_filters=frozenset([3])):
        super().__init__(name_filters, rank_filters)

    def update_shape(self, old_shape):
        return {k: (1, 84, 84) for k, v in old_shape.items()}

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        updated = {}
        for k, v in obs.items():
            if v.dim() == 3:
                result = cv2.resize(
                    v.squeeze(0).numpy(), (84, 84), interpolation=cv2.INTER_AREA
                )
                result = torch.from_numpy(result).unsqueeze(0)
            elif v.dim() == 4:
                result = F.interpolate(v, (84, 84), mode="area")
            else:
                raise ValueError(
                    "cant resize a rank" + str(obs.dim()) + " tensor to 84x84"
                )
            updated[k] = result
        return updated


class Divide255(Operation):
    def __init__(self, name_filters=None, rank_filters=frozenset([3])):
        super().__init__(name_filters, rank_filters)

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return {k: torch.float32 for k in old_dtype.keys()}

    def update_obs(self, obs):
        updated = {}
        for k, v in obs.items():
            if k in self.name_filters:
                v = v.float()
                v *= 1.0 / 255.0
            updated[k] = v
        return updated


class FrameStackCPU(Operation):
    def __init__(
        self, nb_frame, name_filters=None, rank_filters=frozenset([3])
    ):
        super().__init__(name_filters, rank_filters)
        self.nb_frame = nb_frame
        self.frames = None
        self.obs_space = None

    def update_shape(self, old_shape):
        # lazily initialize old observation space
        if self.obs_space is None:
            self.obs_space = old_shape
            self.reset()
        updated = {}
        for k, v in old_shape.items():
            result = (v[0] * self.nb_frame,) + v[1:]
            updated[k] = result
        return updated

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        updated = {}
        for k, v in obs.items():
            self.frames[k].append(v)
            updated[k] = self._update_obs(v)

        return updated

    def _update_obs(self, obs):
        if obs.dim() == 3:  # cpu
            if len(self.frames) == self.nb_frame:
                return torch.cat(list(self.frames))
        else:
            raise NotImplementedError(
                f"Dimensionality not supported: {obs.dim()}"
            )

    def reset(self):
        self.frames = {
            k: deque([torch.zeros(dims)] * self.nb_frame, maxlen=self.nb_frame)
            for k, dims in self.obs_space.items()
        }


class FrameStackGPU(FrameStackCPU):
    def reset(self):
        self.frames = {
            k: deque(
                [torch.zeros((1,) + v)] * self.nb_frame, maxlen=self.nb_frame
            )
            for k, v in self.obs_space.items()
        }

    def _update_obs(self, obs):
        if obs.dim() == 4:
            if len(self.frames) == self.nb_frame:
                return torch.cat(list(self.frames), dim=1)
        else:
            raise NotImplementedError(
                f"Dimensionality not supported: {obs.dim()}"
            )


class FlattenSpace(Operation):
    def __init__(self, name_filters=None, rank_filters=None):
        super().__init__(name_filters, rank_filters)

    def update_shape(self, old_shape):
        updated = {}
        for k, olds in old_shape.items():
            updated[k] = (reduce(lambda prev, cur: prev * cur, olds),)
        return updated

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        updated = {}
        for k, v in obs.items():
            updated[k] = v.view(-1)
        return updated


class FromNumpy(Operation):
    def __init__(self, name_filters=None, rank_filters=None):
        super().__init__(name_filters, rank_filters)

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return {k: numpy_to_torch_dtype(v) for k, v in old_dtype.items()}

    def update_obs(self, obs):
        return {k: torch.from_numpy(v) for k, v in obs.items()}


if __name__ == "__main__":
    # fstack = FrameStackCPU(4)
    # fstack.update_shape({"box": (3, 5, 5)})
    # fstack.update_obs({"box": torch.ones(3, 5, 5)})
    # print(fstack.frames)

    fstack = FrameStackGPU(4)
    fstack.update_shape({"box": (3, 5, 5)})
    fstack.update_obs({"box": torch.ones(3, 3, 5, 5)})
    print(fstack.frames)
