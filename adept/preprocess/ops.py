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

cv2.ocl.setUseOpenCL(False)


class Operation(abc.ABC):
    def __init__(self, name_filters, rank_filters):
        self.name_filters = frozenset(name_filters) if name_filters else None
        self.rank_filters = frozenset(rank_filters) if (rank_filters and not
        name_filters) else None

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


class GrayScaleAndMoveChannel(Operation):
    def __init__(self, name_filters=None, rank_filters=frozenset([3])):
        super(GrayScaleAndMoveChannel, self).__init__(name_filters, rank_filters)

    def update_shape(self, old_shape):
        return {k: (1, ) + v[:-1] for k, v in old_shape.items()}

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        updated = {}
        for k, v in obs.items:
            if v.dim() == 3:
                result = torch.from_numpy(
                    cv2.cvtColor(v.numpy(), cv2.COLOR_RGB2GRAY)
                ).unsqueeze(0)
            elif v.dim() == 4:
                result = v.mean(dim=3).unsqueeze(1)
            else:
                raise ValueError(
                    'cant grayscale a rank' + str(obs.dim()) + ' tensor'
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
                result = F.interpolate(v, (84, 84), mode='area')
            else:
                raise ValueError(
                    'cant resize a rank' + str(obs.dim()) + ' tensor to 84x84'
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
            v = v.float()
            v *= (1. / 255.)
            updated[k] = v
        return updated


class FrameStack(Operation):
    def __init__(self, nb_frame, name_filters=None,
                 rank_filters=frozenset([3])):
        super(FrameStack, self).__init__(name_filters, rank_filters)
        self.nb_frame = nb_frame
        self.frames = deque([], maxlen=nb_frame)

    def update_shape(self, old_shape):
        updated = {}
        for k, v in old_shape.items():
            result = (v[0] * self.nb_frame,) + v[1:]
            updated[k] = result
        return updated

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        

    def _update_obs(self, obs):
        while len(self.frames) < self.nb_frame:
            self.frames.append(obs)

        self.frames.append(obs)
        if obs.dim() == 3:  # cpu
            if len(self.frames) == self.nb_frame:
                return torch.cat(list(self.frames))
        elif obs.dim() == 4:  # gpu
            if len(self.frames) == self.nb_frame:
                return torch.cat(list(self.frames), dim=1)

    def reset(self):
        self.frames = deque([], maxlen=self.nb_frame)


class FlattenSpace(Operation):
    def __init__(self, name_filters=None, rank_filters=None):
        super().__init__(name_filters, rank_filters)

    def update_shape(self, old_shape):
        return (reduce(lambda prev, cur: prev * cur, old_shape), )

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        return obs.view(-1)


class FromNumpy(Operation):
    def __init__(self, name_filters=None, rank_filters=None):
        super().__init__(name_filters, rank_filters)

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        if old_dtype == np.float32:
            return torch.float32
        elif old_dtype == np.float64:
            return torch.float64
        elif old_dtype == np.float16:
            return torch.float16
        elif old_dtype == np.uint8:
            return torch.uint8
        elif old_dtype == np.int8:
            return torch.int8
        elif old_dtype == np.int16:
            return torch.int16
        elif old_dtype == np.int32:
            return torch.int32
        elif old_dtype == np.int16:
            return torch.int16
        else:
            raise ValueError('Unsupported dtype {}'.format(old_dtype))

    def update_obs(self, obs):
        return torch.from_numpy(obs)
