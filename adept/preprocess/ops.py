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
from torch.nn.functional import upsample
import numpy as np

cv2.ocl.setUseOpenCL(False)


class Operation(abc.ABC):
    def __init__(self, filter_names=frozenset(), filter_ranks=frozenset()):
        if filter_names:
            self.filters = frozenset(filter_names)
        elif filter_ranks:
            self.filters = frozenset(filter_ranks)
        else:
            self.filters = frozenset()

    def filter(self, name, rank):
        if self.filters:
            return name in self.filters or rank in self.filters
        else:
            return True

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
    def __init__(self, filter_names=set()):
        super(CastToFloat, self).__init__(filter_names)

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return torch.float32

    def update_obs(self, obs):
        return obs.float()


class GrayScaleAndMoveChannel(Operation):
    def __init__(self, filter_names=set()):
        super(GrayScaleAndMoveChannel, self).__init__(filter_names, {3})

    def update_shape(self, old_shape):
        return (1, ) + old_shape[:-1]

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        if obs.dim() == 3:
            return torch.from_numpy(
                cv2.cvtColor(obs.numpy(), cv2.COLOR_RGB2GRAY)
            ).unsqueeze(0)
            # return obs.mean(dim=2).unsqueeze(0)
        elif obs.dim() == 4:
            return obs.mean(dim=3).unsqueeze(1)
        else:
            raise ValueError(
                'cant grayscale a rank' + str(obs.dim()) + ' tensor'
            )


class ResizeTo84x84(Operation):
    def __init__(self, filter_names=set()):
        super().__init__(filter_names, {3})

    def update_shape(self, old_shape):
        return (1, 84, 84)

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        if obs.dim() == 3:
            obs = cv2.resize(
                obs.squeeze(0).numpy(), (84, 84), interpolation=cv2.INTER_AREA
            )
            return torch.from_numpy(obs).unsqueeze(0)
            # return upsample(obs.unsqueeze(0), (84, 84), mode='bilinear').squeeze(0)
        elif obs.dim() == 4:
            return upsample(obs, (84, 84), mode='bilinear')
        else:
            raise ValueError(
                'cant resize a rank' + str(obs.dim()) + ' tensor to 84x84'
            )


class Divide255(Operation):
    def __init__(self, filter_names=set()):
        super().__init__(filter_names, {3})

    def update_shape(self, old_shape):
        return old_shape

    def update_dtype(self, old_dtype):
        return torch.float32

    def update_obs(self, obs):
        obs = obs.float()
        obs *= (1. / 255.)
        return obs


class FrameStack(Operation):
    def __init__(self, nb_frame, filter_names=set()):
        super(FrameStack, self).__init__(filter_names, {3})
        self.nb_frame = nb_frame
        self.frames = deque([], maxlen=nb_frame)

    def update_shape(self, old_shape):
        return (old_shape[0] * self.nb_frame,) + old_shape[1:]

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
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
    def __init__(self, filter_names=set()):
        super(FlattenSpace, self).__init__(filter_names)

    def update_shape(self, old_shape):
        return (reduce(lambda prev, cur: prev * cur, old_shape), )

    def update_dtype(self, old_dtype):
        return old_dtype

    def update_obs(self, obs):
        return obs.view(-1)


class FromNumpy(Operation):
    def __init__(self, filter_names=set()):
        super().__init__(filter_names)

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
