"""
Copyright (C) 2018 Heron Systems, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import abc
from collections import deque
from functools import reduce

import cv2
import numpy as np
import torch
from torch.nn.functional import upsample

from adept.environments._base import Space

cv2.ocl.setUseOpenCL(False)


class BaseOp(abc.ABC):
    def __init__(self, filter_names=set(), filter_ranks=set()):
        if filter_names:
            self.filters = set(filter_names)
        elif filter_ranks:
            self.filters = set(filter_ranks)
        else:
            self.filters = set()

    def filter(self, name, rank):
        if self.filters:
            return name in self.filters or rank in self.filters
        else:
            return True

    def reset(self):
        pass
    
    @abc.abstractmethod
    def update_space(self, old_space):
        raise NotImplementedError

    @abc.abstractmethod
    def update_obs(self, obs):
        raise NotImplementedError


class CastToFloat(BaseOp):
    def __init__(self, filter_names=set()):
        super(CastToFloat, self).__init__(filter_names)

    def update_space(self, old_space):
        return Space(old_space.shape, old_space.low, old_space.high, np.float32)

    def update_obs(self, obs):
        return obs.float()


class GrayScaleAndMoveChannel(BaseOp):
    def __init__(self, filter_names=set()):
        super(GrayScaleAndMoveChannel, self).__init__(filter_names, {3})

    def update_space(self, old_space):
        return Space((1,) + old_space.shape[:-1], old_space.low, old_space.high, old_space.dtype)

    def update_obs(self, obs):
        if obs.dim() == 3:
            return torch.from_numpy(cv2.cvtColor(obs.numpy(), cv2.COLOR_RGB2GRAY)).unsqueeze(0)
            # return obs.mean(dim=2).unsqueeze(0)
        elif obs.dim() == 4:
            return obs.mean(dim=3).unsqueeze(1)
        else:
            raise ValueError('cant grayscale a rank' + str(obs.dim()) + ' tensor')


class ResizeTo84x84(BaseOp):
    def __init__(self, filter_names=set()):
        super().__init__(filter_names, {3})

    def update_space(self, old_space):
        return Space((1, 84, 84), old_space.low, old_space.high, old_space.dtype)

    def update_obs(self, obs):
        if obs.dim() == 3:
            obs = cv2.resize(obs.squeeze(0).numpy(), (84, 84), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(obs).unsqueeze(0)
            # return upsample(obs.unsqueeze(0), (84, 84), mode='bilinear').squeeze(0)
        elif obs.dim() == 4:
            return upsample(obs, (84, 84), mode='bilinear')
        else:
            raise ValueError('cant resize a rank' + str(obs.dim()) + ' tensor to 84x84')


class Divide255(BaseOp):
    def __init__(self, filter_names=set()):
        super().__init__(filter_names, {3})

    def update_space(self, old_space):
        return Space(old_space.shape, 0., 1., np.float32)

    def update_obs(self, obs):
        obs *= (1. / 255.)
        return obs


class FrameStack(BaseOp):
    def __init__(self, nb_frame, filter_names=set()):
        super(FrameStack, self).__init__(filter_names, {3})
        self.nb_frame = nb_frame
        self.frames = deque([], maxlen=nb_frame)

    def update_space(self, old_space):
        new_shape = (old_space.shape[0] * self.nb_frame,) + old_space.shape[1:]
        return Space(
            new_shape,
            old_space.low,
            old_space.high,
            old_space.dtype
        )

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

class FlattenSpace(BaseOp):
    def __init__(self, filter_names=set()):
        super(FlattenSpace, self).__init__(filter_names)

    def update_space(self, old_space):
        return Space((reduce(lambda prev, cur: prev * cur, old_space.shape),), old_space.low, old_space.high, old_space.dtype)

    def update_obs(self, obs):
        return obs.view(-1)

class FromNumpy(BaseOp):
    def __init__(self, filter_names=set()):
        super().__init__(filter_names)

    def update_space(self, old_space):
        return Space(old_space.shape, old_space.low, old_space.high, old_space.dtype)

    def update_obs(self, obs):
        return torch.from_numpy(obs)
