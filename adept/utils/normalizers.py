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
import torch
import numpy as np


class Normalizer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, reward):
        """
        :param reward: FloatTensor
        :return:
        """
        raise NotImplementedError


class Clip(Normalizer):
    def __init__(self, floor=-1, ceil=1):
        self.floor = floor
        self.ceil = ceil

    def __call__(self, reward):
        return torch.clamp(reward, self.floor, self.ceil)


class Scale(Normalizer):
    def __init__(self, coefficient):
        self.coefficient = coefficient

    def __call__(self, reward):
        return self.coefficient * reward


class ScaleAtari(Normalizer):
    def __init__(self, scale=10 ** -3):
        self.scale = scale

    def __call__(self, item):
        return copy_sign(item) * (torch.sqrt(torch.abs(item) + 1) - 1) \
            + self.scale * item


def copy_sign(x):
    eps = 1e-15
    return (x + eps) / torch.abs(x + eps)
