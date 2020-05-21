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
import torch
from .base import RewardNormModule


class Clip(RewardNormModule):
    args = {"floor": -1, "ceil": 1}

    def __init__(self, floor, ceil):
        self.floor = floor
        self.ceil = ceil

    @classmethod
    def from_args(cls, args):
        return cls(args.floor, args.ceil)

    def __call__(self, reward):
        return torch.clamp(reward, self.floor, self.ceil)


class Scale(RewardNormModule):
    args = {"coefficient": 0.1}

    def __init__(self, coefficient):
        self.coefficient = coefficient

    @classmethod
    def from_args(cls, args):
        return cls(args.coefficient)

    def __call__(self, reward):
        return self.coefficient * reward


class Identity(RewardNormModule):
    args = {}

    @classmethod
    def from_args(cls, args):
        return cls()

    def __call__(self, item):
        return item
