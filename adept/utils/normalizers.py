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


class NormalizerBase(abc.ABC):
    @abc.abstractmethod
    def __call__(self, item):
        raise NotImplementedError


class Clip(NormalizerBase):
    def __init__(self, floor=-1, ceil=1):
        self.floor = floor
        self.ceil = ceil

    def __call__(self, item):
        return float(max(min(item, self.ceil), self.floor))


class Scale(NormalizerBase):
    def __init__(self, coefficient):
        self.coefficient = coefficient

    def __call__(self, item):
        return self.coefficient * item
