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
from adept.network.base.submodule import SubModule
import abc


class SubModule1D(SubModule, metaclass=abc.ABCMeta):
    dim = 1

    def __init__(self, input_shape, id):
        super(SubModule1D, self).__init__(input_shape, id)

    def _to_1d_shape(self):
        return self._output_shape

    def _to_2d_shape(self):
        return self._output_shape + (1,)

    def _to_3d_shape(self):
        return self._output_shape + (1, 1)

    def _to_4d_shape(self):
        return self._output_shape + (1, 1, 1)
