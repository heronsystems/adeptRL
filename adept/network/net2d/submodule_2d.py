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
from ..base.submodule import SubModule
import abc
import math


class SubModule2D(SubModule, metaclass=abc.ABCMeta):
    dim = 2

    def __init__(self, input_shape, id):
        super(SubModule2D, self).__init__(input_shape, id)

    def output_shape(self, dim=None):
        if dim == 1:
            f, l = self._output_shape
            return (f * l, )
        elif dim == 2 or dim is None:
            return self._output_shape
        elif dim == 3:
            f, l = self._output_shape
            return (f, math.sqrt(l), math.sqrt(l))
        elif dim == 4:
            f, l = self._output_shape
            return (f, l, 1, 1)
        else:
            raise ValueError('Invalid dim: {}'.format(dim))

    def _to_1d(self, submodule_output):
        """
        :param submodule_output: torch.Tensor (Batch + 2D)
        :return: torch.Tensor (Batch + 1D)
        """
        n, f, l = submodule_output.size()
        return submodule_output.view(n, f * l)

    def _to_2d(self, submodule_output):
        """
        :param submodule_output: torch.Tensor (Batch + 2D)
        :return: torch.Tensor (Batch + 2D)
        """
        return submodule_output

    def _to_3d(self, submodule_output):
        """
        :param submodule_output: torch.Tensor (Batch + 2D)
        :return: torch.Tensor (Batch + 3D)
        """
        n, f, l = submodule_output.size()
        return submodule_output.view(n, f, int(math.sqrt(l)), int(math.sqrt(l)))

    def _to_4d(self, submodule_output):
        """
        :param submodule_output: torch.Tensor (Batch + 2D)
        :return: torch.Tensor (Batch + 4D)
        """
        n, f, l = submodule_output.size()
        return submodule_output.view(n, f, l, 1, 1)
