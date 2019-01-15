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
from adept.networks.submodule import SubModule
import abc


class SubModule1D(SubModule, metaclass=abc.ABCMeta):
    dim = 1

    def __init__(self, input_shape):
        super(SubModule1D, self).__init__(input_shape, id)

    def output_shape(self, dim=None):
        if dim is None or dim == 1:
            return self._output_shape
        elif dim == 2:
            return self._output_shape.view(-1, 1)
        elif dim == 3:
            return self._output_shape.view(-1, 1, 1)
        elif dim == 4:
            return self._output_shape.view(-1, 1, 1, 1)
        else:
            raise ValueError('Invalid dim: {}'.format(dim))

    def _to_1d(self, submodule_output):
        """
        :param submodule_output: torch.Tensor (1D)
        :return: torch.Tensor (1D)
        """
        return submodule_output

    def _to_2d(self, submodule_output):
        """
        :param submodule_output: torch.Tensor (1D)
        :return: torch.Tensor (2D)
        """
        n, f = submodule_output.size()
        return submodule_output.view(n, f, 1)

    def _to_3d(self, submodule_output):
        """
        :param submodule_output: torch.Tensor (1D)
        :return: torch.Tensor (3D)
        """
        n, f = submodule_output.size()
        return submodule_output.view(n, f, 1, 1)

    def _to_4d(self, submodule_output):
        """
        :param submodule_output: torch.Tensor (1D)
        :return: torch.Tensor (4D)
        """
        n, f = submodule_output.size()
        return submodule_output.view(n, f, 1, 1, 1)
