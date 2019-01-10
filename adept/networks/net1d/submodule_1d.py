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
from adept.networks._base import SubModule
import abc


class SubModule1D(SubModule, metaclass=abc.ABCMeta):
    def __init__(self, input_shape):
        super(SubModule1D, self).__init__(input_shape)
        assert self.dim == 1

    def output_shape(self, input_shape, dim=None):
        if dim is None or dim == 1:
            return self._output_shape(input_shape)
        elif dim == 2:
            return self._output_shape(input_shape).view(-1, 1)
        elif dim == 3:
            return self._output_shape(input_shape).view(-1, 1, 1)
        elif dim == 4:
            return self._output_shape(input_shape).view(-1, 1, 1, 1)
        else:
            raise ValueError('Invalid dim: {}'.format(dim))

    def _to_1d(self, result):
        """
        :param result: torch.Tensor (1D)
        :return: torch.Tensor (1D)
        """
        return result

    def _to_2d(self, result):
        """
        :param result: torch.Tensor (1D)
        :return: torch.Tensor (2D)
        """
        n, f = result.size()
        return result.view(n, f, 1)

    def _to_3d(self, result):
        """
        :param result: torch.Tensor (1D)
        :return: torch.Tensor (3D)
        """
        n, f = result.size()
        return result.view(n, f, 1, 1)

    def _to_4d(self, result):
        """
        :param result: torch.Tensor (1D)
        :return: torch.Tensor (4D)
        """
        n, f = result.size()
        return result.view(n, f, 1, 1, 1)
