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
        """Output shape casted to requested dimension

        Parameters
        ----------
        dim : int, optional
            Desired dimensionality, defaults to 2

        Returns
        -------
        tuple[int]
            Output shape
        """
        if dim == 1:
            f, s = self._output_shape
            return (f * s,)
        elif dim == 2 or dim is None:
            return self._output_shape
        elif dim == 3:
            f, s = self._output_shape
            return (f * s, 1, 1)
        elif dim == 4:
            f, s = self._output_shape
            return (f, s, 1, 1)
        else:
            raise ValueError("Invalid dim: {}".format(dim))

    def _to_1d(self, submodule_output):
        """Convert to Batch + 1D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor

        Returns
        -------
        torch.Tensor
            Batch + 1D Tensor
        """
        n, f, s = submodule_output.size()
        return submodule_output.view(n, f * s)

    def _to_2d(self, submodule_output):
        """Convert to Batch + 2D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor (B, S, F)

        Returns
        -------
        torch.Tensor
            Batch + 2D Tensor (B, S, F)
        """
        return submodule_output

    def _to_3d(self, submodule_output):
        """Convert to Batch + 3D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor (B, S, F)

        Returns
        -------
        torch.Tensor
            Batch + 3D Tensor
        """
        n, f, s = submodule_output.size()
        return submodule_output.view(n, f * s, 1, 1)

    def _to_4d(self, submodule_output):
        """Convert to Batch + 4D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor (B, S, F)

        Returns
        -------
        torch.Tensor
            Batch + 4D Tensor (B, F, S, H, W)
        """
        n, f, s = submodule_output.size()
        return submodule_output.view(n, f, s, 1, 1)
