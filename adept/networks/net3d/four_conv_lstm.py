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
from torch.nn import Conv2d, BatchNorm2d, init, functional as F

from adept.modules import Identity, ConvLSTMCellLayerNorm, calc_conv_output_dim
from adept.networks.net3d.submodule_3d import SubModule3D


class FourConvLSTM(SubModule3D):
    args = {
        'num_chan': 32
    }
    def __init__(self, in_shape, id, num_chan):
        super().__init__(in_shape, id)
        self._in_shape = in_shape
        self._out_shape = None
        self._num_chan = num_chan
        self.conv_out_shapes = []
        self.conv1 = ConvLSTMCellLayerNorm(in_shape, num_chan, 7, stride=2, padding=1)

        o_dim = calc_conv_output_dim(self._in_shape[1], 7, 2, 1, 1)
        self.conv_out_shapes.append((num_chan, o_dim, o_dim))
        self.conv2 = ConvLSTMCellLayerNorm(self.conv_out_shapes[-1], num_chan, 3, stride=2, padding=1)

        o_dim = calc_conv_output_dim(o_dim, 3, 2, 1, 1)
        self.conv_out_shapes.append((num_chan, o_dim, o_dim))
        self.conv3 = ConvLSTMCellLayerNorm(self.conv_out_shapes[-1], num_chan, 3, stride=2, padding=1)

        o_dim = calc_conv_output_dim(o_dim, 3, 2, 1, 1)
        self.conv_out_shapes.append((num_chan, o_dim, o_dim))
        self.conv4 = ConvLSTMCellLayerNorm(self.conv_out_shapes[-1], num_chan, 3, stride=2, padding=1)

        o_dim = calc_conv_output_dim(o_dim, 3, 2, 1, 1)
        self.conv_out_shapes.append((num_chan, o_dim, o_dim))

    @classmethod
    def from_args(cls, args, in_shape, id):
        return cls(in_shape, id, args.num_chan)

    @property
    def _output_shape(self):
        # For 84x84, (32, 5, 5)
        if self._out_shape is None:
            output_dim = calc_conv_output_dim(self._in_shape[1], 7, 2, 1, 1)
            output_dim = calc_conv_output_dim(output_dim, 3, 2, 1, 1)
            output_dim = calc_conv_output_dim(output_dim, 3, 2, 1, 1)
            output_dim = calc_conv_output_dim(output_dim, 3, 2, 1, 1)
            self._out_shape = self._num_chan, output_dim, output_dim
        return self._out_shape

    def _forward(self, xs, internals, **kwargs):
        new_internals = {}
        for c_ind, c in enumerate([self.conv1, self.conv2, self.conv3, self.conv4]):
            h_key, c_key = 'hx{}'.format(c_ind), 'cx{}'.format(c_ind)
            hxs = self.stacked_internals(h_key, internals)
            cxs = self.stacked_internals(c_key, internals)
            xs, new_cxs = c(xs, (hxs, cxs))
            new_internals[h_key] = list(torch.unbind(xs, dim=0))
            new_internals[c_key] = list(torch.unbind(new_cxs, dim=0))
        return xs, new_internals

    def _new_internals(self):
        new_internals = {}
        for c_ind, conv_shape in enumerate(self.conv_out_shapes):
            h_key, c_key = 'hx{}'.format(c_ind), 'cx{}'.format(c_ind)
            new_internals[h_key] = torch.zeros(*conv_shape)
            new_internals[c_key] = torch.zeros(*conv_shape)
        return new_internals

