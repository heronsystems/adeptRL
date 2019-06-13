# Copyright (C) 2019 Heron Systems, Inc.
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

from adept.modules import ConvLSTMCellLayerNorm
from adept.networks.net3d.submodule_3d import SubModule3D


class ConvLSTM(SubModule3D):
    args = {
        'convlstm_normalize': True,
        'convlstm_nb_chan': 32,
        'convlstm_kernel_size': 3,
    }

    def __init__(self, input_shape, id, normalize, nb_chan, kernel_size):
        super().__init__(input_shape, id)
        self._nb_hidden = nb_chan
        self._kernel_size = kernel_size

        if normalize:
            self.lstm = ConvLSTMCellLayerNorm(
                input_shape, nb_chan, kernel_size
            )
        else:
            raise NotImplementedError('ConvLSTM without layer norm not supported')

    @classmethod
    def from_args(cls, args, input_shape, id):
        return cls(input_shape, id, args.convlstm_normalize, args.convlstm_nb_chan, args.convlstm_kernel_size)

    @property
    def _output_shape(self):
        output_dim = calc_output_dim(self._input_shape[1], 3, 1, 0, 1)
        return self._nb_hidden, output_dim, output_dim

    def _forward(self, xs, internals, **kwargs):
        hxs = self.stacked_internals('hx', internals)
        cxs = self.stacked_internals('cx', internals)
        hxs, cxs = self.lstm(xs, (hxs, cxs))

        return (
            hxs, {
                'hx': list(torch.unbind(hxs, dim=0)),
                'cx': list(torch.unbind(cxs, dim=0))
            }
        )

    def _new_internals(self):
        return {
            'hx': torch.zeros(*self._output_shape),
            'cx': torch.zeros(*self._output_shape)
        }


def calc_output_dim(dim_size, kernel_size, stride, padding, dilation):
    numerator = dim_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return numerator // stride + 1

