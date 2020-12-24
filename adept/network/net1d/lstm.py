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
from torch.nn import LSTMCell

from adept.modules import LSTMCellLayerNorm
from .submodule_1d import SubModule1D


class LSTM(SubModule1D):
    args = {"lstm_normalize": True, "lstm_nb_hidden": 512}

    def __init__(self, input_shape, id, normalize, nb_hidden):
        super().__init__(input_shape, id)
        self._nb_hidden = nb_hidden

        if normalize:
            self.lstm = LSTMCellLayerNorm(input_shape[0], nb_hidden)
        else:
            self.lstm = LSTMCell(input_shape[0], nb_hidden)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    @classmethod
    def from_args(cls, args, input_shape, id):
        return cls(input_shape, id, args.lstm_normalize, args.lstm_nb_hidden)

    @property
    def _output_shape(self):
        return (self._nb_hidden,)

    def _forward(self, xs, internals, **kwargs):
        hxs = self.stacked_internals("hx", internals)
        cxs = self.stacked_internals("cx", internals)
        hxs, cxs = hxs.to(xs.device), cxs.to(xs.device)
        hxs, cxs = self.lstm(xs, (hxs, cxs))

        return (
            hxs,
            {
                "hx": list(torch.unbind(hxs, dim=0)),
                "cx": list(torch.unbind(cxs, dim=0)),
            },
        )

    def _new_internals(self, device=None):
        return {
            "hx": torch.zeros(self._nb_hidden).to(device),
            "cx": torch.zeros(self._nb_hidden).to(device),
        }
