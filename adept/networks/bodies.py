"""
Copyright (C) 2018 Heron Systems, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import division

import torch
from torch.nn import LSTMCell, functional as F, Linear, BatchNorm1d

from adept.modules import Identity, LSTMCellLayerNorm
from ._base import NetworkBody


class LSTMBody(NetworkBody):
    def __init__(self, nb_input_channel, nb_out_channel, normalize):
        super().__init__()
        self._nb_output_channel = nb_out_channel

        if normalize:
            self.lstm = LSTMCellLayerNorm(
                nb_input_channel, self._nb_output_channel
            )
        else:
            self.lstm = LSTMCell(nb_input_channel, self._nb_output_channel)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    @classmethod
    def from_args(cls, nb_input_channel, nb_out_channel, args):
        return cls(nb_input_channel, nb_out_channel, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs, internals):
        hxs = torch.stack(internals['hx'])
        cxs = torch.stack(internals['cx'])
        hxs, cxs = self.lstm(xs, (hxs, cxs))

        return (
            hxs, {
                'hx': list(torch.unbind(hxs, dim=0)),
                'cx': list(torch.unbind(cxs, dim=0))
            }
        )

    def new_internals(self, device):
        return {
            'hx': torch.zeros(self.nb_output_channel).to(device),
            'cx': torch.zeros(self.nb_output_channel).to(device)
        }


class LinearBody(NetworkBody):
    def __init__(self, nb_input_channel, nb_output_channel, normalize):
        super().__init__()
        self._nb_output_channel = nb_output_channel
        bias = not normalize

        self.linear = Linear(
            nb_input_channel, self._nb_output_channel, bias=bias
        )
        if normalize:
            self.bn_linear = BatchNorm1d(nb_output_channel)
        else:
            self.bn_linear = Identity()

    @classmethod
    def from_args(cls, nb_input_channel, nb_out_channel, args):
        return cls(nb_input_channel, nb_out_channel, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs, internals):
        xs = F.relu(self.bn_linear(self.linear(xs)))

        return xs, {}

    def new_internals(self, device):
        return {}


class Mnih2013Linear(NetworkBody):
    def __init__(self, nb_input_channel, _, normalize):
        super().__init__()
        self._nb_output_channel = 256
        bias = not normalize

        self.linear = Linear(
            nb_input_channel, self._nb_output_channel, bias=bias
        )
        if normalize:
            self.bn_linear = BatchNorm1d(self._nb_output_channel)
        else:
            self.bn_linear = Identity()

    @classmethod
    def from_args(cls, nb_input_channel, nb_out_channel, args):
        return cls(nb_input_channel, nb_out_channel, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs, internals):
        xs = F.relu(self.bn_linear(self.linear(xs)))

        return xs, {}

    def new_internals(self, device):
        return {}


class Mnih2013LSTM(NetworkBody):
    def __init__(self, nb_input_channel, nb_out_channel, normalize):
        super().__init__()
        self._nb_output_channel = 256
        self.linear = Linear(2592, self._nb_output_channel)

        if normalize:
            self.lstm = LSTMCellLayerNorm(
                self._nb_output_channel, self._nb_output_channel
            )  # hack for experiment
            self.bn_linear = BatchNorm1d(self._nb_output_channel)
        else:
            self.bn_linear = Identity()
            self.lstm = LSTMCell(
                self._nb_output_channel, self._nb_output_channel
            )
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    @classmethod
    def from_args(cls, nb_input_channel, nb_out_channel, args):
        return cls(nb_input_channel, nb_out_channel, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs, internals):
        xs = F.relu(self.bn_linear(self.linear(xs)))
        hxs = torch.stack(internals['hx'])
        cxs = torch.stack(internals['cx'])
        hxs, cxs = self.lstm(xs, (hxs, cxs))

        return (
            hxs, {
                'hx': list(torch.unbind(hxs, dim=0)),
                'cx': list(torch.unbind(cxs, dim=0))
            }
        )

    def new_internals(self, device):
        return {
            'hx': torch.zeros(self.nb_output_channel).to(device),
            'cx': torch.zeros(self.nb_output_channel).to(device)
        }
