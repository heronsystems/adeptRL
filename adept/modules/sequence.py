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
from torch.nn import Module, Linear, LayerNorm


class LSTMCellLayerNorm(Module):
    """
    A lstm cell that layer norms the cell state
    https://github.com/seba-1511/lstms.pth/blob/master/lstms/lstm.py for reference.
    Original License Apache 2.0
    """

    def __init__(self, input_size, hidden_size, bias=True, forget_bias=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih = Linear(input_size, 4 * hidden_size, bias=bias)
        self.hh = Linear(hidden_size, 4 * hidden_size, bias=bias)

        if bias:
            self.ih.bias.data.fill_(0)
            self.hh.bias.data.fill_(0)
            # forget bias init
            self.ih.bias.data[hidden_size : hidden_size * 2].fill_(forget_bias)
            self.hh.bias.data[hidden_size : hidden_size * 2].fill_(forget_bias)

        self.ln_cell = LayerNorm(hidden_size)

    def forward(self, x, hidden):
        """
        LSTM Cell that layer normalizes the cell state.
        :param x: Tensor{B, C}
        :param hidden: A Tuple[Tensor{B, C}, Tensor{B, C}] of (previous output, cell state)
        :return:
        """
        h, c = hidden

        # Linear mappings
        i2h = self.ih(x)
        h2h = self.hh(h)
        preact = i2h + h2h

        # activations
        gates = preact[:, : 3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size :].tanh()
        i_t = gates[:, : self.hidden_size]
        f_t = gates[:, self.hidden_size : 2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size :]

        # cell computations
        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        c_t = self.ln_cell(c_t)
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t

    def to(self, device):
        super().to(device)
        self.ih = self.ih.to(device)
        self.hh = self.hh.to(device)
        return self
