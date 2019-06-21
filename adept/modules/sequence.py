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

    Modified to follow tensorflow implementation here:
    https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L2453
    """

    def __init__(self, input_size, hidden_size, forget_bias=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.ih = Linear(input_size + hidden_size, 4 * hidden_size, bias=False)

        self.ln_g_t = LayerNorm(hidden_size)
        self.ln_i_t = LayerNorm(hidden_size)
        self.ln_f_t = LayerNorm(hidden_size)
        self.ln_o_t = LayerNorm(hidden_size)
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
        preact = self.ih(torch.cat([x, h], dim=-1))

        # activations
        it, ft, ot, gt = torch.chunk(preact, 4, dim=-1)
        i_t = self.ln_i_t(it).sigmoid_()
        f_t = self.ln_f_t(ft)
        # forget bias
        if self.forget_bias != 0:
            f_t += self.forget_bias
            f_t.sigmoid_()
        o_t = self.ln_o_t(ot).sigmoid_()
        g_t = self.ln_g_t(gt).tanh_()

        # cell computations cannot be inplace 
        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        c_t = self.ln_cell(c_t)
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t


class JANETCellLayerNorm(Module):
    """
    A lstm cell that only has forget gates
    "The unreasonable effectiveness of the forget gate"

    https://arxiv.org/pdf/1804.04849.pdf
    """

    def __init__(self, input_size, hidden_size, tmax=20, forget_bias=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.ih = Linear(input_size + hidden_size, 2 * hidden_size, bias=False)

        self.ln_s_t = LayerNorm(hidden_size)
        # chrono init log(uniform(1, Tmax - 1))
        bias_forget = torch.log(1 + (torch.rand_like(self.ln_s_t.bias.data) * tmax))
        self.ln_s_t.bias.data = bias_forget
        self.ln_f_t = LayerNorm(hidden_size)

    def forward(self, x, hidden):
        """
        LSTM Cell that layer normalizes the cell state.
        :param x: Tensor{B, C}
        :param hidden: A Tuple[Tensor{B, C}, Tensor{B, C}] of (previous output, cell state)
        :return:
        """
        h, c = hidden

        # Linear mappings
        preact = self.ih(torch.cat([x, h], dim=-1))

        # activations using notation from the paper
        st, ft = torch.chunk(preact, 2, dim=-1)

        s_t = self.ln_s_t(st)
        f_t = self.ln_f_t(ft)
        c_tilda_t = f_t.tanh_()
        c_t = s_t.sigmoid() * c + (1 - (s_t - self.forget_bias).sigmoid()) * c_tilda_t

        return c_t, c_t

