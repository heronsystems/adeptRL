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
import math
from torch.nn import Linear, LSTMCell
import torch.nn.functional as F

from adept.modules import LSTMCellLayerNorm
from adept.networks.net1d.submodule_1d import SubModule1D


class IQNLSTM(SubModule1D):
    args = {
        'lstm_normalize': True,
        'lstm_nb_hidden': 5,
        'nb_embedding': 64
    }

    def __init__(self, input_shape, id, normalize, nb_hidden, nb_embedding):
        super().__init__(input_shape, id)
        self._nb_hidden = nb_hidden
        self._nb_embedding = nb_embedding
        self._arange_embedding = torch.arange(self._nb_embedding, dtype=torch.float, requires_grad=False)

        # learned linear weighting
        self.embed = Linear(self._nb_embedding, nb_hidden)
        # TODO: normalize linear weighting?
        if normalize:
            self.lstm = LSTMCellLayerNorm(
                input_shape[0], nb_hidden
            )
        else:
            self.lstm = LSTMCell(input_shape[0], nb_hidden)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    @classmethod
    def from_args(cls, args, input_shape, id):
        return cls(input_shape, id, args.lstm_normalize, args.lstm_nb_hidden, args.nb_embedding)

    @property
    def _output_shape(self):
        return (self._nb_hidden, )

    def _forward(self, xs, internals, num_samples=8, quantiles=None, **kwargs):
        # compute lstm 
        hxs = self.stacked_internals('hx', internals)
        cxs = self.stacked_internals('cx', internals)
        hxs, cxs = self.lstm(xs, (hxs, cxs))

        # compute embedding
        batch_size = xs.shape[0]
        if quantiles is None:
            quantiles = torch.FloatTensor(num_samples, batch_size).uniform_(0, 1).to(xs)
        quantiles_embedding = self._arange_embedding.expand(num_samples, batch_size, -1) * quantiles.unsqueeze(-1)
        embedding_input = torch.cos(math.pi * quantiles_embedding).to(xs)
        # sum over embedding dim
        embedding = F.relu(self.embed(embedding_input))
        # embedding shape [quantiles, batch, features]

        # combine with lstm output by broadcasting over samples
        combined = embedding * hxs.unsqueeze(0)

        return (
            combined, {
                'hx': list(torch.unbind(hxs, dim=0)),
                'cx': list(torch.unbind(cxs, dim=0))
            }
        )

    def _new_internals(self):
        return {
            'hx': torch.zeros(self._nb_hidden),
            'cx': torch.zeros(self._nb_hidden)
        }
