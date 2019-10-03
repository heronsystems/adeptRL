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
from __future__ import division

from torch import nn
from torch.nn import functional as F

from adept.modules import Identity

from .submodule_1d import SubModule1D


class Linear(SubModule1D):
    args = {
        'linear_normalize': True,
        'linear_nb_hidden': 512,
        'nb_layer': 3
    }

    def __init__(self, input_shape, id, normalize, nb_hidden, nb_layer):
        super().__init__(input_shape, id)
        self._nb_hidden = nb_hidden

        nb_input_channel = input_shape[0]

        bias = not normalize
        self.linear = nn.Linear(
            nb_input_channel, nb_hidden, bias=bias
        )
        self.linears = nn.ModuleList([
            nn.Linear(nb_hidden, nb_hidden, bias)
            for _ in range(nb_layer - 1)
        ])
        if normalize:
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(nb_hidden) for _ in range(nb_layer)
            ])
        else:
            self.norms = nn.ModuleList([
                Identity() for _ in range(nb_layer)
            ])

    @classmethod
    def from_args(cls, args, input_shape, id):
        return cls(
            input_shape, id, args.linear_nb_hidden, args.linear_nb_hidden,
            args.nb_layer
        )

    def _forward(self, xs, internals, **kwargs):
        xs = F.relu(self.norms[0](self.linear(xs)))
        for linear, norm in zip(self.linears, self.norms[1:]):
            xs = F.relu(norm(linear(xs)))
        return xs, {}

    def _new_internals(self):
        return {}

    @property
    def _output_shape(self):
        return (self._nb_hidden, )
