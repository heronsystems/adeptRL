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
        'linear_normalize': 'bn',
        'linear_nb_hidden': 512,
        'nb_layer': 3
    }

    def __init__(self, input_shape, id, normalize, nb_hidden, nb_layer):
        super().__init__(input_shape, id)
        self._nb_hidden = nb_hidden

        nb_input_channel = input_shape[0]

        bias = not normalize
        self.linears = nn.ModuleList([
            nn.Linear(
                nb_input_channel if i == 0 else nb_hidden,
                nb_hidden,
                bias
            )
            for i in range(nb_layer - 1)
        ])
        if normalize == 'bn':
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(nb_hidden) for _ in range(nb_layer)
            ])
        elif normalize == 'gn':
            if nb_hidden % 16 != 0:
                raise Exception('linear_nb_hidden must be divisible by 16 for Group Norm')
            self.norms = nn.ModuleList([
                nn.GroupNorm(nb_hidden // 16, nb_hidden) for _ in range(nb_layer)
            ])
        else:
            self.norms = nn.ModuleList([
                Identity() for _ in range(nb_layer)
            ])

    @classmethod
    def from_args(cls, args, input_shape, id):
        return cls(
            input_shape, id, args.linear_normalize, args.linear_nb_hidden,
            args.nb_layer
        )

    def _forward(self, xs, internals, **kwargs):
        for linear, norm in zip(self.linears, self.norms):
            xs = F.relu(norm(linear(xs)))
        return xs, {}

    def _new_internals(self):
        return {}

    @property
    def _output_shape(self):
        return (self._nb_hidden, )
