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

import torch
from torch.nn import functional as F, BatchNorm1d

from adept.modules import Identity

from adept.networks.net1d.submodule_1d import SubModule1D


class Linear(SubModule1D):
    args = {
        'linear_normalize': True,
        'linear_nb_hidden': 512
    }

    def __init__(self, input_shape, id, normalize, nb_hidden):
        super().__init__(input_shape, id)
        self._nb_hidden = nb_hidden


        bias = not normalize
        self.linear = torch.nn.Linear(
            nb_input_channel, self._nb_output_channel, bias=bias
        )
        if normalize:
            self.bn_linear = BatchNorm1d(nb_output_channel)
        else:
            self.bn_linear = Identity()

    @classmethod
    def from_args(cls, args, input_shape, id):
        return cls(
            input_shape, id, args.linear_nb_hidden, args.linear_nb_hidden
        )

    def _forward(self, xs, internals, **kwargs):
        xs = F.relu(self.bn_linear(self.linear(xs)))

        return xs, {}

    def _new_internals(self):
        return {}

    @property
    def _output_shape(self):
        return (self._nb_hidden, )
