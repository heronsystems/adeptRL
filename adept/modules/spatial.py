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
from torch import nn as nn
from torch.nn import functional as F


class Residual2DPreact(nn.Module):
    def __init__(self, nb_in_chan, nb_out_chan, stride=1):
        super(Residual2DPreact, self).__init__()

        self.nb_in_chan = nb_in_chan
        self.nb_out_chan = nb_out_chan
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(nb_in_chan)
        self.conv1 = nn.Conv2d(
            nb_in_chan, nb_out_chan, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(nb_out_chan)
        self.conv2 = nn.Conv2d(
            nb_out_chan, nb_out_chan, 3, stride=1, padding=1, bias=False
        )

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)

        self.do_projection = self.nb_in_chan != self.nb_out_chan or self.stride > 1
        if self.do_projection:
            self.projection = nn.Conv2d(
                nb_in_chan, nb_out_chan, 3, stride=stride, padding=1
            )
            self.projection.weight.data.mul_(relu_gain)

    def forward(self, x):
        first = F.relu(self.bn1(x))
        if self.do_projection:
            projection = self.projection(first)
        else:
            projection = x
        x = self.conv1(first)
        x = self.conv2(F.relu(self.bn2(x)))
        return x + projection
