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
from adept.networks.bodies import LinearBody, LSTMBody, Mnih2013Linear, Mnih2013LSTM
from adept.networks.discrete.networks import DiscreteIdentity
from adept.networks.vision.networks import (
    FourConv, FourConvSpatialAttention, FourConvLarger, Nature, Mnih2013,
    ResNet18, ResNet18V2, ResNet34, ResNet34V2, ResNet50, ResNet50V2,
    ResNet101V2, ResNet101, ResNet152V2, ResNet152
)

VISION_NETWORKS = {
    'FourConv': FourConv,
    'FourConvSpatialAttention': FourConvSpatialAttention,
    'FourConvLarger': FourConvLarger,
    'Nature': Nature,
    'Mnih2013': Mnih2013,
    'ResNet18': ResNet18,
    'ResNet18V2': ResNet18V2,
    'ResNet34': ResNet34,
    'ResNet34V2': ResNet34V2,
    'ResNet50': ResNet50,
    'ResNet50V2': ResNet50V2,
    'ResNet101': ResNet101,
    'ResNet101V2': ResNet101V2,
    'ResNet152': ResNet152,
    'ResNet152V2': ResNet152V2
}
DISCRETE_NETWORKS = {'Identity': DiscreteIdentity}
NETWORK_BODIES = {
    'Linear': LinearBody,
    'LSTM': LSTMBody,
    'Mnih2013Linear': Mnih2013Linear,
    'Mnih2013LSTM': Mnih2013LSTM
}
