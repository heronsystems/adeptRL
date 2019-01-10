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
import abc

import torch
from torch.nn import Conv2d, BatchNorm2d, init, functional as F, Linear

from adept.modules import MultiHeadSelfAttention, Identity
from ._resnets import (
    resnet18, resnet18v2, resnet34, resnet34v2, resnet50v2, resnet101,
    resnet101v2, resnet152, resnet152v2
)
from .._base import SubModule


class Nature(SubModule):
    """
    https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    """

    def __init__(self, in_shape, normalize):
        super().__init__()
        bias = not normalize
        self._nb_output_channel = 3136
        self.conv1 = Conv2d(in_shape[0], 32, 8, stride=4, padding=0, bias=bias)
        self.conv2 = Conv2d(32, 64, 4, stride=2, padding=0, bias=bias)
        self.conv3 = Conv2d(64, 64, 3, stride=1, padding=0, bias=bias)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(64)
            self.bn3 = BatchNorm2d(64)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))

        xs = xs.view(xs.size(0), -1)

        return xs


class Mnih2013(SubModule):
    def __init__(self, in_shape, normalize):
        super().__init__()
        bias = not normalize
        self._nb_output_channel = 2592
        self.conv1 = Conv2d(in_shape[0], 16, 8, stride=4, padding=0, bias=bias)
        self.conv2 = Conv2d(16, 32, 4, stride=2, padding=0, bias=bias)

        if normalize:
            self.bn1 = BatchNorm2d(16)
            self.bn2 = BatchNorm2d(32)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))

        xs = xs.view(xs.size(0), -1)

        return xs


class FourConv(SubModule):
    def __init__(self, in_shape, normalize):
        super().__init__()
        bias = not normalize
        self._nb_output_channel = 800
        self.conv1 = Conv2d(in_shape[0], 32, 7, stride=2, padding=1, bias=bias)
        self.conv2 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)
        self.conv3 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)
        self.conv4 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(32)
            self.bn4 = BatchNorm2d(32)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn4 = Identity()

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = F.relu(self.bn4(self.conv4(xs)))

        xs = xs.view(xs.size(0), -1)

        return xs


class FourConvSpatialAttention(SubModule):
    """
    https://arxiv.org/pdf/1806.01830.pdf
    """

    def __init__(self, in_shape, nb_head, normalize):
        self._nb_output_channel = 800
        super().__init__()
        self.normalize = normalize
        bias = not normalize
        self.conv1 = Conv2d(
            in_shape[0], 32, kernel_size=3, stride=2, padding=1, bias=bias
        )
        self.conv2 = Conv2d(
            32, 32, kernel_size=3, stride=2, padding=1, bias=bias
        )

        self.attention = MultiHeadSelfAttention(20 * 20, 34, 34, nb_head)
        self.mlp = Linear(34, 34)

        self.conv3 = Conv2d(
            34, 32, kernel_size=3, stride=2, padding=1, bias=bias
        )
        self.conv4 = Conv2d(
            32, 32, kernel_size=3, stride=2, padding=1, bias=bias
        )

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(32)
            self.bn4 = BatchNorm2d(32)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn4 = Identity()

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.nb_head, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, input, internals):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))

        xs_chan = torch.linspace(-1, 1, 20)\
            .view(1, 1, 1, 20)\
            .expand(input.size(0), 1, 20, 20)\
            .to(input.device)
        ys_chan = torch.linspace(-1, 1, 20)\
            .view(1, 1, 20, 1)\
            .expand(input.size(0), 1, 20, 20)\
            .to(input.device)
        x = torch.cat([x, xs_chan, ys_chan], dim=1)
        h = x.size(2)
        w = x.size(3)
        # need to transpose because attention
        # expects attention dim before channel dim
        x = x.view(x.size(0), x.size(1), h * w).transpose(1, 2)

        x = self.attention(x.contiguous())
        x = F.relu(self.mlp(x))
        # need to undo the transpose before output
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), h, w)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        return x


class FourConvLarger(SubModule):
    def __init__(self, in_shape, normalize):
        super().__init__()
        bias = not normalize
        self._nb_output_channel = 3200
        self.conv1 = Conv2d(in_shape[0], 32, 7, stride=2, padding=1, bias=bias)
        self.conv2 = Conv2d(32, 64, 3, stride=2, padding=1, bias=bias)
        self.conv3 = Conv2d(64, 64, 3, stride=2, padding=1, bias=bias)
        self.conv4 = Conv2d(64, 128, 3, stride=2, padding=1, bias=bias)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(64)
            self.bn3 = BatchNorm2d(64)
            self.bn4 = BatchNorm2d(128)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn4 = Identity()

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = F.relu(self.bn4(self.conv4(xs)))

        xs = xs.view(xs.size(0), -1)

        return xs


class BaseResNet(SubModule, metaclass=abc.ABCMeta):
    def __init__(self, in_shape, normalize):
        super().__init__()
        bias = not normalize
        self.conv1 = Conv2d(
            in_shape[0], 64, 7, stride=2, padding=1, bias=bias
        )  # 40x40
        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)

        if normalize:
            self.bn1 = BatchNorm2d(64)
        else:
            self.bn1 = Identity()

    @property
    @abc.abstractmethod
    def resnet(self):
        raise NotImplementedError()

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.normalize)

    @property
    def nb_output_channel(self):
        return self.resnet.nb_output_channel

    def forward(self, xs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = self.resnet(xs)
        xs = xs.view(xs.size(0), -1)
        return xs


class ResNet18(BaseResNet):
    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet18()

    @property
    def resnet(self):
        return self._resnet


class ResNet18V2(BaseResNet):
    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet18v2()

    @property
    def resnet(self):
        return self._resnet


class ResNet34(BaseResNet):
    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet34()

    @property
    def resnet(self):
        return self._resnet


class ResNet34V2(BaseResNet):
    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet34v2()

    @property
    def resnet(self):
        return self._resnet


class ResNet50(BaseResNet):
    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet50v2()

    @property
    def resnet(self):
        return self._resnet


class ResNet50V2(BaseResNet):
    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet50v2()

    @property
    def resnet(self):
        return self._resnet


class ResNet101(BaseResNet):
    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet101()

    @property
    def resnet(self):
        return self._resnet


class ResNet101V2(BaseResNet):
    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet101v2()

    @property
    def resnet(self):
        return self._resnet


class ResNet152(BaseResNet):
    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet152()

    @property
    def resnet(self):
        return self._resnet


class ResNet152V2(BaseResNet):
    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet152v2()

    @property
    def resnet(self):
        return self._resnet
