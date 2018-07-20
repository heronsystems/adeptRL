from torch.nn import BatchNorm1d, Conv2d, BatchNorm2d, init, Linear
from torch.nn import functional as F

from adept.modules import Identity
from ._base import Network


class FourConvCNN(Network):
    def __init__(self, nb_input, output_shape_dict, normalize):
        self.embedding_shape = 512
        super().__init__(self.embedding_shape, output_shape_dict)
        bias = not normalize
        self.conv1 = Conv2d(nb_input, 32, 3, stride=2, padding=1, bias=bias)
        self.conv2 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)
        self.conv3 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)
        self.conv4 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)
        self.linear = Linear(800, self.embedding_shape, bias=bias)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(32)
            self.bn4 = BatchNorm2d(32)
            self.bn_linear = BatchNorm1d(self.embedding_shape)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn_linear = Identity()

        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

    def forward(self, xs, internals):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = F.relu(self.bn4(self.conv4(xs)))

        xs = xs.view(xs.size(0), -1)

        xs = F.relu(self.bn_linear(self.linear(xs)))

        return self.network_head(xs, {})

    def new_internals(self, device):
        return {}
