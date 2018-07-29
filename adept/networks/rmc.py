import torch
from torch.nn import Conv2d, Linear, BatchNorm2d, BatchNorm1d, functional as F

from adept.modules import RMCCell, Identity
from ._base import NetworkInterface


# TODO
class RMC(NetworkInterface):
    """
    Relational Memory Core
    https://arxiv.org/pdf/1806.01822.pdf
    """
    def __init__(self, nb_in_chan, output_shape_dict, normalize):
        self.embedding_size = 512
        super(RMC, self).__init__(self.embedding_size, output_shape_dict)
        bias = not normalize
        self.conv1 = Conv2d(nb_in_chan, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.conv2 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.conv3 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.attention = RMCCell(100, 100, 34)
        self.conv4 = Conv2d(34, 8, kernel_size=3, stride=1, padding=1, bias=bias)
        # BATCH x 8 x 10 x 10
        self.linear = Linear(800, 512, bias=bias)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(32)
            self.bn4 = BatchNorm2d(8)
            self.bn_linear = BatchNorm1d(512)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn_linear = Identity()

    def forward(self, input, prev_memories):
        """
        :param input: Tensor{B, C, H, W}
        :param prev_memories: Tuple{B}[Tensor{C}]
        :return:
        """

        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        h = x.size(2)
        w = x.size(3)
        xs_chan = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(input.size(0), 1, w, w).to(input.device)
        ys_chan = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(input.size(0), 1, h, h).to(input.device)
        x = torch.cat([x, xs_chan, ys_chan], dim=1)

        # need to transpose because attention expects attention dim before channel dim
        x = x.view(x.size(0), x.size(1), h * w).transpose(1, 2)
        prev_memories = torch.stack(prev_memories)
        x = next_memories = self.attention(x.contiguous(), prev_memories)
        # need to undo the transpose before output
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), h, w)

        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_linear(self.linear(x)))
        return x, list(torch.unbind(next_memories, 0))

    def new_internals(self, device):
        pass
