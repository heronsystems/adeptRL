import torch
from torch.nn import BatchNorm2d, BatchNorm1d
from torch.nn import Conv2d, Linear
from torch.nn import functional as F

from src.models._base import Identity, MultiHeadSelfAttention


class AttentionCNN(torch.nn.Module):
    def __init__(self, nb_in_chan, nb_action, nb_head, normalize):
        super(AttentionCNN, self).__init__()
        self.normalize = normalize
        self.conv1 = Conv2d(nb_in_chan, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.attention = MultiHeadSelfAttention(20 * 20, 34, 34, nb_head)
        self.mlp = Linear(34, 34)

        self.conv3 = Conv2d(34, 8, kernel_size=3, stride=2, padding=1)
        # BATCH x 8 x 10 x 10
        self.linear = Linear(800, 512)
        self.actor_linear = Linear(512, nb_action)
        self.critic_linear = Linear(512, 1)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(8)
            self.bn_linear = BatchNorm1d(512)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn_linear = Identity()

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.ln(x)

        xs_chan = torch.linspace(-1, 1, 20).view(1, 1, 1, 20).expand(input.size(0), 1, 20, 20).to(input.device)
        ys_chan = torch.linspace(-1, 1, 20).view(1, 1, 20, 1).expand(input.size(0), 1, 20, 20).to(input.device)
        x = torch.cat([x, xs_chan, ys_chan], dim=1)
        h = x.size(2)
        w = x.size(3)
        # need to transpose because attention expects attention dim before channel dim
        x = x.view(x.size(0), x.size(1), h * w).transpose(1, 2)

        x = self.attention(x.contiguous())
        x = F.relu(self.mlp(x))
        # need to undo the transpose before output
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), h, w)

        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_linear(self.linear(x)))
        return self.critic_linear(x), self.actor_linear(x)
