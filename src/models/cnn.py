import torch
from torch.nn import Conv2d, Linear, BatchNorm2d, BatchNorm1d
from torch.nn import functional as F
from src.models._base import Identity


class SimpleCNN(torch.nn.Module):
    def __init__(self, nb_input, nb_action, normalize):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2d(nb_input, 32, 8, stride=4)
        self.conv2 = Conv2d(32, 64, 4, stride=2)
        self.conv3 = Conv2d(64, 32, 3, stride=1)
        self.linear = Linear(1152, 512)
        self.actor_linear = Linear(512, nb_action)
        self.critic_linear = Linear(512, 1)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(64)
            self.bn3 = BatchNorm2d(32)
            self.bn_linear = BatchNorm1d(512)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn_linear = Identity()

        # default init > this:
        # self.apply(weights_init)
        # relu_gain = nn.init.calculate_gain('relu')
        # self.conv1.weight.data.mul_(relu_gain)
        # self.conv2.weight.data.mul_(relu_gain)
        # self.conv3.weight.data.mul_(relu_gain)
        # self.linear.weight.data = norm_col_init(self.linear.weight.data, 0.01)
        # self.linear.bias.data.fill_(0)
        # self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_linear(self.linear(x)))
        return self.critic_linear(x), self.actor_linear(x)
