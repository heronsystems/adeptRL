import math

import torch
from src.models._base import Identity
from torch.nn import Conv2d, Linear, Softmax, init
from torch.nn import functional as F
from src.utils import weights_init, norm_col_init
from torch.nn import BatchNorm2d, BatchNorm1d


class AttentionCNN(torch.nn.Module):
    def __init__(self, nb_in_chan, nb_action, nb_head, normalize):
        super(AttentionCNN, self).__init__()
        self.normalize = normalize
        self.conv1 = Conv2d(nb_in_chan, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.attention = RelationalMHDPA(20 * 20, 34, nb_head)

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

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     self.apply(weights_init)
    #     gain = init.calculate_gain('relu')
    #     self.conv1.weight.data.mul_(gain)
    #     self.conv2.weight.data.mul_(gain)
    #     self.conv3.weight.data.mul_(gain)
    #
    #     self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
    #     self.actor_linear.bias.data.fill_(0)
    #     self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
    #     self.critic_linear.bias.data.fill_(0)

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
        # need to undo the transpose before output
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), h, w)

        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_linear(self.linear(x)))
        return self.critic_linear(x), self.actor_linear(x)


class RelationalMHDPA(torch.nn.Module):
    """
    Multi-head dot product attention.

    Adapted from:
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_py.py

    Reference implementation (Tensorflow):
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L2674
    """
    def __init__(self, seq_len, nb_channel, nb_head, scale=False):
        super(RelationalMHDPA, self).__init__()
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert nb_channel % nb_head == 0
        self.register_buffer('b', torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))
        self.nb_head = nb_head
        self.split_size = nb_channel
        self.scale = scale
        self.projection = Linear(nb_channel, nb_channel * 3)
        self.mlp = Linear(nb_channel, nb_channel)
        # self.layer_norm = LayerNorm([seq_len, nb_channel])

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        w = Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        # keep dims, but expand the last dim to be [head, chan // head]
        new_x_shape = x.size()[:-1] + (self.nb_head, x.size(-1) // self.nb_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            # batch, head, channel, attend
            return x.permute(0, 2, 3, 1)
        else:
            # batch, head, attend, channel
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        """
        :param x: A tensor with a shape of [batch, seq_len, nb_channel]
        :return: A tensor with a shape of [batch, seq_len, nb_channel]
        """
        size_out = x.size()[:-1] + (self.split_size * 3,)
        x = self.projection(x.view(-1, x.size(-1)))
        x = x.view(*size_out)

        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        a = self._attn(query, key, value)
        e = self.merge_heads(a)

        return self.mlp(e)
