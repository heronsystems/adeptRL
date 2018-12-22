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
import math

import torch
from torch import nn
from torch.nn import Parameter, functional as F


class GaussianLinear(nn.Module):
    def __init__(self, fan_in, nodes):
        super().__init__()
        self.mu = nn.Linear(fan_in, nodes)
        # init_tflearn_fc_(self.mu)
        self.std = nn.Linear(fan_in, nodes)
        # init_tflearn_fc_(self.std)

    def forward(self, x):
        mu = self.mu(x)
        if self.training:
            std = self.std(x)
            std = torch.exp(0.5 * std)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def get_parameter_names(self):
        return ['Mu_W', 'Mu_b', 'Std_W', 'Std_b']


class NoisyLinear(nn.Linear):
    """
    Reference implementation:
    https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py
    """

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(
            torch.Tensor(out_features, in_features)
        )  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.init_params()

    def init_params(self):
        limit = math.sqrt(3 / self.in_features)

        self.weight.data.uniform_(-limit, limit)
        self.bias.data.uniform_(-limit, limit)
        self.sigma_weight.data.fill_(self.sigma_init)
        self.sigma_bias.data.fill_(self.sigma_init)

    def forward(self, x, internals):
        if self.training:
            w = self.weight + self.sigma_weight * internals[0]
            b = self.bias + self.sigma_bias * internals[1]
        else:
            w = self.weight + self.sigma_weight
            b = self.bias + self.sigma_bias
        return F.linear(x, w, b)

    def batch_forward(self, x, internals, batch_size=None):
        print(
            'WARNING: calling forward multiple times is actually'
            'faster than this and takes less memory'
        )
        batch_size = batch_size if batch_size is not None else x.shape[0]
        x = x.unsqueeze(1)
        # internals come in as [[w, b], ...] reshape to [w, ...], [b, ...]
        eps_w, eps_b = zip(*internals)
        eps_w = torch.stack(eps_w)
        eps_b = torch.stack(eps_b)
        batch_w = self.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        ) + self.sigma_weight.unsqueeze(0).expand(batch_size, -1, -1)
        batch_w += eps_w
        # permute to b x m x p
        batch_w = batch_w.permute(0, 2, 1)
        batch_b = self.bias.expand(batch_size, -1) \
                  + self.sigma_bias.expand(batch_size, -1)
        batch_b += eps_b

        bmm = torch.bmm(x, batch_w).squeeze(1)

        return bmm + batch_b

    def reset(self, gpu=False, device=None):
        # sample new noise
        if not gpu:
            return (
                torch.randn(self.out_features, self.in_features).detach(),
                torch.randn(self.out_features).detach()
            )
        else:
            return (
                torch.randn(self.out_features,
                            self.in_features).cuda(device,
                                                   non_blocking=True).detach(),
                torch.randn(self.out_features).cuda(device,
                                                    non_blocking=True).detach()
            )

    def get_parameter_names(self):
        return ['W', 'b', 'sigma_W', 'sigma_b']
