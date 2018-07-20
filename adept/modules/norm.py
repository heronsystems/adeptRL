import torch


class Identity(torch.nn.Module):
    def forward(self, x):
        return x