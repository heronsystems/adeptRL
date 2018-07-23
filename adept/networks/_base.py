import abc
from torch.nn import Module
import torch


class Network(Module, abc.ABC):
    def __init__(self, embedding_shape, output_shape_dict):
        super().__init__()
        self._network_head = NetworkHead(embedding_shape, output_shape_dict)

    @abc.abstractmethod
    def new_internals(self, device):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, input, internals):
        raise NotImplementedError

    @property
    def network_head(self):
        return self._network_head

    @staticmethod
    def stack_internals(internals):
        return {key: torch.stack(internal) for key, internal in internals.items()}

    @staticmethod
    def unstack_internals(stacked_internals):
        return {key: list(torch.unbind(stacked_internal)) for key, stacked_internal in stacked_internals.items()}


class NetworkHead(torch.nn.Module):
    def __init__(self, input_shape, head_dict):
        super().__init__()

        self.heads = torch.nn.ModuleList()
        # Must be sorted for mpi methods so that the creation order is deterministic
        for head_name in sorted(head_dict.keys()):
            head_size = head_dict[head_name]
            self.heads.add_module(head_name, torch.nn.Linear(input_shape, head_size))

    def forward(self, embedding, internals):
        stuff = {name: module(embedding) for name, module in self.heads.named_children()}, internals
        return stuff
