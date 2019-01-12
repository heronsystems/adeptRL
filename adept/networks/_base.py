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
from adept.utils.requires_args import RequiresArgs
from collections import ChainMap


class BaseNetwork(torch.nn.Module):
    @classmethod
    @abc.abstractmethod
    def from_args(
        cls,
        args,
        observation_space,
        headname_to_output_shape
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def new_internals(self, device):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, obsname_to_obs, internals):
        raise NotImplementedError


class NetworkModule(BaseNetwork, RequiresArgs, metaclass=abc.ABCMeta):
    pass


class NetworkHead(torch.nn.Module):
    def __init__(self, nb_channel, head_dict):
        super().__init__()

        self.heads = torch.nn.ModuleList()
        # Must be sorted for mpi methods so that the creation order is
        # deterministic
        for head_name in sorted(head_dict.keys()):
            head_size = head_dict[head_name]
            self.heads.add_module(
                head_name, torch.nn.Linear(nb_channel, head_size)
            )

    def forward(self, embedding, internals):
        return {
            name: module(embedding)
            for name, module in self.heads.named_children()
        }, internals


class ModularNetwork(BaseNetwork, metaclass=abc.ABCMeta):
    def __init__(
        self,
        obs_key_to_input_net,
        body_submodule,
        head_submodules,
        output_space
    ):
        """
        :param obs_key_to_input_net: Dict[ObsKey, SubModule]
        :param body_submodule: SubModule
        :param head_submodules: List[SubModule]
        :param output_space: Dict[OutputKey, Shape]
        """
        # modular network doesn't have networks for unused inputs
        super().__init__()

        # Input Nets
        self._obs_keys = list(obs_key_to_input_net.keys())
        self.obs_key_to_input_submod = torch.nn.ModuleDict(
            [(key, net) for key, net in obs_key_to_input_net.items()]
        )

        # Body
        self.body_submodule = body_submodule

        # Heads
        self.dim_to_head = torch.nn.ModuleDict(
            [(submod.dim, submod) for submod in head_submodules]
        )

        # Outputs
        self._output_keys = list(output_space.keys())
        self.output_space = output_space
        outputs = []
        for output_name, shape in self.output_space.items():
            dim = len(shape)
            if dim == 1:
                layer = torch.nn.Linear(
                    self.head_submodules[dim].output_shape()[0], shape[0]
                )
            elif dim == 2:
                layer = torch.nn.Conv1d(
                    self.head_submodules[dim].output_shape()[0], shape[0],
                    kernel_size=1
                )
            elif dim == 3:
                layer = torch.nn.Conv2d(
                    self.head_submodules[dim].output_shape()[0], shape[0],
                    kernel_size=1
                )
            elif dim == 4:
                layer = torch.nn.Conv3d(
                    self.head_submodules[dim].output_shape()[0], shape[0],
                    kernel_size=1
                )
            else:
                raise ValueError('Invalid dim {}'.format(dim))
            outputs.append((output_name, layer))
        self.output_key_to_layer = torch.nn.ModuleDict(outputs)

    def _validate_shapes(self):
        # outputs can't be higher dim than heads
        # 3d input H,W must match body H, W
        pass

    @classmethod
    def from_args(
        cls,
        args,
        observation_space,
        headname_to_output_shape
    ):
        pass  # TODO

    def forward(self, obs_key_to_obs, internals):
        """

        :param obs_key_to_obs: Dict[str, torch.Tensor (1D | 2D | 3D | 4D)]
        :param internals: Dict[str, torch.Tensor (ND)]
        :return: Tuple[
            Dict[str, torch.Tensor (1D | 2D | 3D | 4D)],
            ChainMap[str, torch.Tensor (ND)]
        ]
        """
        # Process input networks
        nxt_internals = []
        processed_inputs = []
        for key in self._obs_keys:
            result, nxt_internal = self.obs_key_to_input_submod[key].forward(
                obs_key_to_obs[key],
                internals,
                dim=self.body_submodule.dim
            )
            processed_inputs.append(result)
            nxt_internals.append(nxt_internal)

        # Process body
        body_out, nxt_internal = self.body.forward(
            torch.cat(processed_inputs, dim=1),
            internals
        )

        # Process heads
        head_dim_to_head_out = {}
        for head_submod in self.head_submodules:
            head_out, nxt_internal = head_submod.forward(
                self.body_submodule.to_dim(body_out, head_submod.dim),
                internals
            )
            head_dim_to_head_out[head_submod.dim] = head_out
            nxt_internals.append(nxt_internal)

        # Process final outputs
        output_key_to_output = {}
        for key in self._output_keys:
            output = self.output_key_to_layer[key](
                head_dim_to_head_out[len(self.output_space[key])]
            )
            output_key_to_output[key] = output

        return output_key_to_output, ChainMap(*internals)

    def new_internals(self, device):
        # TODO
        return self.body.new_internals(device)


class NetworkBody(torch.nn.Module, RequiresArgs, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_args(cls, nb_input_channel, nb_out_channel, args):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def new_internals(self, device):
        raise NotImplementedError


class NetworkJunction(torch.nn.Module):
    def __init__(self, obsname_to_inputnet):
        super().__init__()

        nb_output_channel = 0
        self.input_nets = torch.nn.ModuleList()
        for name, input_net in obsname_to_inputnet.items():
            nb_output_channel += input_net.nb_output_channel
            self.input_nets.add_module(name, input_net)
        self.nb_output_channel = nb_output_channel

    def forward(self, obs_dict):
        """
        :param obs_dict: Dict[str, Tensor] mapping the pathname to observation
        Tensors
        :return: a Tensor embedding
        """
        embeddings = []
        for name, input_net in self.input_nets.named_children():
            embeddings.append(input_net.forward(obs_dict[name]))
        return torch.cat(embeddings, dim=1)


class SubModule(torch.nn.Module, RequiresArgs, metaclass=abc.ABCMeta):
    """
    SubModule of a ModularNetwork.
    """
    def __init__(self, input_shape):
        super(SubModule, self).__init__()
        self._input_shape = input_shape

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args, input_shape):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _output_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self, dim=None):
        raise NotImplementedError

    @abc.abstractmethod
    def _forward(self, *input):
        """
        :param input: torch.Tensor (1D | 2D | 3D | 4D)
        :return: Tuple[Result, Internals]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _to_1d(self, submodule_output):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_2d(self, submodule_output):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_3d(self, submodule_output):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_4d(self, submodule_output):
        raise NotImplementedError

    @property
    def dim(self):
        return len(self._input_shape)

    @property
    def input_shape(self):
        return self._input_shape

    def to_dim(self, submodule_output, dim):
        """
        :param submodule_output: torch.Tensor (1D | 2D | 3D | 4D)
        Output of a forward pass to be converted.
        :param dim: int Desired dimensionality
        :return:
        """
        if dim <= 0 or dim > 4:
            raise ValueError('Invalid dim: {}'.format(dim))
        elif dim == 1:
            return self._to_1d(submodule_output)
        elif dim == 2:
            return self._to_2d(submodule_output)
        elif dim == 3:
            return self._to_3d(submodule_output)
        elif dim == 4:
            return self._to_4d(submodule_output)

    def forward(self, *input, dim=None):
        submodule_output, internals = self._forward(*input)
        if dim is None:
            return submodule_output, internals
        if dim == 1:
            return self._to_1d(submodule_output), internals
        elif dim == 2:
            return self._to_2d(submodule_output), internals
        elif dim == 3:
            return self._to_3d(submodule_output), internals
        elif dim == 4:
            return self._to_4d(submodule_output), internals
        else:
            raise ValueError('Invalid dim: {}'.format(dim))
