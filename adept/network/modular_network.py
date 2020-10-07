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

from adept.network.base.base import BaseNetwork


class ModularNetwork(BaseNetwork, metaclass=abc.ABCMeta):
    """
    A neural network comprised of SubModules. Tries to be smart about
    converting dimensionality. Does not need or build submodules for unused
    source nets or heads.
    """

    def __init__(
        self,
        source_nets,
        body_submodule,
        head_submodules,
        output_space,
        gpu_preprocessor,
    ):
        """
        :param source_nets: Dict[ObsKey, SubModule]
        :param body_submodule: SubModule
        :param head_submodules: Dict[Dim, SubModule]
        :param output_space: Dict[OutputKey, Shape]
        :param gpu_preprocessor: ObsPreprocessor
        """
        super().__init__()
        self.gpu_preprocessor = gpu_preprocessor

        # Source Nets
        self.source_nets = torch.nn.ModuleDict(
            [(key, net) for key, net in source_nets.items()]
        )

        # Body
        self.body = body_submodule

        # Heads
        self.heads = torch.nn.ModuleDict(head_submodules)

        # Outputs
        self.output_layers = self._build_out_layers(output_space, self.heads)

        self._obs_keys = list(source_nets.keys())
        self._output_keys = list(output_space.keys())
        self._output_space = output_space
        self._output_dims = set(
            [len(shape) for shape in self._output_space.values()]
        )
        self._check_outputs_have_heads()
        self._validate_shapes()

    @staticmethod
    def _build_out_layers(output_space, heads):
        """
        Build output_layers to match the desired output space.
        * For 1D outputs, converts uses a Linear layer
        * For 2D outputs, uses a Conv1D, kernel size 1
        * For 3D outputs, uses a 1x1 Conv
        * For 4D outputs, uses a 1x1x1 Conv

        :param output_space: Dict[OutputKey, Shape]
        :param heads: Dict[DimStr, SubModule]
        :return: ModuleDict[OutputKey, torch.nn.Module]
        """
        outputs = []
        for output_name, shape in output_space.items():
            dim = len(shape)
            if dim == 1:
                layer = torch.nn.Linear(
                    heads[str(dim)].output_shape(dim)[0], shape[0]
                )
            elif dim == 2:
                layer = torch.nn.Conv1d(
                    heads[str(dim)].output_shape(dim)[0],
                    shape[0],
                    kernel_size=1,
                )
            elif dim == 3:
                layer = torch.nn.Conv2d(
                    heads[str(dim)].output_shape(dim)[0],
                    shape[0],
                    kernel_size=1,
                )
            elif dim == 4:
                layer = torch.nn.Conv3d(
                    heads[str(dim)].output_shape(dim)[0],
                    shape[0],
                    kernel_size=1,
                )
            else:
                raise ValueError("Invalid dim {}".format(dim))
            outputs.append((output_name, layer))
        return torch.nn.ModuleDict(outputs)

    def _validate_shapes(self):
        """
        Ensures SubModule graph is valid.
        :return:
        """
        # non feature dims of source nets match non feature dim of body
        # Doesn't matter if converting to 1D
        if self.body.dim > 1:
            for submod in self.source_nets.values():
                if submod.dim > 1:
                    shape = submod.output_shape(dim=self.body.dim)
                    for a, b in zip(shape[1:], self.body.input_shape[1:]):
                        assert (
                            a == b or a == 1 or b == 1
                        ), "Source-Body conflict: {} {}".format(
                            shape, self.body.input_shape
                        )
            # non feature dims of body out must match non feature dims of head
            for submod in self.heads.values():
                if submod.dim > 1:
                    shape = self.body.output_shape(dim=submod.dim)
                    for a, b in zip(shape[1:], submod.input_shape[1:]):
                        assert (
                            a == b or a == 1 or b == 1
                        ), "Body-Head conflict: {} {}".format(
                            shape, submod.input_shape
                        )

        # non-feature dims of heads == non-feature dims of output shapes
        for shape in self._output_space.values():
            dim = len(shape)
            if dim > 1:
                submod = self.heads[str(dim)]
                head_shp = submod.output_shape(dim)
                for a, b in zip(shape[1:], head_shp[1:]):
                    assert a == b, "Head-Output conflict: {}-{}".format(
                        head_shp, shape
                    )

    def _check_outputs_have_heads(self):
        for dim in self._output_dims:
            assert str(dim) in self.heads

    @classmethod
    def from_args(
        cls, args, observation_space, output_space, gpu_preprocessor, net_reg
    ):
        """
        Construct a Modular Network from arguments.

        :param args: Dict[ArgName, Any]
        :param observation_space: Dict[ObsKey, Shape]
        :param output_space: Dict[OutputKey, Shape]
        :param gpu_preprocessor: ObsPreprocessor
        :param net_reg: NetworkRegistry
        :return: ModularNetwork
        """
        # Dict[ObsKey, SubModule]
        # for shape in observation space, get dim
        # instantiate input submodule of that dim
        obs_key_to_submod = {}
        for obs_key, shape in observation_space.items():
            dim = len(shape)
            if dim == 1:
                submod = net_reg.lookup_submodule(args.net1d).from_args(
                    args, shape, obs_key
                )
            elif dim == 2:
                submod = net_reg.lookup_submodule(args.net2d).from_args(
                    args, shape, obs_key
                )
            elif dim == 3:
                submod = net_reg.lookup_submodule(args.net3d).from_args(
                    args, shape, obs_key
                )
            elif dim == 4:
                submod = net_reg.lookup_submodule(args.net4d).from_args(
                    args, shape, obs_key
                )
            else:
                raise ValueError("Invalid dim: {}".format(dim))
            obs_key_to_submod[obs_key] = submod

        # SubModule
        # initialize body submodule
        body_cls = net_reg.lookup_submodule(args.netbody)
        nb_body_feature = sum(
            [
                submod.output_shape(dim=body_cls.dim)[0]
                for submod in obs_key_to_submod.values()
            ]
        )
        if body_cls.dim > 1:
            other_dims = [
                submod.output_shape(dim=body_cls.dim)[1:]
                for submod in obs_key_to_submod.values()
                if submod.dim == body_cls.dim
            ][0]
        else:
            other_dims = []
        input_shape = [nb_body_feature,] + list(other_dims)
        body_submod = body_cls.from_args(args, input_shape, "body")

        # Dict[Dim, SubModule]
        # instantiate heads based on output_shapes
        head_submodules = {}
        for output_key, shape in output_space.items():
            dim = len(shape)
            if dim in head_submodules:
                continue
            elif dim == 1:
                submod_cls = net_reg.lookup_submodule(args.head1d)
            elif dim == 2:
                submod_cls = net_reg.lookup_submodule(args.head2d)
            elif dim == 3:
                submod_cls = net_reg.lookup_submodule(args.head3d)
            elif dim == 4:
                submod_cls = net_reg.lookup_submodule(args.head4d)
            else:
                raise ValueError("Invalid dim: {}".format(dim))
            submod = submod_cls.from_args(
                args,
                body_submod.output_shape(submod_cls.dim),
                "head" + str(dim) + "d",
            )
            head_submodules[str(dim)] = submod
        return cls(
            obs_key_to_submod,
            body_submod,
            head_submodules,
            output_space,
            gpu_preprocessor,
        )

    def forward(self, observation, internals):
        """

        :param observation: Dict[str, torch.Tensor (1D | 2D | 3D | 4D)]
        :param internals: Dict[str, torch.Tensor (ND)]
        :return: Tuple[
            Dict[str, torch.Tensor (1D | 2D | 3D | 4D)],
            Dict[str, torch.Tensor (ND)]
        ]
        """
        proc_obs = self.gpu_preprocessor(observation)
        # Process input network
        nxt_internals = []
        processed_inputs = []
        for key in self._obs_keys:
            result, nxt_internal = self.source_nets[key].forward(
                proc_obs[key], internals, dim=self.body.dim
            )
            processed_inputs.append(result)
            nxt_internals.append(nxt_internal)

        # Process body
        processed_inputs = self._expand_dims(processed_inputs)
        body_out, nxt_internal = self.body.forward(
            torch.cat(processed_inputs, dim=1), internals
        )
        nxt_internals.append(nxt_internal)

        # Process heads
        head_out_by_dim = {}
        for dim in self._output_dims:
            cur_head = self.heads[str(dim)]
            head_out, next_internal = cur_head.forward(
                self.body.to_dim(body_out, cur_head.dim), internals, dim=dim
            )
            head_out_by_dim[dim] = head_out
            nxt_internals.append(nxt_internal)

        # Process final outputs
        output_by_key = {}
        for key in self._output_keys:
            output = self.output_layers[key].forward(
                head_out_by_dim[len(self._output_space[key])]
            )
            output_by_key[key] = output

        merged_internals = {}
        for internal in nxt_internals:
            for k, v in internal.items():
                merged_internals[k] = v
        return output_by_key, merged_internals, proc_obs

    @staticmethod
    def _expand_dims(inputs):
        """
        Expands dimensions when input dimension is 1.

        :param inputs: List[torch.Tensor]
        :return: List[torch.Tensor]
        """
        if len(inputs[0].shape) <= 2:
            return inputs

        target_shape = max([inpt.shape[2:] for inpt in inputs])
        processed_inputs = []
        for inpt in inputs:
            if inpt.shape[2:] < target_shape:
                processed_inputs.append(inpt.expand(-1, -1, *target_shape))
            else:
                processed_inputs.append(inpt)

        return processed_inputs

    def new_internals(self, device):
        """

        :param device:
        :return: Dict[
        """
        internals = [
            submod.new_internals(device) for submod in self.source_nets.values()
        ]
        internals.append(self.body.new_internals(device))
        internals += [
            submod.new_internals(device) for submod in self.heads.values()
        ]

        merged_internals = {}
        for internal in internals:
            for k, v in internal.items():
                merged_internals[k] = v
        return merged_internals

    def to(self, device):
        super().to(device)
        self.gpu_preprocessor = self.gpu_preprocessor.to(device)
        return self
