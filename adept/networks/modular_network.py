import abc
from collections import ChainMap

import torch

from adept.networks._base import BaseNetwork


class ModularNetwork(BaseNetwork, metaclass=abc.ABCMeta):
    def __init__(
        self,
        obs_key_to_submod,
        body_submodule,
        head_submodules,
        output_space
    ):
        """
        :param obs_key_to_submod: Dict[ObsKey, SubModule]
        :param body_submodule: SubModule
        :param head_submodules: List[SubModule]
        :param output_space: Dict[OutputKey, Shape]
        """
        # modular network doesn't have networks for unused inputs
        super().__init__()

        # Input Nets
        self._obs_keys = list(obs_key_to_submod.keys())
        self.obs_key_to_input_submod = torch.nn.ModuleDict(
            [(key, net) for key, net in obs_key_to_submod.items()]
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
        self._validate_shapes()

    def _validate_shapes(self):
        # heads can't be higher dim than body
        assert all([
            head.dim <= self.body_submodule.dim
            for head in self.head_submodules
        ])
        # output dims must have a corresponding head of the same dim
        pass
        # non feature dims of input submodules must be same
        pass
        # non-feature dimensions of heads must match desired output_shape
        pass
        # there must exist an input submodule such that input_dim == body_dim
        pass

    @classmethod
    def from_args(
        cls,
        args,
        observation_space,
        output_space,
        network_registry
    ):
        """
        Construct a Modular Network from arguments.

        :param args: Dict[ArgName, Any]
        :param observation_space: Dict[ObsKey, Shape]
        :param output_space: Dict[OutputKey, Shape]
        :param network_registry: NetworkRegistry
        :return: ModularNetwork
        """
        # Dict[ObsKey, SubModule]
        # for shape in observation space, get dim
        # instantiate input submodule of that dim
        obs_key_to_submod = {}
        for obs_key, shape in observation_space.items():
            dim = len(shape)
            if dim == 1:
                submod = network_registry.lookup_submodule(
                    args.net1d).from_args(args, shape, obs_key)
            elif dim == 2:
                submod = network_registry.lookup_submodule(
                    args.net2d).from_args(args, shape, obs_key)
            elif dim == 3:
                submod = network_registry.lookup_submodule(
                    args.net3d).from_args(args, shape, obs_key)
            elif dim == 4:
                submod = network_registry.lookup_submodule(
                    args.net4d).from_args(args, shape, obs_key)
            else:
                raise ValueError('Invalid dim: {}'.format(dim))
            obs_key_to_submod[obs_key] = submod

        # SubModule
        # initialize body submodule
        body_cls = network_registry.lookup_submodule(args.netbody)
        nb_body_feature = sum([
            shape[0]
            for shape in obs_key_to_submod.values()
        ])
        if body_cls.dim > 1:
            other_dims = [
                shape[1:]
                for shape in obs_key_to_submod.values()
                if len(shape) == body_cls.dim
            ][0]
        else:
            other_dims = []
        input_shape = [nb_body_feature, ] + other_dims
        body_submod = body_cls.from_args(args, input_shape, 'body')

        # List[SubModule]
        # instantiate heads based on output_shapes
        head_submodules = []
        for output_key, shape in output_space.items():
            dim = len(shape)
            if dim == 1:
                submod = network_registry.lookup_submodule(
                    args.head1d).from_args(args, shape, output_key)
            elif dim == 2:
                submod = network_registry.lookup_submodule(
                    args.head2d).from_args(args, shape, output_key)
            elif dim == 3:
                submod = network_registry.lookup_submodule(
                    args.head3d).from_args(args, shape, output_key)
            elif dim == 4:
                submod = network_registry.lookup_submodule(
                    args.head4d).from_args(args, shape, output_key)
            else:
                raise ValueError('Invalid dim: {}'.format(dim))
            head_submodules.append(submod)
        return cls(
            obs_key_to_submod, body_submod, head_submodules, output_space
        )

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
