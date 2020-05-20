"""
Custom submodule stub
"""
from adept.networks import (
    SubModule1D,
    SubModule2D,
    SubModule3D,
    SubModule4D,
    NetworkRegistry,
)
from adept.scripts.local import parse_args, main


# If your Module processes 2D, then inherit SubModule2D and so on.
# Dimensionality refers to feature map dimensions not including batch.
# ie. (F, ) = 1D, (F, L) = 2D, (F, H, W) = 3D, (F, D, H, W) = 4D
class MyCustomSubModule1D(SubModule1D):
    # You will be prompted for these when training script starts
    args = {"example_arg1": True, "example_arg2": 5}

    def __init__(self, input_shape, id):
        super(MyCustomSubModule1D, self).__init__(input_shape, id)

    @classmethod
    def from_args(cls, args, input_shape, id):
        """
        Construct a MyCustomSubModule1D from arguments.

        :param args: Dict[ArgName, Any]
        :param input_shape: Tuple[*int]
        :param id: str
        :return: MyCustomSubModule1D
        """
        pass

    @property
    def _output_shape(self):
        """
        Return the output shape. If it's a function of the input shape, you can
        access the input shape via ``self.input_shape``.

        :return: Tuple[*int]
        """
        pass

    def _forward(self, input, internals, **kwargs):
        """
        Compute forward pass.

        ObsKey = str
        InternalKey = str

        :param observation: Dict[ObsKey, torch.Tensor]
        :param internals: Dict[InternalKey, torch.Tensor (ND)]
        :return: torch.Tensor
        """
        pass

    def _new_internals(self):
        """
        Define any initial hidden states here, move them to device if necessary.

        InternalKey=str

        :return: Dict[InternalKey, torch.Tensor (ND)]
        """
        pass


if __name__ == "__main__":
    args = parse_args()
    network_reg = NetworkRegistry()
    network_reg.register_submodule(MyCustomSubModule1D)

    main(args, net_registry=network_reg)

    # Call script like this to train agent:
    # python -m custom_submodule_stub.py --net1d MyCustomSubModule1D
