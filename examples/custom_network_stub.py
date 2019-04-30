"""
Use a custom network.
"""
from adept.networks import NetworkModule, NetworkRegistry
from adept.scripts.local import parse_args, main


class MyCustomNetwork(NetworkModule):
    # You will be prompted for these when training script starts
    args = {
        'example_arg1': True,
        'example_arg2': 5
    }

    def __init__(self):
        super(MyCustomNetwork, self).__init__()
        # Set properties and whatnot here

    @classmethod
    def from_args(
        cls,
        args,
        observation_space,
        output_space,
        net_reg
    ):
        """
        Construct a MyCustomNetwork from arguments.

        ArgName = str
        ObsKey = str
        OutputKey = str
        Shape = Tuple[*int]

        :param args: Dict[ArgName, Any]
        :param observation_space: Dict[ObsKey, Shape]
        :param output_space: Dict[OutputKey, Shape]
        :param net_reg: NetworkRegistry
        :return: MyCustomNetwork
        """
        pass

    def new_internals(self, device):
        """
        Define any initial hidden states here, move them to device if necessary.

        InternalKey=str

        :return: Dict[InternalKey, torch.Tensor (ND)]
        """
        pass

    def forward(self, observation, internals):
        """
        Compute forward pass.

        ObsKey = str
        InternalKey = str

        :param observation: Dict[ObsKey, torch.Tensor (1D | 2D | 3D | 4D)]
        :param internals: Dict[InternalKey, torch.Tensor (ND)]
        :return: torch.Tensor
        """
        pass


if __name__ == '__main__':
    args = parse_args()
    network_reg = NetworkRegistry()
    network_reg.register_custom_net(MyCustomNetwork)

    main(args, net_registry=network_reg)

    # Call script like this to train agent:
    # python -m custom_network_stub.py --custom-network MyCustomNetwork
