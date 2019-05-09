from adept.networks import NetworkModule, NetworkRegistry
from adept.networks.net2d.gpt2 import GPT2
from torch.nn import Conv2d, Linear, BatchNorm2d, BatchNorm1d, Conv1d
from torch.nn import functional as F
import torch


class GPT2RL(NetworkModule):
    # You will be prompted for these when training script starts
    args = GPT2.args

    def __init__(self, in_shape, out_space, nb_layer, nb_head, layer_norm_eps):
        super(GPT2RL, self).__init__()
        self.conv1 = Conv2d(in_shape[0], 32, 7, stride=2, padding=1, bias=False)
        self.conv2 = Conv2d(32, 30, 3, stride=2, padding=1, bias=False)
        self.gpt2 = GPT2((400, 32), 'gpt2', nb_layer, nb_head, layer_norm_eps)
        self.conv3 = Conv2d(32, 32, 3, stride=2, padding=1, bias=False)
        self.conv4 = Conv2d(32, 32, 3, stride=2, padding=1, bias=False)
        self.fc = Linear(800, 512, bias=False)

        self.bn1 = BatchNorm2d(32)
        self.bn2 = BatchNorm2d(30)
        self.bn3 = BatchNorm2d(32)
        self.bn4 = BatchNorm2d(32)
        self.bn_fc = BatchNorm1d(512)

        output_heads = {}
        for output_key, shape in out_space.items():
            dim = len(shape)
            if dim == 1:
                module = Linear(512, shape[0])
            else:
                raise NotImplementedError
            output_heads[output_key] = module
        self.out_heads = torch.nn.ModuleDict(output_heads)

        h, w = 20, 20
        self.register_buffer('x_enc', torch.linspace(0, 1, steps=w))
        self.register_buffer('y_enc', torch.linspace(0, 1, steps=h))

    @classmethod
    def from_args(
        cls,
        args,
        observation_space,
        output_space,
        net_reg
    ):
        """
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
        assert len(observation_space) == 1
        in_shape = list(observation_space.values())[0]
        return cls(in_shape, output_space, args.nb_layer, args.nb_head,
                                args.layer_norm_eps)

    def new_internals(self, device):
        """
        Define any initial hidden states here, move them to device if necessary.
        InternalKey=str
        :return: Dict[InternalKey, torch.Tensor (ND)]
        """
        return self.gpt2.new_internals(device)

    def forward(self, observation, internals):
        """
        Compute forward pass.
        ObsKey = str
        InternalKey = str
        :param observation: Dict[ObsKey, torch.Tensor (1D | 2D | 3D | 4D)]
        :param internals: Dict[InternalKey, torch.Tensor (ND)]
        :return: Dict[OutKey, torch.Tensor], Dict[InternalKey, torch.Tensor]
        """
        x = observation['Box']
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        b, f, h, w = x.size()

        x_enc = self.x_enc.view(1, 1, 1, -1).expand(b, -1, h, -1)
        y_enc = self.y_enc.view(1, 1, -1, 1).expand(b, -1, -1, w)
        x = torch.cat([x, x_enc, y_enc], dim=1).view(b, f + 2, h * w)
        x = x.permute(0, 2, 1)
        x, new_internals = self.gpt2.forward(x, internals)

        x = x.permute(0, 2, 1)
        x = x.view(b, f + 2, h, w)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = x.view(b, -1)
        x = self.fc(x)
        x = self.bn_fc(x)
        x = F.relu(x)

        results = {key: module(x) for key, module in self.out_heads.items()}

        return results, new_internals


if __name__ == '__main__':
    from adept.scripts.local import parse_args, main
    args = parse_args()
    network_reg = NetworkRegistry()
    network_reg.register_custom_net(GPT2RL)

    main(args, net_registry=network_reg)

    # Call script like this to train agent:
    # python -m custom_network_stub.py --custom-net MyCustomNetwork