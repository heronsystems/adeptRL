import unittest

from adept.networks.modular_network import ModularNetwork
from adept.networks.net1d.identity_1d import Identity1D
from adept.networks.net2d.identity_2d import Identity2D
from adept.networks.net3d.identity_3d import Identity3D
from adept.networks.net4d.identity_4d import Identity4D


class TestModularNetwork(unittest.TestCase):
    # Example of valid structure
    stub_1d = Identity1D((16,), 'stub_1d')
    stub_2d = Identity2D((16, 32 * 32), 'stub_2d')
    stub_3d = Identity3D((16, 32, 32), 'stub_3d')
    stub_4d = Identity4D((16, 32, 32, 32), 'stub_4d')

    source_nets = {
        'source_1d': stub_1d,
        'source_2d': stub_2d,
        'source_3d': stub_3d,
        'source_4d': stub_4d
    }
    body = stub_3d
    heads = [stub_1d, stub_2d, stub_3d]
    output_space = {
        'output_1d': (16,),
        'output_2d': (16, 32 * 32),
        'output_3d': (16, 32, 32),
    }

    def test_heads_not_higher_dim_than_body(self):
        stub_1d = Identity1D((32,), 'stub_1d')
        stub_2d = Identity2D((32, 32), 'stub_2d')

        source_nets = {'source': stub_1d}
        body = stub_1d
        heads = [stub_2d]  # should error
        output_space = {'output': (32, 32)}

        with self.assertRaises(AssertionError):
            ModularNetwork(source_nets, body, heads, output_space)

    def test_source_nets_match_body(self):
        stub_32 = Identity2D((32, 32), 'stub_32')
        stub_64 = Identity2D((32, 64), 'stub_64')

        source_nets = {'source': stub_32}
        body = stub_64  # should error
        heads = [stub_64]
        output_space = {'output': (32, 64)}

        with self.assertRaises(AssertionError):
            ModularNetwork(source_nets, body, heads, output_space)

    def test_body_matches_heads(self):
        stub_32 = Identity2D((32, 32), 'stub_32')
        stub_64 = Identity2D((32, 64), 'stub_64')

        source_nets = {'source': stub_32}
        body = stub_32
        heads = [stub_64]  # should error
        output_space = {'output': (32, 64)}

        with self.assertRaises(AssertionError):
            ModularNetwork(source_nets, body, heads, output_space)

    def test_output_has_a_head(self):
        stub_2d = Identity2D((32, 32), 'stub_2d')

        source_nets = {'source': stub_2d}
        body = stub_2d
        heads = [stub_2d]
        output_space = {'output': (32, 32, 32)}  # should error
        with self.assertRaises(AssertionError):
            ModularNetwork(source_nets, body, heads, output_space)

    def test_heads_match_out_shapes(self):
        stub_2d = Identity2D((32, 32), 'stub_2d')

        source_nets = {'source': stub_2d}
        body = stub_2d
        heads = [stub_2d]
        output_space = {'output': (32, 64)}  # should error
        with self.assertRaises(AssertionError):
            ModularNetwork(source_nets, body, heads, output_space)

    def test_valid_structure(self):
        try:
            ModularNetwork(
                self.source_nets,
                self.body,
                self.heads,
                self.output_space
            )
        except:
            self.fail('Unexpected exception')

    def test_forward(self):
        import torch
        BATCH = 8
        obs = {
            'source_1d': torch.zeros((BATCH, 16,)),
            'source_2d': torch.zeros((BATCH, 16, 32 * 32)),
            'source_3d': torch.zeros((BATCH, 16, 32, 32)),
            'source_4d': torch.zeros((BATCH, 16, 32, 32, 32))
        }
        # try:
        net = ModularNetwork(
            self.source_nets,
            self.body,
            self.heads,
            self.output_space
        )
        outputs, _ = net.forward(obs, {})
        print(outputs)
        # except:
        #     pass


if __name__ == '__main__':
    unittest.main(verbosity=1)
