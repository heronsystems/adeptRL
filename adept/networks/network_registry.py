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
from adept.networks.network_module import NetworkModule
from adept.networks.net1d.submodule_1d import SubModule
from adept.utils.requires_args import RequiresArgs


class NetworkRegistry:
    def __init__(self):
        self.name_to_custom_net = {}
        self.name_to_submodule = {}

        from adept.networks.net1d.identity_1d import Identity1D
        from adept.networks.net1d.linear import Linear
        from adept.networks.net1d.lstm import LSTM
        from adept.networks.net1d.janet import JANET
        from adept.networks.net1d.iqn_lstm import IQNLSTM
        net_1d_cls_by_name = {
            'Identity1D': Identity1D,
            'LSTM': LSTM,
            'JANET': JANET,
            'IQNLSTM': IQNLSTM,
            'Linear': Linear,
        }
        for name, submod_cls in net_1d_cls_by_name.items():
            self.register_submodule(submod_cls)

        from adept.networks.net2d.identity_2d import Identity2D
        net_2d_cls_by_name = {
            'Identity2D': Identity2D
        }
        for name, submod_cls in net_2d_cls_by_name.items():
            self.register_submodule(submod_cls)

        from adept.networks.net3d.identity_3d import Identity3D
        from adept.networks.net3d.four_conv import FourConv
        from adept.networks.net3d.four_conv_lstm import FourConvLSTM
        from adept.networks.net3d.nature import Nature
        net_3d_cls_by_name = {
            'Identity3D': Identity3D,
            'FourConv': FourConv,
            'FourConvLSTM': FourConvLSTM,
            # 'FourConvSpatialAttention': FourConvSpatialAttention,
            # 'FourConvLarger': FourConvLarger,
            'Nature': Nature,
            # 'Mnih2013': Mnih2013,
            # 'ResNet18': ResNet18,
            # 'ResNet18V2': ResNet18V2,
            # 'ResNet34': ResNet34,
            # 'ResNet34V2': ResNet34V2,
            # 'ResNet50': ResNet50,
            # 'ResNet50V2': ResNet50V2,
            # 'ResNet101': ResNet101,
            # 'ResNet101V2': ResNet101V2,
            # 'ResNet152': ResNet152,
            # 'ResNet152V2': ResNet152V2
        }
        for name, submod_cls in net_3d_cls_by_name.items():
            self.register_submodule(submod_cls)

        from adept.networks.net4d.identity_4d import Identity4D
        net_4d_cls_by_name = {
            'Identity4D': Identity4D
        }
        for name, submod_cls in net_4d_cls_by_name.items():
            self.register_submodule(submod_cls)

        from adept.networks.custom import I2A, I2AEmbed, Embedder
        self.register_custom_net(I2A)
        self.register_custom_net(I2AEmbed)
        self.register_custom_net(Embedder)

    def register_custom_net(self, net_cls):
        """
        Add your custom network.

        :param name: str
        :param net_cls: NetworkModule
        :return:
        """
        assert issubclass(net_cls, NetworkModule)
        net_cls.check_args_implemented()
        self.name_to_custom_net[net_cls.__name__] = net_cls
        return self

    def register_submodule(self, submod_cls):
        """
        Add your own SubModule.

        :param name: str
        :param submod_cls: Net1D
        :return:
        """
        assert issubclass(submod_cls, SubModule)
        submod_cls.check_args_implemented()
        self.name_to_submodule[submod_cls.__name__] = submod_cls
        return self

    def lookup_custom_net(self, net_name):
        """
        Get a NetworkModule by name.

        :param net_name: str
        :return: NetworkModule.__class__
        """
        return self.name_to_custom_net[net_name]

    def lookup_submodule(self, submodule_name):
        """
        Get a SubModule by name.

        :param submodule_name: str
        :return: SubModule.__class__
        """
        return self.name_to_submodule[submodule_name]

    def lookup_modular_args(self, args):
        """
        :param args: Dict[name, Any]
        :return: Dict[str, Any]
        """
        return {
            **self.lookup_submodule(args.net1d).args,
            **self.lookup_submodule(args.net2d).args,
            **self.lookup_submodule(args.net3d).args,
            **self.lookup_submodule(args.net4d).args,
            **self.lookup_submodule(args.netbody).args,
            **self.lookup_submodule(args.head1d).args,
            **self.lookup_submodule(args.head2d).args,
            **self.lookup_submodule(args.head3d).args,
            **self.lookup_submodule(args.head4d).args,
        }

    def prompt_modular_args(self, args):
        """
        :param args: Dict[name, Any]
        :return: Dict[str, Any]
        """
        return RequiresArgs._prompt(
            'ModularNetwork',
            self.lookup_modular_args(args)
        )
