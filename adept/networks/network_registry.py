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
from adept.networks.net1d.submodule_1d import SubModule1D
from adept.utils.requires_args import RequiresArgs


class NetworkRegistry:
    def __init__(self):
        self.name_to_custom_net = {}
        self.name_to_submodule = {}

        from adept.networks.net1d.identity_1d import Identity1D
        from adept.networks.net1d.linear import Linear
        from adept.networks.net1d.lstm import LSTM
        net_1d_cls_by_name = {
            'Identity1D': Identity1D,
            'LSTM': LSTM,
            'Linear': Linear,
        }
        for name, submod_cls in net_1d_cls_by_name.items():
            self.register_submodule(name, submod_cls)

        from adept.networks.net3d.four_conv import FourConv
        net_3d_cls_by_name = {
            'FourConv': FourConv,
            # 'FourConvSpatialAttention': FourConvSpatialAttention,
            # 'FourConvLarger': FourConvLarger,
            # 'Nature': Nature,
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
            self.register_submodule(name, submod_cls)

    def register_custom_net(self, name, net_cls):
        """
        Add your custom network.

        :param name: str
        :param net_cls: NetworkModule
        :return:
        """
        assert issubclass(net_cls, NetworkModule)
        net_cls.check_args_implemented()
        self.name_to_custom_net[name] = net_cls

    def register_submodule(self, name, submod_cls):
        """
        Add your own SubModule.

        :param name: str
        :param submod_cls: Net1D
        :return:
        """
        assert issubclass(submod_cls, SubModule1D)
        submod_cls.check_args_implemented()
        self.name_to_custom_net[name] = submod_cls

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
            **self.lookup_submodule(args.netbody).args
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
