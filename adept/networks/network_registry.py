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
from adept.networks._base import NetworkModule
from adept.networks.net1d.net_1d import Net1D
from adept.utils.requires_args import RequiresArgs


class NetworkRegistry:
    def __init__(self):
        self.name_to_custom_net = {}
        self.name_to_net1d_cls = {}
        self.name_to_net2d_cls = {}
        self.name_to_net3d_cls = {}
        self.name_to_net4d_cls = {}
        self.name_to_netjunc_cls = {}
        self.name_to_netbody_cls = {}


    def register_custom_net(self, name, net_class):
        """
        Add your custom network.

        :param name: str
        :param net_class: NetworkModule
        :return:
        """
        assert issubclass(net_class, NetworkModule)
        net_class.check_args_implemented()
        self.name_to_custom_net[name] = net_class

    def register_net1d_class(self, name, net_class):
        """
        Add your own 1D network.

        :param name: str
        :param net_class: Net1D
        :return:
        """
        assert issubclass(net_class, Net1D)
        net_class.check_args_implemented()
        self.name_to_custom_net[name] = net_class

    def register_net2d_class(self, name, net_class):
        """
        Add your own 2D network.

        :param name: str
        :param net_class: Net2D
        :return:
        """
        assert issubclass(net_class, Net2D)
        net_class.check_args_implemented()
        self.name_to_custom_net[name] = net_class

    def register_net3d_class(self, name, net_class):
        """
        Add your own 3D network.

        :param name: str
        :param net_class: Net3D
        :return:
        """
        assert issubclass(net_class, Net3D)
        net_class.check_args_implemented()
        self.name_to_custom_net[name] = net_class

    def register_net4d_class(self, name, net_class):
        """
        Add your own 4D network.

        :param name: str
        :param net_class: Net4D
        :return:
        """
        assert issubclass(net_class, Net4D)
        net_class.check_args_implemented()
        self.name_to_custom_net[name] = net_class

    def lookup_custom_network(self, net_name):
        """
        Get a custom network by name.

        :param net_name: str
        :return:
        """
        return self.name_to_custom_net[net_name]

    def lookup_net1d(self, name):
        return self.name_to_net1d_cls[name]

    def lookup_net2d(self, name):
        return self.name_to_net2d_cls[name]

    def lookup_net3d(self, name):
        return self.name_to_net3d_cls[name]

    def lookup_net4d(self, name):
        return self.name_to_net4d_cls[name]

    def lookup_netjunc(self, name):
        return self.name_to_netjunc_cls[name]

    def lookup_netbody(self, name):
        return self.name_to_netbody_cls[name]

    def lookup_modular_args(self, net1d, net2d, net3d, net4d, netjunc, netbody):
        """
        :param net1d: str
        :param net2d: str
        :param net3d: str
        :param net4d: str
        :param netjunc: str
        :param netbody: str
        :return: Dict[str, Any]
        """
        return {
            **self.lookup_net1d(net1d).args,
            **self.lookup_net2d(net2d).args,
            **self.lookup_net3d(net3d).args,
            **self.lookup_net4d(net4d).args,
            **self.lookup_netjunc(netjunc).args,
            **self.lookup_netbody(netbody).args
        }

    def prompt_modular_args(self, net1d, net2d, net3d, net4d, netjunc, netbody):
        """
        :param net1d: str
        :param net2d: str
        :param net3d: str
        :param net4d: str
        :param netjunc: str
        :param netbody: str
        :return: Dict[str, Any]
        """
        return RequiresArgs._prompt('NetworkModule', self.lookup_modular_args(
            net1d, net2d, net3d, net4d, netjunc, netbody
        ))
