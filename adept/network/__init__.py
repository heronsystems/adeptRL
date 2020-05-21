from .base.network_module import NetworkModule
from .modular_network import ModularNetwork
from .net1d.submodule_1d import SubModule1D
from .net2d.submodule_2d import SubModule2D
from .net3d.submodule_3d import SubModule3D
from .net4d.submodule_4d import SubModule4D

from .net1d.linear import Linear
from .net1d.identity_1d import Identity1D
from .net1d.lstm import LSTM

from .net2d.identity_2d import Identity2D

from .net3d.identity_3d import Identity3D
from .net3d.four_conv import FourConv

from .net4d.identity_4d import Identity4D

NET_REG = []
SUBMOD_REG = [
    Identity1D,
    Linear,
    LSTM,
    Identity2D,
    Identity3D,
    FourConv,
    Identity4D,
]
