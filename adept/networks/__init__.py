from adept.networks.bodies import LinearBody, LSTMBody, Mnih2013Linear, Mnih2013LSTM
from adept.networks.discrete.networks import DiscreteIdentity
from adept.networks.vision.networks import (
    FourConv,
    FourConvSpatialAttention,
    FourConvLarger,
    Nature,
    Mnih2013,
    ResNet18,
    ResNet18V2,
    ResNet34,
    ResNet34V2,
    ResNet50,
    ResNet50V2,
    ResNet101V2,
    ResNet101,
    ResNet152V2,
    ResNet152
)

VISION_NETWORKS = {
    'FourConv': FourConv,
    'FourConvSpatialAttention': FourConvSpatialAttention,
    'FourConvLarger': FourConvLarger,
    'Nature': Nature,
    'Mnih2013': Mnih2013,
    'ResNet18': ResNet18,
    'ResNet18V2': ResNet18V2,
    'ResNet34': ResNet34,
    'ResNet34V2': ResNet34V2,
    'ResNet50': ResNet50,
    'ResNet50V2': ResNet50V2,
    'ResNet101': ResNet101,
    'ResNet101V2': ResNet101V2,
    'ResNet152': ResNet152,
    'ResNet152V2': ResNet152V2
}
DISCRETE_NETWORKS = {
    'Identity': DiscreteIdentity
}
NETWORK_BODIES = {
    'Linear': LinearBody,
    'LSTM': LSTMBody,
    'Mnih2013Linear': Mnih2013Linear,
    'Mnih2013LSTM': Mnih2013LSTM
}
