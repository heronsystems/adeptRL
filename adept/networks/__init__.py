from .cnn import FourConvCNN
from .lstm import ResNetLSTM, FourConvLSTM
from .attention import AttentionCNN, AttentionLSTM

NETWORKS = {
    'FourConvCNN': FourConvCNN,
    'AttentionCNN': AttentionCNN,
    'FourConvLSTM': FourConvLSTM,
    'AttentionLSTM': AttentionLSTM,
    'ResNetLSTM': ResNetLSTM
}
NETWORK_ARGS = {
    'FourConvCNN': lambda args: (args.normalize,),
    'AttentionCNN': lambda args: (args.nb_head, args.normalize),
    'FourConvLSTM': lambda args: (args.normalize,),
    'AttentionLSTM': lambda args: (args.nb_head, args.normalize,),
    'ResNetLSTM': lambda args: (args.normalize,),
}
FRAME_STACK_NETWORKS = {
    'FourConvCNN',
    'AttentionCNN'
}
