import os

from scripts.attention import TrainingLoop as AttnLoop
from scripts.cnn import TrainingLoop as CNNLoop
from scripts.lstm import TrainingLoop as LSTMLoop
from src.utils import base_parser

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = base_parser()
    args = parser.parse_args()

    for env in [
        'PongNoFrameskip-v4',
        'BeamRiderNoFrameskip-v4',
        'QbertNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        'SpaceInvadersNoFrameskip-v4'
    ]:
        args.env = env

        if env != 'PongNoFrameskip-v4':
            args.name = 'cnn_baseline'
            CNNLoop(args).run()

        if env != 'PongNoFrameskip-v4':
            args.name = 'lstm_baseline'
            LSTMLoop(args).run()

        args.name = 'attention_baseline'
        args.nb_head = 1
        AttnLoop(args).run()
