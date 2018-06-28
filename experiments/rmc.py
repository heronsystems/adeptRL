import os

from scripts.rmc import TrainingLoop
from src.utils import base_parser

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = base_parser()
    args = parser.parse_args()

    for env in [
        # 'PongNoFrameskip-v4',
        'BeamRiderNoFrameskip-v4',
        'QbertNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        'SpaceInvadersNoFrameskip-v4'
    ]:
        args.env = env

        args.name = 'rmc'
        TrainingLoop(args).run()
