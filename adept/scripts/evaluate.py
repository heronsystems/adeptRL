#!/usr/bin/env python
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
"""
             __           __
  ____ _____/ /__  ____  / /_
 / __ `/ __  / _ \/ __ \/ __/
/ /_/ / /_/ /  __/ /_/ / /_
\__,_/\__,_/\___/ .___/\__/
               /_/
Evaluate

Evaluates an agent after training. Computes N-episode average reward by
loading a saved model from each epoch. N-episode averages are computed by
running N env in parallel.

Usage:
    evaluate (--logdir <path>) [options]
    evaluate (-h | --help)

Required:
    --logdir <path>     Path to train logs (.../logs/<env-id>/<log-id>)

Options:
    --epoch <int>           Epoch number to load [default: None]
    --actor <str>           Name of the eval actor [default: ACActorEval]
    --gpu-id <int>          CUDA device ID of GPU [default: 0]
    --nb-episode <int>      Number of episodes to average [default: 30]
    --start <float>         Epoch to start from [default: 0]
    --end <float>           Epoch to end on [default: -1]
    --seed <int>            Seed for random variables [default: 512]
    --custom-network <str>  Name of custom network class
"""
from adept.container import EvalContainer
from adept.container import Init
from adept.registry import REGISTRY as R
from adept.utils.script_helpers import parse_path, parse_none
from adept.utils.util import DotDict


def parse_args():
    from docopt import docopt

    args = docopt(__doc__)
    args = {k.strip("--").replace("-", "_"): v for k, v in args.items()}
    del args["h"]
    del args["help"]
    args = DotDict(args)
    args.logdir = parse_path(args.logdir)
    # TODO implement Option utility
    epoch_option = parse_none(args.epoch)
    if epoch_option:
        args.epoch = int(float(epoch_option))
    else:
        args.epoch = epoch_option
    args.gpu_id = int(args.gpu_id)
    args.nb_episode = int(args.nb_episode)
    args.start = float(args.start)
    args.end = float(args.end)
    args.seed = int(args.seed)
    return args


def main(args):
    """
    Run an evaluation.
    :param args: Dict[str, Any]
    :return:
    """
    args = DotDict(args)

    Init.print_ascii_logo()
    logger = Init.setup_logger(args.logdir, "eval")
    Init.log_args(logger, args)
    R.load_extern_classes(args.logdir)

    eval_container = EvalContainer(
        args.actor,
        args.epoch,
        logger,
        args.logdir,
        args.gpu_id,
        args.nb_episode,
        args.start,
        args.end,
        args.seed,
    )
    try:
        eval_container.run()
    finally:
        eval_container.close()


if __name__ == "__main__":
    main(parse_args())
