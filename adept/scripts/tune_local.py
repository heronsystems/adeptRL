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

Local Mode

Train an agent with a single GPU.

Usage:
    local [options]
    local --resume <path>
    local (-h | --help)

Agent Options:
    --agent <str>           Name of agent class [default: PPO]

Environment Options:
    --env <str>             Environment name [default: PongNoFrameskip-v4]
    --rwd-norm <str>        Reward normalizer name [default: Clip]
    --manager <str>         Manager to use [default: SubProcEnvManager]

Script Options:
    --nb-env <int>          Number of parallel env [default: 64]
    --checkpoint-freq <int>  How often to checkpoint models [default: 5]
    --seed <int>            Seed for random variables [default: 0]
    --nb-step <int>         Number of steps to train for [default: 10e6]
    --load-network <path>   Path to network file (for pretrained weights)
    --load-optim <path>     Path to optimizer file
    --resume <path>         Resume training from log ID .../<logdir>/<env>/<log-id>/
    --config <path>         Use a JSON config file for arguments
    --eval                  Run an evaluation after training
    --prompt                Prompt to modify arguments

Tune Options:
    --num-cpu <int>         Number of cpus to give ray tune [default: 2]
    --cpu-per-trial <int>   Number of cpu cores to give to each trial [default: 1]
    --num-gpu <int>         Number of gpus to give ray tune [default: 1]
    --gpu-per-trial <float>   Number of gpus to gives to each trial [default: .5]
    --num-trials <int>      Number of Tune samples to run [default: 2]
    --training-iter <int>  Number of times to call train() per trial [default: 20]

Network Options:
    --net1d <str>           Network to use for 1d input [default: Identity1D]
    --net2d <str>           Network to use for 2d input [default: Identity2D]
    --net3d <str>           Network to use for 3d input [default: FourConv]
    --net4d <str>           Network to use for 4d input [default: Identity4D]
    --netbody <str>         Network to use on merged inputs [default: LSTM]
    --head1d <str>          Network to use for 1d output [default: Identity1D]
    --head2d <str>          Network to use for 2d output [default: Identity2D]
    --head3d <str>          Network to use for 3d output [default: Identity3D]
    --head4d <str>          Network to use for 4d output [default: Identity4D]
    --custom-network <str>  Name of custom network class

Optimizer Options:
    --optim <str>           Name of optimizer [default: RMSprop]
    --lr <float>            Learning rate [default: 0.0007]
    --warmup <int>          Number of steps to warm up for [default: 0]

Logging Options:
    --tag <str>             Name your run [default: None]
    --logdir <path>         Path to logging directory [default: /tmp/adept_logs/]
    --epoch-len <int>       Save a model every <int> frames [default: 1e6]
    --nb-eval-env <int>     Evaluate agent in a separate thread [default: 0]
    --summary-freq <int>    Tensorboard summary frequency [default: 10]

Troubleshooting Options:
    --profile               Profile this script
"""
import os

from absl import flags
import ray
from adept.container import Init
from adept.trainables.ray_tune_local import Trainable
from adept.utils.script_helpers import (
    parse_none, parse_path
)
from adept.utils.util import DotDict
from adept.registry import REGISTRY as R

# hack to use bypass pysc2 flags
FLAGS = flags.FLAGS
FLAGS(['ray_tune_local.py'])

MODE = 'Local'

def parse_args():
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.strip('--').replace('-', '_'): v for k, v in args.items()}
    del args['h']
    del args['help']
    args = DotDict(args)

    # Ignore other args if resuming
    if args.resume:
        args.resume = parse_path(args.resume)
        return args

    if args.config:
        args.config = parse_path(args.config)
    args.logdir = parse_path(args.logdir)
    args.nb_env = int(args.nb_env)
    args.seed = int(args.seed)
    args.nb_step = int(float(args.nb_step))
    args.tag = parse_none(args.tag)
    args.nb_eval_env = int(args.nb_eval_env)
    args.summary_freq = int(args.summary_freq)
    args.lr = float(args.lr)
    args.warmup = int(float(args.warmup))
    args.epoch_len = int(float(args.epoch_len))
    args.profile = bool(args.profile)
    args.num_cpu = int(args.num_cpu)
    args.num_gpu = int(args.num_gpu)
    args.gpu_per_trial = float(args.gpu_per_trial)
    args.cpu_per_trial = float(args.cpu_per_trial)
    args.num_trials = int(args.num_trials)
    args.training_iter = int(args.training_iter)
    return args

def main(args):
    """
    Run local training.
    :param args: Dict[str, Any]
    :return:
    """

    args, log_id_dir, initial_step, logger = Init.main(MODE, args)
    R.save_extern_classes(log_id_dir)
    args.log_id_dir = log_id_dir
    args.initial_step = initial_step
    args.logger = logger

    from ray import tune
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from hyperopt import hp
    from hyperopt.pyll import scope
    from ray.tune.schedulers import ASHAScheduler

    ray.init(num_cpus=args.num_cpu, num_gpus=args.num_gpu)

    space = {
        'lr': hp.loguniform('lr', 1e-10, .01) - 1,
        'warmup': scope.int(hp.quniform('warmup', 0, 100, q=10)),
        'lstm_nb_hidden' : scope.int(hp.quniform('lstm_nb_hidden', 32, 1024, q=30)),
        'linear_nb_hidden' : scope.int(hp.quniform('linear_nb_hidden', 32, 1024, q=30)),
        'discount' : hp.quniform('max_depth', .985, .999, .001),
        'rollout_minibatch_len': scope.int(hp.quniform('rollout_minibatch_len', 64, 128, q=64)),
        'rollout_len': scope.int(hp.quniform('rollout_len', 128, 256, q=128)),
        'head1d': hp.choice('head1d',['Linear', 'Identity1D']),
        'nb_layer': hp.choice('nb_layer', list(range(1,4)))
    }

    algo = HyperOptSearch(space, metric="term_reward", mode="max")

    analysis = tune.run(Trainable,
                    config=args,
                    search_alg=algo,
                    num_samples=args.num_trials,
                    scheduler=ASHAScheduler(metric="term_reward", mode="max", grace_period=1),
                    resources_per_trial={"cpu": args.cpu_per_trial, "gpu": args.gpu_per_trial},
                    reuse_actors=False,
                    checkpoint_freq=args.checkpoint_freq,
                    stop={"training_iteration": args.training_iter,
                          "term_reward": 21.0}
                        )
    return analysis


if __name__ == '__main__':

    args = parse_args()
    analysis = main(args)
    print("\nBEST_CONFIG:", analysis.dataframe('term_reward')['logdir'].astype(str).get(0))

