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
import os
from time import time

import torch

from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.utils import listd_to_dlist, dtensor_to_dev
from adept.utils.logging import SimpleModelSaver
from torch.utils.tensorboard import SummaryWriter

from .base import Container


class ActorLearnerHost(Container):
    def __init__(
            self,
            args,
            logger,
            log_id_dir,
            initial_step_count,
            local_rank,
            global_rank,
            world_size,
            groups
    ):
        seed = args.seed \
            if global_rank == 0 \
            else args.seed + args.nb_env * global_rank
        logger.info('Using {} for rank {} seed.'.format(seed, global_rank))

        # ENV (temporary)
        env_cls = REGISTRY.lookup_env(args.env)
        env = env_cls.from_args(args.env, 0)
        env.close()

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device("cuda:{}".format(local_rank))
        output_space = REGISTRY.lookup_output_space(
            args.agent, env.action_space)
        if args.custom_network:
            net_cls = REGISTRY.lookup_network(args.custom_network)
        else:
            net_cls = ModularNetwork
        net = net_cls.from_args(
            args,
            env.observation_space,
            output_space,
            env.gpu_preprocessor,
            REGISTRY
        )
        logger.info('Network parameters: ' + str(self.count_parameters(net)))

        def optim_fn(x):
            return torch.optim.RMSprop(x, lr=args.lr, eps=1e-5, alpha=0.99)

        # LEARNER / EXP
        rwd_norm = REGISTRY.lookup_reward_normalizer(
            args.rwd_norm).from_args(args)
        actor = REGISTRY.lookup_actor(args.actor_host).from_args(
            args, env.action_space)
        learner = REGISTRY.lookup_learner(args.learner).from_args(args)
        exp_cls = REGISTRY.lookup_exp(args.exp)

        self.actor = actor
        self.learner = learner
        self.exps = [exp_cls.from_args(args, rwd_norm) for _ in range(groups)]
        self.batch_size = args.learn_batch_size
        self.nb_step = args.nb_step
        self.network = net.to(device)
        self.optimizer = optim_fn(self.network.parameters())
        self.device = device
        self.initial_step_count = initial_step_count
        self.log_id_dir = log_id_dir
        self.epoch_len = args.epoch_len
        self.summary_freq = args.summary_freq
        self.logger = logger
        self.summary_writer = SummaryWriter(
            os.path.join(log_id_dir, 'rank{}'.format(global_rank))
        )
        self.saver = SimpleModelSaver(log_id_dir)
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size

        if args.load_network:
            self.network = self.load_network(self.network, args.load_network)
            logger.info('Reloaded network from {}'.format(args.load_network))
        if args.load_optim:
            self.optimizer = self.load_optim(self.optimizer, args.load_optim)
            logger.info('Reloaded optimizer from {}'.format(args.load_optim))

        self.network.train()

    def run(self):
        step_count = self.initial_step_count
        next_save = self.init_next_save(self.initial_step_count, self.epoch_len)
        prev_step_t = time()

        start_time = time()

        # sync exp from workers
        # actor recompute forward
        # write into cache
        # learn on ready

        while step_count < self.nb_step:
            w_exps = self.exp.recv()
            for ob, internal in zip(w_exps['obs'], w_exps['internals']):
                _, h_exp, _ = self.actor.act(self.network, ob, internal)
                self.exp.write_actor(h_exp)

                # Perform state updates
                step_count += self.batch_size

            if self.exp.is_ready():
                loss_dict, metric_dict = self.learner.compute_loss(
                    self.network, self.exp.read(),
                    w_exps['next_obs'], w_exps['internals'][-1]
                )
                total_loss = torch.sum(
                    torch.stack(tuple(loss for loss in loss_dict.values()))
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                self.exp.clear()

                # write summaries
                cur_step_t = time()
                if cur_step_t - prev_step_t > self.summary_freq:
                    self.write_summaries(
                        self.summary_writer, step_count, total_loss,
                        loss_dict, metric_dict, self.network.named_parameters()
                    )
                    prev_step_t = cur_step_t

    def close(self):
        return None


class ActorLearnerWorker(Container):
    """
    Actor Learner Architecture worker.
    """

    def __init__(
            self,
            args,
            logger,
            log_id_dir,
            initial_step_count,
            local_rank,
            global_rank,
            world_size,
            group
    ):
        self.env_mgr = None

    def run(self, nb_step, initial_step_count=None):
        pass

    def close(self):
        self.env_mgr.close()
