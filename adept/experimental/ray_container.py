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

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from adept.container.base import Container
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.experimental.rollout_queuer import RolloutQueuerAsync
from adept.experimental.rollout_worker import RolloutWorker
from adept.utils import dtensor_to_dev, listd_to_dlist
from adept.utils.logging import SimpleModelSaver


class RayContainer(Container):
    def __init__(
            self,
            args,
            logger,
            log_id_dir,
            initial_step_count
    ):
        ray.init()
        # DISTRIBUTED WORKERS
        # TODO: actually lookup from registry
        self.workers = [RolloutWorker.as_remote(num_cpus=args.worker_cpu_alloc,
                                           num_gpus=args.worker_gpu_alloc)
                        .remote(args, initial_step_count, w_ind)
                        for w_ind in range(args.nb_workers)]
        # ENV
        engine = REGISTRY.lookup_engine(args.env)
        env_cls = REGISTRY.lookup_env(args.env)
        # make a temporary env to get stuff
        # TODO: this can be done by a worker
        dummy_env = env_cls.from_args_curry(args, 0)()

        # NETWORK
        torch.manual_seed(args.seed)
        # TODO: this should come from cluster spec
        device = torch.device("cuda:{}".format(0))
        output_space = REGISTRY.lookup_output_space(
            args.actor_worker, dummy_env.action_space)
        if args.custom_network:
            net_cls = REGISTRY.lookup_network(args.custom_network)
        else:
            net_cls = ModularNetwork
        net = net_cls.from_args(
            args,
            dummy_env.observation_space,
            output_space,
            dummy_env.gpu_preprocessor,
            REGISTRY
        )
        logger.info('Network parameters: ' + str(self.count_parameters(net)))

        def optim_fn(x):
            return torch.optim.RMSprop(x, lr=args.lr, eps=1e-5, alpha=0.99)

        # Learner
        rwd_norm = REGISTRY.lookup_reward_normalizer(
            args.rwd_norm).from_args(args)
        learner = REGISTRY.lookup_learner(args.learner).from_args(args)

        # close dummy env
        dummy_env.close()

        self.learner = learner
        self.nb_step = args.nb_step
        self.nb_env = args.nb_env
        self.network = net.to(device)
        self.network.device = device
        self.dist_net = torch.nn.DataParallel(self.network)
        self.optimizer = optim_fn(self.network.parameters())
        self.device = device
        self.initial_step_count = initial_step_count
        self.log_id_dir = log_id_dir
        self.epoch_len = args.epoch_len
        self.summary_freq = args.summary_freq
        self.logger = logger
        self.summary_writer = SummaryWriter(log_id_dir)
        self.saver = SimpleModelSaver(log_id_dir)
        self.nb_rollouts_in_batch = args.nb_rollouts_in_batch
        self.rollout_queue_size = args.rollout_queue_size
        self.worker_rollout_len = args.worker_rollout_len

        if args.load_network:
            self.network = self.load_network(self.network, args.load_network)
            logger.info('Reloaded network from {}'.format(args.load_network))
        if args.load_optim:
            self.optimizer = self.load_optim(self.optimizer, args.load_optim)
            logger.info('Reloaded optimizer from {}'.format(args.load_optim))

        self.network.train()

        # synchronize worker weights
        self.synchronize_weights(blocking=True)

    def run(self):
        # setup queuer
        self.rollout_queuer = RolloutQueuerAsync(self.workers, self.nb_rollouts_in_batch, self.rollout_queue_size)
        self.rollout_queuer.start()

        # initial setup
        global_step_count = self.initial_step_count
        next_save = self.init_next_save(self.initial_step_count, self.epoch_len)
        prev_step_t = time()
        ep_rewards = torch.zeros(self.nb_env)

        start_time = time()

        # loop until total number steps
        while global_step_count < self.nb_step:
            # possible save
            if global_step_count >= next_save:
                self.saver.save_state_dicts(
                    self.network, global_step_count, self.optimizer
                )
                next_save += self.epoch_len

            # Learn
            batch, terminal_rewards = self.rollout_queuer.get()
            # Perform state updates
            global_step_count += self.nb_env * self.nb_rollouts_in_batch * self.worker_rollout_len

            loss_dict, metric_dict = self.learner.compute_loss(
                self.network, batch, self.dist_net
            )
            total_loss = torch.sum(
                torch.stack(tuple(loss for loss in loss_dict.values()))
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # send weights
            self.synchronize_weights()

            # TODO: send global step to workers

            # write reward summaries
            if any(terminal_rewards):
                terminal_rewards = list(filter(lambda x: x is not None, terminal_rewards))
                self.summary_writer.add_scalar(
                    'reward', np.mean(terminal_rewards), global_step_count
                )

            # write summaries
            cur_step_t = time()
            if cur_step_t - prev_step_t > self.summary_freq:
                print('Metrics:', self.rollout_queuer.metrics())
                self.write_summaries(
                    self.summary_writer, global_step_count, total_loss,
                    loss_dict, metric_dict, self.network.named_parameters()
                )
                prev_step_t = cur_step_t

    def close(self):
        self.rollout_queuer.stop()

    def get_parameters(self):
        params = [p.cpu() for p in self.network.parameters()]
        params.extend([b.cpu() for b in self.network.buffers()])
        return params

    def synchronize_weights(self, blocking=False):
        parameters = self.get_parameters()
        futures = [w.set_weights.remote(parameters) for w in self.workers]
        if blocking:
            ray.get(futures)

