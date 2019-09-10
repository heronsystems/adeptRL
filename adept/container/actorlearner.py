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
import torch
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter

from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.rewardnorm import Identity
from adept.utils import listd_to_dlist, dtensor_to_dev
from adept.utils.logging import SimpleModelSaver
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
        actor_cls = REGISTRY.lookup_actor(args.actor_host)
        builder = actor_cls.exp_spec_builder(
            env.observation_space,
            env.action_space,
            net.internal_space(),
            args.nb_env
        )
        actor = actor_cls.from_args(args, env.action_space)
        learner = REGISTRY.lookup_learner(args.learner).from_args(args)
        exp_cls = REGISTRY.lookup_exp(args.exp).from_args(args, rwd_norm, builder)

        self.actor = actor
        self.learner = learner
        self.exp = REGISTRY.lookup_exp(args.exp_host).from_args(args, rwd_norm)
        self.exps = [
            exp_cls.from_args(args, Identity()) for _ in range(len(groups))
        ]
        self.batch_size = args.learn_batch_size
        self.nb_learn_batch = args.nb_learn_batch
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
        self.groups = groups

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

        dist.barrier()
        handles = []
        for i, exp in enumerate(self.exps):
            handles.append(exp.sync(i + 1, self.groups[i]))

        while step_count < self.nb_step:
            # wait for n batches to sync
                # this needs to run in a thread
            learn_indices = []
            while len(learn_indices) < self.nb_learn_batch:
                for i, hand in enumerate(handles):
                    if all([h.is_completed() for h in hand]):
                        learn_indices.append(i)

            # merge tensors along batch dimension
                # need to pick a data structure for experiences
                # Dict[str,List[Tensor]]

            for rollout_idx in range(len(self.exp)):
                merged_exp = {}
                for k in self.exps[0].keys():
                    tensors_to_cat = []
                    for i in learn_indices:
                        exp = self.exps[i]
                        tensors_to_cat.append(exp[k][rollout_idx])

                    cat = torch.cat(tensors_to_cat)
                    merged_exp[k] = cat
                self.exp.write_actor(merged_exp, no_env=True)


            # unblock the selected workers
                # resync
            for i in learn_indices:
                dist.barrier(self.groups[i])
                handles[i] = self.exps[i].sync(i + 1, self.groups[i])

            # forward passes
            # learning step

            for ob, internal in zip(
                merged_exp['observations'],
                merged_exp['internals'],

            ):
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
        seed = args.seed \
            if global_rank == 0 \
            else args.seed + args.nb_env * global_rank
        logger.info('Using {} for rank {} seed.'.format(seed, global_rank))

        # ENV
        engine = REGISTRY.lookup_engine(args.env)
        env_cls = REGISTRY.lookup_env(args.env)
        env_mgr = SubProcEnvManager.from_args(args, engine, env_cls, seed=seed)

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device("cuda:{}".format(self._device(local_rank)))
        output_space = REGISTRY.lookup_output_space(
            args.agent, env_mgr.action_space)
        if args.custom_network:
            net_cls = REGISTRY.lookup_network(args.custom_network)
        else:
            net_cls = ModularNetwork
        net = net_cls.from_args(
            args,
            env_mgr.observation_space,
            output_space,
            env_mgr.gpu_preprocessor,
            REGISTRY
        )
        actor = REGISTRY.lookup_actor(args.actor_worker).from_args(
            args, env_mgr.action_space
        )
        exp = REGISTRY.lookup_exp(args.exp).from_args(args, Identity())

        self.actor = actor
        self.exp = exp
        self.nb_step = args.nb_step
        self.env_mgr = env_mgr
        self.nb_env = args.nb_env
        self.network = net.to(device)
        self.device = device
        self.initial_step_count = initial_step_count
        self.log_id_dir = log_id_dir
        self.epoch_len = args.epoch_len
        self.logger = logger
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.group = group

        if args.load_network:
            self.network = self.load_network(self.network, args.load_network)
            logger.info('Reloaded network from {}'.format(args.load_network))
        if args.load_optim:
            self.optimizer = self.load_optim(self.optimizer, args.load_optim)
            logger.info('Reloaded optimizer from {}'.format(args.load_optim))

        self.network.eval()

    def run(self):
        step_count = self.initial_step_count
        ep_rewards = torch.zeros(self.nb_env)
        future = None

        obs = dtensor_to_dev(self.env_mgr.reset(), self.device)
        internals = listd_to_dlist([
            self.network.new_internals(self.device) for _ in
            range(self.nb_env)
        ])
        start_time = time()
        dist.barrier()
        while step_count * (self.world_size - 1) < self.nb_step:  # is this okay?
            with torch.no_grad():
                actions, exp, internals = self.actor.act(self.network, obs, internals)
            self.exp.write_actor(exp)

            next_obs, rewards, terminals, infos = self.env_mgr.step(actions)
            next_obs = dtensor_to_dev(next_obs, self.device)
            self.exp.write_env(
                obs,
                rewards.to(self.device),
                terminals.to(self.device),
                infos
            )

            # Perform state updates
            step_count += self.nb_env
            ep_rewards += rewards.float()
            obs = next_obs

            term_rewards = []
            for i, terminal in enumerate(terminals):
                if terminal:
                    for k, v in self.network.new_internals(self.device).items():
                        internals[k][i] = v
                    term_rewards.append(ep_rewards[i].item())
                    ep_rewards[i].zero_()

            if term_rewards:
                term_reward = np.mean(term_rewards)
                delta_t = time() - start_time
                self.logger.info(
                    'RANK: {} '
                    'LOCAL STEP: {} '
                    'REWARD: {} '
                    'LOCAL STEP/S: {}'.format(
                        self.global_rank,
                        step_count,
                        term_reward,
                        (step_count - self.initial_step_count) / delta_t
                    )
                )

            if self.exp.is_ready():
                if future is not None:
                    future.wait()
                self.exp.sync(self.local_rank, self.group)
                future = dist.barrier(self.group, async_op=True)

    @staticmethod
    def _device(local_rank):
        if local_rank == 0:
            return 0
        else:
            return (local_rank % (torch.cuda.device_count() - 1)) + 1

    def close(self):
        self.env_mgr.close()
