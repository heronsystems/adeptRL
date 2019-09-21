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
from collections import deque
from glob import glob
from itertools import chain
import random
from time import time

import numpy as np
import torch
from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.rewardnorm import Identity
from adept.utils import listd_to_dlist, dtensor_to_dev
from adept.utils.logging import SimpleModelSaver
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter

from .base import Container


def gpu_id(local_rank, gpu_ids):
    device_count = len(gpu_ids)
    if local_rank == 0:
        return gpu_ids[0]
    elif device_count == 1:
        return gpu_ids[0]
    else:
        gpu_idx = (local_rank % (device_count - 1)) + 1
        return gpu_ids[gpu_idx]


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
        env = env_cls.from_args(args, 0)
        env.close()

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device("cuda:{}".format(
            gpu_id(local_rank, args.gpu_ids))
        )
        output_space = REGISTRY.lookup_output_space(
            args.actor_host, env.action_space
        )
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
            args.nb_env * args.nb_learn_batch
        )
        w_builder = REGISTRY.lookup_actor(args.actor_worker).exp_spec_builder(
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
        self.exp = exp_cls.from_args(args, rwd_norm, builder).to(device)
        self.worker_exps = [
            exp_cls.from_args(args, Identity(), w_builder).to(device)
            for _ in range(len(groups))
        ]
        self.batch_size = args.nb_env * args.nb_learn_batch
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

        os.makedirs('/tmp/actorlearner', exist_ok=True)
        if os.path.exists('/tmp/actorlearner/done'):
            os.rmdir('/tmp/actorlearner/done')

        self.network.train()

    def run(self):
        from pyinstrument import Profiler

        if self.nb_learn_batch > len(self.groups):
            self.logger.warn('More learn batches than workers, reducing '
                             'learn batches to {}'.format(len(self.groups)))
            self.nb_learn_batch = len(self.groups)

        step_count = self.initial_step_count
        next_save = self.init_next_save(self.initial_step_count, self.epoch_len)
        prev_step_t = time()
        ep_rewards = torch.zeros(self.batch_size)

        # dist.barrier()
        # self.network.sync(0)

        dist.barrier()
        e_handles = []
        for i, exp in enumerate(self.worker_exps):
            w_local_rank = i + 1
            e_handles.append(exp.sync(w_local_rank, self.groups[i], async_op=True))

        start_time = time()
        # profiler = Profiler()
        # self.nb_step = 10e3
        # profiler.start()
        while step_count < self.nb_step:
            # check the queue for finished rollouts
            q, q_lookup = deque(), set()
            while len(q) < self.nb_learn_batch:
                i = random.randint(0, len(self.groups) - 1)
                handles = e_handles[i]
                if i not in q_lookup and all([h.is_completed() for h in handles]):
                    q.append(i)
                    q_lookup.add(i)

            print(f'HOST syncing {[i+1 for i in q]}')

            # real slow
            self.exp.write_exps([self.worker_exps[i] for i in q])

            r = self.exp.read()
            internals = {k: t.unbind(0) for k, t in r.internals[0].items()}
            for obs, rewards, terminals in zip(
                    r.observations,
                    r.rewards,
                    r.terminals
            ):
                _, h_exp, internals = self.actor.act(self.network, obs,
                                                     internals)
                self.exp.write_actor(h_exp, no_env=True)

                # Perform state updates
                step_count += self.batch_size
                ep_rewards += rewards.cpu().float()

                term_rewards = []
                for i, terminal in enumerate(terminals.cpu()):
                    if terminal:
                        for k, v in self.network.new_internals(
                                self.device).items():
                            internals[k][i] = v
                        term_rewards.append(ep_rewards[i].item())
                        ep_rewards[i].zero_()

                if term_rewards:
                    term_reward = np.mean(term_rewards)
                    delta_t = time() - start_time
                    self.logger.info(
                        'RANK: {} '
                        'STEP: {} '
                        # 'REWARD: {} '
                        'STEP/S: {}'.format(
                            self.local_rank,
                            step_count,
                            # term_reward,
                            (step_count - self.initial_step_count) / delta_t
                        )
                    )
                    self.summary_writer.add_scalar(
                        'reward', term_reward, step_count
                    )

                if step_count >= next_save:
                    self.saver.save_state_dicts(
                        self.network, step_count, self.optimizer
                    )
                    next_save += self.epoch_len

            # Backprop
            loss_dict, metric_dict = self.learner.compute_loss(
                self.network, self.exp.read(), r.next_observation, internals
            )
            total_loss = torch.sum(
                torch.stack(tuple(loss for loss in loss_dict.values()))
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.exp.clear()
            [exp.clear() for exp in self.worker_exps]  # necessary?

            # write summaries
            cur_step_t = time()
            if cur_step_t - prev_step_t > self.summary_freq:
                self.write_summaries(
                    self.summary_writer, step_count, total_loss,
                    loss_dict, metric_dict, self.network.named_parameters()
                )
                prev_step_t = cur_step_t

            for i in q:
                # update worker networks
                # self.network.sync(i + 1, self.groups[i], async_op=False)
                # unblock the selected workers
                dist.barrier(self.groups[i], async_op=True)
                self.network.sync(0, self.groups[i], async_op=True)
                e_handles[i] = self.worker_exps[i].sync(
                    i + 1,
                    self.groups[i],
                    async_op=True
                )
        # profiler.stop()
        # print(profiler.output_text(unicode=True, color=True))
        os.mkdir('/tmp/actorlearner/done')

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
        device = torch.device("cuda:{}".format(
            gpu_id(local_rank, args.gpu_ids))
        )
        output_space = REGISTRY.lookup_output_space(
            args.actor_host, env_mgr.action_space
        )
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
        actor_cls = REGISTRY.lookup_actor(args.actor_worker)
        actor = actor_cls.from_args(args, env_mgr.action_space)
        builder = actor_cls.exp_spec_builder(
            env_mgr.observation_space,
            env_mgr.action_space,
            net.internal_space(),
            env_mgr.nb_env
        )
        exp = REGISTRY.lookup_exp(args.exp).from_args(args, Identity(), builder)

        self.actor = actor
        self.exp = exp.to(device)
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

        self.network.train()

    def run(self):
        step_count = self.initial_step_count
        ep_rewards = torch.zeros(self.nb_env)
        is_done = False
        first = True

        handles = None

        obs = dtensor_to_dev(self.env_mgr.reset(), self.device)
        internals = listd_to_dlist([
            self.network.new_internals(self.device) for _ in
            range(self.nb_env)
        ])
        start_time = time()

        # dist.barrier()
        # self.network.sync(0)

        while not is_done:
            # Block until exp and network have been sync'ed
            if handles:
                for handle in handles:
                    handle.wait()

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
                # self.logger.info(
                #     'RANK: {} '
                #     'LOCAL STEP: {} '
                #     'REWARD: {} '
                    # 'LOCAL STEP/S: {}'.format(
                    #     self.global_rank,
                    #     step_count,
                    #     term_reward,
                    #     (step_count - self.initial_step_count) / delta_t
                    # )
                # )

            if self.exp.is_ready():
                self.exp.write_next_obs(obs)

                if first:
                    dist.barrier()
                    handles = self.exp.sync(self.local_rank, self.group, async_op=True)
                    first = False
                else:
                    future = dist.barrier(self.group, async_op=True)
                    while not future.is_completed():
                        if glob('/tmp/actorlearner/done'):
                            print(f'complete - exiting worker {self.local_rank}')
                            is_done = True
                            break

                    if not is_done:
                        n_handles = self.network.sync(0, self.group, async_op=True)
                        e_handles = self.exp.sync(self.local_rank, self.group, async_op=True)
                        handles = chain(n_handles, e_handles)

                self.exp.clear()

    def close(self):
        self.env_mgr.close()
