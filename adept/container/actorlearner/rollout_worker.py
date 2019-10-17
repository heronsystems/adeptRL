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
from collections import namedtuple, deque
from time import time

import numpy as np
import ray
import torch

from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.utils.util import dtensor_to_dev, listd_to_dlist

from adept.container.base import Container


class ActorLearnerWorker(Container):
    @classmethod
    def as_remote(cls,
                  num_cpus=None,
                  num_gpus=None,
                  memory=None,
                  object_store_memory=None,
                  resources=None):
        # Worker can't use more than 1 gpu, but can also be cpu only
        assert num_gpus is None or num_gpus <= 1
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources)(cls)

    def __init__(
            self,
            args,
            log_id_dir,
            initial_step_count,
            global_rank
    ):
        seed = args.seed \
            if global_rank == 0 \
            else args.seed + args.nb_env * global_rank
        print('Worker {} using seed {}'.format(global_rank, seed))

        # load saved registry classes
        REGISTRY.load_extern_classes(log_id_dir)

        # ENV
        engine = REGISTRY.lookup_engine(args.env)
        env_cls = REGISTRY.lookup_env(args.env)
        env_mgr = SubProcEnvManager.from_args(args, engine, env_cls, seed=seed)

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device(
            "cuda"
            if (torch.cuda.is_available())
            else "cpu"
        )
        output_space = REGISTRY.lookup_output_space(
            args.actor_worker, env_mgr.action_space
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
        exp = REGISTRY.lookup_exp(args.exp).from_args(args, builder)

        self.actor = actor
        self.exp = exp.to(device)
        self.nb_step = args.nb_step
        self.env_mgr = env_mgr
        self.nb_env = args.nb_env
        self.network = net.to(device)
        self.device = device
        self.initial_step_count = initial_step_count

        # TODO: this should be set to eval after some number of training steps
        self.network.train()

        # SETUP state variables for run
        self.step_count = self.initial_step_count
        self.global_step_count = self.initial_step_count
        self.prev_step_t = time()
        self.ep_rewards = torch.zeros(self.nb_env)
        self.rolling_ep_rewards = deque(maxlen=100)
        self.global_rank = global_rank
        self.summary_freq = args.summary_freq
        self.prev_step_t = time()

        self.obs = dtensor_to_dev(self.env_mgr.reset(), self.device)
        self.internals = listd_to_dlist([
            self.network.new_internals(self.device) for _ in
            range(self.nb_env)
        ])
        self.start_time = time()
        self._weights_synced = False

    def run(self):
        if not self._weights_synced:
            raise Exception("Must set weights before calling run")

        self.exp.clear()
        all_terminal_rewards = []

        # loop to generate a rollout
        with torch.no_grad():
            while not self.exp.is_ready():
                actions, exp, self.internals = self.actor.act(self.network, self.obs, self.internals)

                self.exp.write_actor(exp)

                next_obs, rewards, terminals, infos = self.env_mgr.step(actions)
                next_obs = dtensor_to_dev(next_obs, self.device)
                self.exp.write_env(
                    self.obs,
                    rewards.to(self.device),  # TODO: make configurable?
                    terminals.to(self.device),  # TODO: make configurable?
                    infos
                )

                # Perform state updates
                self.step_count += self.nb_env
                self.ep_rewards += rewards.float()
                self.obs = next_obs

                term_rewards = []
                for i, terminal in enumerate(terminals):
                    if terminal:
                        for k, v in self.network.new_internals(self.device).items():
                            self.internals[k][i] = v
                        rew = self.ep_rewards[i].item()
                        term_rewards.append(rew)
                        self.rolling_ep_rewards.append(rew)
                        self.ep_rewards[i].zero_()

                # avg rewards
                if term_rewards:
                    term_reward = np.mean(term_rewards)
                    all_terminal_rewards.append(term_reward)

                    # print metrics
                    cur_step_t = time()
                    if cur_step_t - self.prev_step_t > self.summary_freq:
                        delta_t = time() - self.start_time
                        print(
                            'RANK: {} '
                            'LOCAL STEP: {} '
                            'REWARD: {} '
                            'AVG REWARD: {} '
                            'LOCAL STEP/S: {:.2f}'.format(
                                self.global_rank,
                                self.step_count,
                                term_reward,
                                round(np.mean(self.rolling_ep_rewards), 2),
                                (self.step_count - self.initial_step_count) / delta_t
                            )
                        )
                        self.prev_step_t = cur_step_t

        # rollout is full return it
        self.exp.write_next_obs(self.obs)
        # TODO: compression?
        if len(all_terminal_rewards) > 0:
            return {'rollout': self._ray_pack(self.exp), 'terminal_rewards': np.mean(all_terminal_rewards)}
        else:
            return {'rollout': self._ray_pack(self.exp), 'terminal_rewards': None}

    def set_weights(self, weights):
        for w, local_w in zip(weights, self.get_parameters()):
            # use data to ignore weights requiring grads
            local_w.data.copy_(w, non_blocking=True)
        self._weights_synced = True

    def set_global_step(self, global_step_count):
        self.global_step_count = global_step_count

    def get_parameters(self):
        params = [p for p in self.network.parameters()]
        params.extend([b for b in self.network.buffers()])
        return params

    def close(self):
        return self.env_mgr.close()

    def _ray_pack(self, exp):
        on_cpu = {k: self._to_cpu(v) for k, v in exp.items()}
        return on_cpu

    def _to_cpu(self, var):
        # TODO: this is a hack, should instead register a custom serializer for torch tensors to go
        # to CPU
        if isinstance(var, list):
            # list of dict -> dict of lists
            # observations/actions/internals
            if isinstance(var[0], dict):
                # if empty dict it doesn't matter
                if len(var[0]) == 0:
                    return {}
                first_v = next(iter(var[0].values()))
                # observations/actions
                if isinstance(first_v, torch.Tensor):
                    return {k: torch.stack(v).cpu() for k, v in listd_to_dlist(var).items()}
                # internals
                elif isinstance(first_v, list):
                    # TODO: there's gotta be a better way to do this
                    assert len(var) == 1
                    return {k: torch.stack(v).cpu().unsqueeze(0) for k, v in var[0].items()}
            # other actor stuff
            elif isinstance(var[0], torch.Tensor):
                return torch.stack(var).cpu()
            else:
                raise NotImplementedError("Expected rollout item to be a Tensor or dict(Tensors) got {}".format(type(var[0])))
        elif isinstance(var, dict):
            # next obs
            if isinstance(first_v, torch.Tensor):
                return {k: v.cpu() for k, v in var.items()}
            else:
                raise NotImplementedError("Expected rollout dict item to be a tensor got {}".format(type(var)))
        else:
            raise NotImplementedError("Expected rollout object to be a list got {}".format(type(var)))

