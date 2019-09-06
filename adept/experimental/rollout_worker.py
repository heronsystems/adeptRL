"""
Init:
sets up environment
sets up network(s)/actor(s)
set network weights from args
Resets env to be ready

Run:
run one rollout and put data into cache
return cache
"""
from time import time

import numpy as np
import ray
import torch

from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.utils.util import dtensor_to_dev, listd_to_dlist

from adept.container.base import Container


class RolloutWorker(Container):
    @classmethod
    def as_remote(cls,
                  num_cpus=None,
                  num_gpus=None,
                  memory=None,
                  object_store_memory=None,
                  resources=None):
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
            initial_step_count,
            global_rank
    ):
        seed = args.seed \
            if global_rank == 0 \
            else args.seed + args.nb_env * global_rank

        # ENV
        engine = REGISTRY.lookup_engine(args.env)
        env_cls = REGISTRY.lookup_env(args.env)
        env_mgr = SubProcEnvManager.from_args(args, engine, env_cls, seed=seed)

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device(
            "cuda:{}".format(args.gpu_id)
            if (torch.cuda.is_available() and args.gpu_id >= 0)
            else "cpu"
        )
        output_space = REGISTRY.lookup_output_space(
            args.actor_worker, env_mgr.action_space)
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
        rwd_norm = REGISTRY.lookup_reward_normalizer(
            args.rwd_norm).from_args(args)
        actor = REGISTRY.lookup_actor(args.actor_worker).from_args(
            args, env_mgr.action_space
        )
        exp = REGISTRY.lookup_exp(args.exp_worker).from_args(args, rwd_norm)

        self.actor = actor
        self.exp = exp
        self.nb_step = args.nb_step
        self.env_mgr = env_mgr
        self.nb_env = args.nb_env
        self.network = net.to(device)
        self.device = device
        self.initial_step_count = initial_step_count
        self.rollout_len = int(args.worker_rollout_len)

        self.network.eval()

        # SETUP state variables for run
        self.step_count = self.initial_step_count
        self.prev_step_t = time()
        self.ep_rewards = torch.zeros(self.nb_env)

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

        # loop to generate a rollout
        for _ in range(self.rollout_len):
            with torch.no_grad():
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
                    term_rewards.append(self.ep_rewards[i].item())
                    self.ep_rewards[i].zero_()

            if term_rewards:
                term_reward = np.mean(term_rewards)
                delta_t = time() - self.start_time
                print(
                    'RANK: {} '
                    'LOCAL STEP: {} '
                    'REWARD: {} '
                    'LOCAL STEP/S: {}'.format(
                        self.global_rank,
                        self.step_count,
                        term_reward,
                        (self.step_count - self.initial_step_count) / delta_t
                    )
                )

        # rollout is full return it
        self.exp.write_next_obs(self.obs)
        return self.exp.read()

        # TODO: pack/compress or something before sending 
        # return self.exp.pack()

    def set_weights(self, weights):
        for w, local_w in zip(weights, self.get_parameters()):
            local_w.copy_(w)
        self._weights_synced = True

    def get_parameters(self):
        return []

    def close(self):
        return self.env_mgr.close()


if __name__ == '__main__':
    from adept.scripts.actorlearner import parse_args
    from adept.container import Init
    args = parse_args()
    args.nb_env = 2
    args = Init.from_defaults(args)

    # ray init
    ray.init()

    # create some number of remotes
    num_work = 1
    workers = [RolloutWorker.as_remote(num_cpus=args.nb_env, num_gpus=0).remote(args, 0, w_ind)
               for w_ind in range(num_work)]

    # synchronize weights
    futures = [w.set_weights.remote([]) for w in workers]
    ray.get(futures)

    # get batches
    futures = [w.run.remote() for w in workers]
    batches = ray.get(futures)



    print(batches)
