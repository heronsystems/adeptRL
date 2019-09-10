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
from collections import namedtuple
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
        self.global_rank = global_rank

        # ENV
        engine = REGISTRY.lookup_engine(args.env)
        env_cls = REGISTRY.lookup_env(args.env)
        env_mgr = SubProcEnvManager.from_args(args, engine, env_cls, seed=seed)

        # NETWORK
        torch.manual_seed(args.seed)
        # TODO: cluster config to get gpu id
        device = torch.device(
            "cuda"
            if (torch.cuda.is_available())
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
        with torch.no_grad():
            for _ in range(self.rollout_len):
                actions, exp, self.internals = self.actor.act(self.network, self.obs, self.internals)

                self.exp.write_actor(exp)

                # need to copy the obs since zmq returns a pointer
                self.obs = {k: v.clone() for k, v in self.obs.items()}

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
        # TODO: compression?
        return self._ray_pack(self.exp)

    def set_weights(self, weights):
        for w, local_w in zip(weights, self.get_parameters()):
            # use data to ignore weights requiring grads
            local_w.data.copy_(w, non_blocking=True)
        self._weights_synced = True

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
        if isinstance(var, list):
            # list of dict -> dict of lists
            if isinstance(var[0], dict):
                return {k: torch.stack(v).cpu() for k, v in listd_to_dlist(var).items()}
            elif isinstance(var[0], torch.Tensor):
                return torch.stack(var).cpu()
            else:
                raise NotImplementedError("Expected rollout item to be a Tensor or dict(Tensors) got {}".format(type(var[0])))
        elif isinstance(var, dict):
            return {k: v.cpu() for k, v in var.items()}
        else:
            raise NotImplementedError("Expected rollout object to be a list got {}".format(type(var)))


if __name__ == '__main__':
    from adept.scripts.actorlearner import parse_args
    from adept.container import Init
    args = parse_args()
    args.nb_env = 8
    args.gpu_id = 0
    args = Init.from_defaults(args)

    # ray init
    ray.init()

    # create some number of remotes
    num_work = 4
    workers = [RolloutWorker.as_remote(num_cpus=12/num_work, num_gpus=0).remote(args, 0, w_ind)
               for w_ind in range(num_work)]

    # synchronize weights
    futures = [w.set_weights.remote([]) for w in workers]
    ray.get(futures)

    # get batches directly, just to make sure it works
    futures = [w.run.remote() for w in workers]
    rollouts = ray.get(futures)
    print('got batches manually')
    batch = {}
    # TODO: this assumes all rollouts have the same keys
    for k in rollouts[0].keys():
        # cat over batch dimension
        if isinstance(rollouts[0][k], torch.Tensor):
            v_list = [r[k] for r in rollouts]
            agg = torch.cat(v_list, dim=1)
        elif isinstance(rollouts[0][k], dict):
            # cat all elements of dict
            agg = {}
            for r_key in rollouts[0][k].keys():
                agg[r_key] = torch.cat([r[k][r_key] for r in rollouts], dim=1)
        batch[k] = agg

    print(batch['states']['Box'].shape)


    # setup queuer
    from rollout_queuer import RolloutQueuerAsync
    import time
    manager = RolloutQueuerAsync(workers, 2, 4)
    manager.start()

    # get batch from queuer this is blocking
    st = time.time()
    for i in range(1000):
        batches = manager.get()
        print('got batches')
    et = time.time()
    print('bps', 1000 / (et - st))

    manager.stop()



