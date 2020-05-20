import random
from collections import namedtuple

import ray
from torch import distributed as dist
import torch
import time
from itertools import chain


@ray.remote(num_gpus=0.25)
class Learner:
    def __init__(self, rank, learner_ranks, worker_ranks, ip, port):
        world_size = len(learner_ranks) + len(worker_ranks)

        dist.init_process_group(
            "nccl",
            init_method="tcp://{}:{}".format(ip, port),
            rank=rank,
            world_size=world_size,
        )
        groups = {}
        for learner_rank in learner_ranks:
            for worker_rank in worker_ranks:
                g = dist.new_group([learner_rank, worker_rank])
                if learner_rank == rank:
                    groups[worker_rank] = g
        learner_group = dist.new_group(learner_ranks)

        self.groups = groups
        self.learner_group = learner_group
        self.device = torch.device(f"cuda:{ray.get_gpu_ids()[0]}")
        self.rank = rank

        self.exps = {
            w_rank: torch.zeros(2).to(self.device) for w_rank in worker_ranks
        }
        self.network = torch.ones(3).to(self.device)
        self.network_grads = [torch.ones(3).to(self.device)]
        self.exp_handles = None

    def sync_exp(self, worker_rank):
        handle = dist.broadcast(
            self.exps[worker_rank],
            worker_rank,
            self.groups[worker_rank],
            async_op=True,
        )
        return handle

    def sync_exps(self, worker_ranks):
        print(f"learner {self.rank} syncing exps from {worker_ranks}")
        handles = []
        for worker_rank in worker_ranks:
            h = self.sync_exp(worker_rank)
            handles.append(h)
        self.exp_handles = handles
        return self.rank

    def sync_network(self, worker_ranks):
        print(f"learner {self.rank} sending network to {worker_ranks}")
        for worker_rank in worker_ranks:
            dist.broadcast(
                self.network, self.rank, self.groups[worker_rank], async_op=True
            )
        return self.rank

    def step(self):
        print(f"learner {self.rank} step")

        # make sure exp_handles are done
        for handle in self.exp_handles:
            handle.wait()

        # batch together exp
        time.sleep(random.randint(0, 3))

        # update with other learners
        dist.barrier(self.learner_group)
        for p in self.network_grads:
            dist.all_reduce(p, group=self.learner_group)
        print(f"learner {self.rank} shared gradients")
        return True


@ray.remote(num_gpus=0.25)
class Worker:
    def __init__(self, rank, learner_ranks, worker_ranks, ip, port):
        world_size = len(learner_ranks) + len(worker_ranks)
        dist.init_process_group(
            "nccl",
            init_method="tcp://{}:{}".format(ip, port),
            rank=rank,
            world_size=world_size,
        )
        groups = {}
        for learner_rank in learner_ranks:
            for worker_rank in worker_ranks:
                g = dist.new_group([learner_rank, worker_rank])
                if worker_rank == rank:
                    groups[learner_rank] = g
        dist.new_group(learner_ranks)

        self.groups = groups
        self.device = torch.device(f"cuda:{ray.get_gpu_ids()[0]}")
        self.rank = rank
        self.network = torch.zeros(3).to(self.device)
        self.exp = None
        self.network_handle = None

    def step(self):
        print(f"worker {self.rank} stepping")
        # block if a network is copied
        if self.network_handle:
            self.network_handle.wait()
        print(f"worker {self.rank} network {self.network}")

        self.exp = torch.zeros(2).to(self.device)
        self.exp.fill_(self.rank)
        return self.rank

    def sync_exp(self, host_rank):
        handle = dist.broadcast(
            self.exp, self.rank, self.groups[host_rank], async_op=True
        )
        return self.rank

    def sync_network(self, host_rank):
        handle = dist.broadcast(
            self.network, host_rank, self.groups[host_rank], async_op=True
        )
        return self.rank


def flatten(items):
    return chain.from_iterable(items)


if __name__ == "__main__":
    nb_host = 2
    nb_worker = 2
    ip = "127.0.0.1"
    port = 6009
    nb_step = 5

    ray.init(num_gpus=1)

    learner_ranks = list(range(nb_host))
    worker_ranks = list(range(nb_host, nb_host + nb_worker))

    # instantiate as many hosts/workers as requested
    learners = [
        Learner.remote(rank, learner_ranks, worker_ranks, ip, port)
        for rank in learner_ranks
    ]
    workers = {
        rank: Worker.remote(rank, learner_ranks, worker_ranks, ip, port)
        for rank in worker_ranks
    }

    # all workers compute exp
    undone_ranks = [worker.step.remote() for worker in workers.values()]

    for step in range(nb_step):
        l_steps = []

        all_w_ranks = []
        for l_rank, learner in enumerate(learners):
            # host wait for n to be ready
            w_ranks, undone_ranks = ray.wait(undone_ranks, num_returns=1)
            w_ranks = [ray.get(rank) for rank in w_ranks]
            learner_sync = learner.sync_exps.remote(w_ranks)
            for rank in w_ranks:
                worker_sync = workers[rank].sync_exp.remote(l_rank)
            # when ready, merge batch and learn
            l_steps.append(learner.step.remote())
            all_w_ranks.append(w_ranks)

        # sync learns
        # wait for all learners to step and sync
        [ray.get(l_step) for l_step in l_steps]

        # sync networks
        for w_ranks, l_rank, learner in zip(
            all_w_ranks, range(len(learners)), learners
        ):
            learner_sync = learner.sync_network.remote(w_ranks)
            for rank in w_ranks:
                worker_sync = workers[rank].sync_network.remote(l_rank)

        # unblock workers
        for rank in flatten(all_w_ranks):
            undone_ranks.append(workers[rank].step.remote())
