import ray
from torch import distributed as dist
from itertools import cycle
import torch


@ray.remote(num_gpus=0.25)
class Host:
    def __init__(self, rank, port, world_size):
        dist.init_process_group(
            'nccl',
            init_method='tcp://127.0.0.1:{}'.format(port),
            rank=rank,
            world_size=world_size
        )
        self.learner_group = None
        self.worker_group = None

    def hello(self):
        print('hello')


@ray.remote(num_gpus=0.25)
class Worker:
    def __init__(self, rank, port, world_size):
        dist.init_process_group(
            'nccl',
            init_method='tcp://127.0.0.1:{}'.format(port),
            rank=rank,
            world_size=world_size
        )
        self.cur_exp = None
        self.worker_group = None

    def hello(self, name):
        print('hello', name)

    def compute_exp(self):
        self.cur_exp = torch.ones()

    def sync(self):
        pass


if __name__ == '__main__':
    nb_host = 2
    nb_worker = 2
    port = 6009
    world_size = nb_host + nb_worker
    names = ['joe', 'ben']

    ray.init(num_gpus=1)

    # instantiate as many hosts/workers as requested
    remote_hosts = [
        Host.remote(rank, port, world_size) for rank in range(nb_host)
    ]
    host_gen = cycle(range(nb_host))

    remote_workers = [
        Worker.remote(rank + nb_host, port, world_size) for rank in range(nb_worker)
    ]

    # all workers compute exp
    # host wait for n to be ready
    # when ready, merge batch and learn
    step_count = 0
    queue = []

    for h in remote_hosts:
        ray.get(h.hello.remote())

    for h, name in zip(remote_workers, names):
        ray.get(h.hello.remote(name))
