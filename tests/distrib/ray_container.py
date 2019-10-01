import ray
import torch
# shared state
# step count
# queue

# main script will instantiate as many hosts/workers as requested
# all workers compute exp
# host wait for n to be ready
# when ready, merge batch and learn
@ray.remote(num_gpus=0.25)
class Host:
    def __init__(self):
        self.stuff = 'stuff'
        print(ray.get_gpu_ids())

    def hello(self):
        print('hello')


@ray.remote(num_gpus=0.25)
class Worker:
    def __init__(self):
        print(ray.get_gpu_ids())
        self.stuff = 'stuff'

    def hello(self, name):
        print('hello', name)


if __name__ == '__main__':
    nb_host = 2
    names = ['joe', 'ben']

    ray.init(num_gpus=1)
    print(ray.get_gpu_ids())
    remote_hosts = [Host.remote() for _ in range(nb_host)]

    remote_workers = [Worker.remote() for _ in range(nb_host)]

    for h in remote_hosts:
        ray.get(h.hello.remote())

    for h, name in zip(remote_workers, names):
        ray.get(h.hello.remote(name))
