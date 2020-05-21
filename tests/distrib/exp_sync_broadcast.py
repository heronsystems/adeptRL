import os
from itertools import chain

import torch
import torch.distributed as dist

WORLD_SIZE = int(os.environ["WORLD_SIZE"])
GLOBAL_RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
NB_NODE = int(os.environ["NB_NODE"])
LOCAL_SIZE = WORLD_SIZE // NB_NODE

print("w", WORLD_SIZE)
print("g", GLOBAL_RANK)
print("l", LOCAL_RANK)
print("n", NB_NODE)


cache_spec = {"xs": (2, 6, 3, 3), "ys": (2, 6, 12), "rewards": (2, 6, 16)}


def gpu_id(local_rank, device_count):
    if local_rank == 0:
        return 0
    elif device_count == 1:
        return 0
    else:
        return (local_rank % (device_count - 1)) + 1


class WorkerCache(dict):
    def __init__(self, cache_spec, gpu_id):
        super(WorkerCache, self).__init__()
        self.sorted_keys = sorted(cache_spec.keys())
        self.gpu_id = gpu_id

        for k in self.sorted_keys:
            self[k] = self._init_rollout(cache_spec, k)

    def _init_rollout(self, spec, key):
        return [
            torch.ones(*spec[key][1:]).to(f"cuda:{self.gpu_id}")
            for _ in range(spec[key][0])
        ]

    def sync(self, src, grp):
        handles = []
        for k in self.sorted_keys:
            for t in self[k]:
                handles.append(
                    dist.broadcast(t, src=src, group=grp, async_op=True)
                )
        return handles

    def iter_tensors(self):
        return chain(*[self[k] for k in self.sorted_keys])


class HostCache(WorkerCache):
    def _init_rollout(self, spec, key):
        return [
            torch.zeros(*spec[key][1:]).to(f"cuda:{self.gpu_id}")
            for _ in range(spec[key][0])
        ]


def on_worker():
    return LOCAL_RANK != 0


def on_host():
    return LOCAL_RANK == 0


if __name__ == "__main__":
    nb_gpu = torch.cuda.device_count()
    print("Device Count", nb_gpu)

    dist.init_process_group(
        backend="nccl", world_size=WORLD_SIZE, rank=LOCAL_RANK
    )

    groups = []
    for i in range(1, LOCAL_SIZE):
        grp = [0, i]
        groups.append(dist.new_group(grp))

    print("LOCAL_RANK", LOCAL_RANK, "initialized.")
    if on_worker():
        cache = WorkerCache(cache_spec, gpu_id(LOCAL_RANK, nb_gpu))
        [t.fill_(LOCAL_RANK) for t in cache.iter_tensors()]
    else:
        caches = [
            HostCache(cache_spec, gpu_id(LOCAL_RANK, nb_gpu))
            for _ in range(LOCAL_SIZE - 1)
        ]

    # tags to identify tensors
    # loop thru workers

    if on_worker():
        handle = cache.sync(LOCAL_RANK, groups[LOCAL_RANK - 1])
    else:
        handles = []
        for i, cache in enumerate(caches):
            handles.append(cache.sync(i + 1, groups[i]))

    if on_worker():
        [h.wait() for h in handle]
    else:
        for handle in handles:
            [h.wait() for h in handle]

    if on_host():
        for cache in caches:
            for t in cache.iter_tensors():
                print(t)
