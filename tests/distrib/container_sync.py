import os
import time
from itertools import chain
from collections import deque
from glob import glob

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
    def __init__(self, cache_spec, device):
        super(WorkerCache, self).__init__()
        self.sorted_keys = sorted(cache_spec.keys())
        self.device = device

        for k in self.sorted_keys:
            self[k] = self._init_rollout(cache_spec, k)

    def _init_rollout(self, spec, key):
        return [
            torch.ones(*spec[key][1:]).to(self.device)
            for _ in range(spec[key][0])
        ]

    def sync(self, src, grp, is_async=True):
        handles = []
        for k in self.sorted_keys:
            for t in self[k]:
                handles.append(
                    dist.broadcast(t, src=src, group=grp, async_op=is_async)
                )
        return handles

    def iter_tensors(self):
        return chain(*[self[k] for k in self.sorted_keys])


class HostCache(WorkerCache):
    def _init_rollout(self, spec, key):
        return [
            torch.zeros(*spec[key][1:]).to(self.device)
            for _ in range(spec[key][0])
        ]


class HostContainer:
    def __init__(self, local_rank, global_rank, world_size, groups):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.groups = groups
        self.device = torch.device(
            "cuda:{}".format(gpu_id(local_rank, torch.cuda.device_count()))
        )
        self.exps = [
            HostCache(cache_spec, self.device) for _ in range(len(groups))
        ]
        os.makedirs("/tmp/actorlearner", exist_ok=True)
        if os.path.exists("/tmp/actorlearner/done"):
            os.rmdir("/tmp/actorlearner/done")

    def run(self):
        step_count = 0
        nb_batch = 2
        dist.barrier()
        handles = []

        for i, exp in enumerate(self.exps):
            handles.append(exp.sync(i + 1, self.groups[i]))

        while step_count < 10:
            # wait for n batches to sync
            # this needs to run in a thread

            if nb_batch > len(self.groups):
                print("warn")
                nb_batch = len(self.groups)

            q, q_lookup = deque(), set()
            while len(q) < nb_batch:
                for i, hand in enumerate(handles):
                    if i not in q_lookup:
                        if all([h.is_completed() for h in hand]):
                            q.append(i)
                            q_lookup.add(i)

            print("synced", [i + 1 for i in q], "merging...")
            # merge tensors along batch dimension
            # need to pick a data structure for experiences
            # Dict[str,List[Tensor]]
            merged_exp = {}
            keys = self.exps[q[0]].sorted_keys  # cache these or from spec
            lens = [len(self.exps[q[0]][k]) for k in keys]  # cache these
            for k, l in zip(keys, lens):
                for j in range(l):
                    tensors_to_cat = []
                    for i in q:
                        exp = self.exps[i]
                        tensors_to_cat.append(exp[k][j])

                    cat = torch.cat(tensors_to_cat)
                    if k in merged_exp:
                        merged_exp[k].append(cat)
                    else:
                        merged_exp[k] = [cat]

            # unblock the selected workers
            # resync
            for i in q:
                print(f"HOST barrier {i+1}...")
                dist.barrier(self.groups[i])
                print(f"HOST sync {i + 1}...")
                handles[i] = self.exps[i].sync(i + 1, self.groups[i])

            step_count += 1
        os.mkdir("/tmp/actorlearner/done")


class WorkerContainer:
    def __init__(self, local_rank, global_rank, world_size, group):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.group = group
        self.device = torch.device(
            "cuda:{}".format(gpu_id(local_rank, torch.cuda.device_count()))
        )
        self.exp = WorkerCache(cache_spec, self.device)

    def run(self):
        is_done = False
        first = True

        while not is_done:
            if first:
                dist.barrier()
                self.exp.sync(self.local_rank, self.group, is_async=False)
                first = False
            else:
                print(f"WORKER barrier {self.local_rank}")
                future = dist.barrier(self.group, async_op=True)

                while not future.is_completed():
                    if glob("/tmp/actorlearner/done"):
                        print(f"complete - exiting worker {self.local_rank}")
                        is_done = True
                        break

                if not is_done:
                    print(f"WORKER sync {self.local_rank}")
                    self.exp.sync(self.local_rank, self.group, is_async=True)


def on_worker():
    return LOCAL_RANK != 0


def on_host():
    return LOCAL_RANK == 0


if __name__ == "__main__":
    dist.init_process_group(
        backend="nccl", world_size=WORLD_SIZE, rank=LOCAL_RANK
    )

    groups = []
    for i in range(1, LOCAL_SIZE):
        grp = [0, i]
        groups.append(dist.new_group(grp))

    print("LOCAL_RANK", LOCAL_RANK, "initialized.")
    if LOCAL_RANK == 0:
        container = HostContainer(LOCAL_RANK, GLOBAL_RANK, WORLD_SIZE, groups)
    else:
        container = WorkerContainer(
            LOCAL_RANK, GLOBAL_RANK, WORLD_SIZE, groups[LOCAL_RANK - 1]
        )

    container.run()
