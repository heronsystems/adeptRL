import os
from itertools import chain

import torch
import torch.distributed as dist

WORLD_SIZE = int(os.environ['WORLD_SIZE'])
GLOBAL_RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
NB_NODE = int(os.environ['NB_NODE'])
LOCAL_SIZE = WORLD_SIZE // NB_NODE

print('w', WORLD_SIZE)
print('g', GLOBAL_RANK)
print('l', LOCAL_RANK)
print('n', NB_NODE)


cache_spec = {
    'xs': (2, 6, 3, 3),
    'ys': (2, 6, 12),
    'rewards': (2, 6, 16)
}


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
        return [torch.ones(*spec[key][1:]).to(f'cuda:{self.gpu_id}') for _ in range(spec[key][0])]

    def sync(self, src, grp):
        handles = []
        for k in self.sorted_keys:
            for t in self[k]:
                handles.append(dist.broadcast(t, src=src, group=grp, async_op=True))
        return handles

    def iter_tensors(self):
        return chain(
            *[self[k] for k in self.sorted_keys]
        )


class HostCache(WorkerCache):
    def _init_rollout(self, spec, key):
        return [torch.zeros(*spec[key][1:]).to(f'cuda:{self.gpu_id}') for _ in
                range(spec[key][0])]


class HostContainer:
    def __init__(
            self,
            local_rank,
            global_rank,
            world_size,
            groups
    ):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.groups = groups
        self.exps = [
            HostCache(
                cache_spec, gpu_id(local_rank, torch.cuda.device_count())
            ) for _ in range(len(groups))
        ]

    def run(self):
        step_count = 0
        dist.barrier()
        handles = []
        for i, exp in enumerate(self.exps):
            handles.append(exp.sync(i + 1, self.groups[i]))

        while step_count < 10:
            # wait for n batches to sync
                # this needs to run in a thread
            learn_indices = []
            while len(learn_indices) < 2:
                for i, hand in enumerate(handles):
                    if all([h.is_completed() for h in hand]):
                        learn_indices.append(i)
            
            print('syncing', learn_indices)
            # merge tensors along batch dimension
            # need to pick a data structure for experiences
            # Dict[str,List[Tensor]]
            merged_exp = {}
            keys = self.exps[learn_indices[0]].sorted_keys  # cache these or from spec
            lens = [len(self.exps[learn_indices[0]][k]) for k in keys]  # cache these
            for k, l in zip(keys, lens):
                for j in range(l):
                    tensors_to_cat = []
                    for i in learn_indices:
                        exp = self.exps[i]
                        tensors_to_cat.append(exp[k][j])

                    cat = torch.cat(tensors_to_cat)
                    if k in merged_exp:
                        merged_exp[k].append(cat)
                    else:
                        merged_exp[k] = [cat]

            # unblock the selected workers
            # resync
            for i in learn_indices:
                dist.barrier(self.groups[i])
                handles[i] = self.exps[i].sync(i + 1, self.groups[i])

            step_count += 1


class WorkerContainer:
    def __init__(
            self,
            local_rank,
            global_rank,
            world_size,
            group
    ):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.group = group
        self.exp = WorkerCache(
            cache_spec, gpu_id(local_rank, torch.cuda.device_count())
        )

    def run(self):
        step_count = 0
        future = None
        dist.barrier()
        while step_count * (self.world_size - 1) < 10:
            if future is not None:
                self.exp.sync(self.local_rank, self.group)
            future = dist.barrier(self.group, async_op=True)
            step_count += 1


def on_worker():
    return LOCAL_RANK != 0


def on_host():
    return LOCAL_RANK == 0


if __name__ == '__main__':
    nb_gpu = torch.cuda.device_count()
    print('Device Count', nb_gpu)

    dist.init_process_group(
        backend='nccl',
        world_size=WORLD_SIZE,
        rank=LOCAL_RANK
    )

    groups = []
    for i in range(1, LOCAL_SIZE):
        grp = [0, i]
        groups.append(dist.new_group(grp))

    print('LOCAL_RANK', LOCAL_RANK, 'initialized.')
    if LOCAL_RANK == 0:
        container = WorkerContainer(LOCAL_RANK, GLOBAL_RANK, WORLD_SIZE, groups[LOCAL_RANK - 1])
    else:
        container = HostContainer(LOCAL_RANK, GLOBAL_RANK, WORLD_SIZE, groups)

    container.run()
