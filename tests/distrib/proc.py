import os
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


def on_worker():
    return LOCAL_RANK != 0


def on_host():
    return LOCAL_RANK == 0


if __name__ == '__main__':
    dist.init_process_group(
        backend='gloo',
        world_size=WORLD_SIZE,
        rank=LOCAL_RANK
    )
    print('LOCAL_RANK', LOCAL_RANK, 'initialized.')
    # groups = assign_groups()
    if on_worker():
        t = torch.Tensor([LOCAL_RANK])
    else:
        ts = [torch.Tensor([LOCAL_RANK]) for _ in range(LOCAL_SIZE - 1)]

    # tags to identify tensors
    # loop thru workers

    if on_worker():
        handle = dist.isend(t, 0)
    else:
        handles = []
        for i in range(1, LOCAL_SIZE):
            handle = dist.irecv(ts[i - 1], i)
        for handle in handles:
            handle.wait()
        print(ts)

    # dist.broadcast_multigpu([t], src=LOCAL_RANK, group=groups[GLOBAL_RANK])
