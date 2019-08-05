import os
import torch
import torch.distributed as dist

WORLD_SIZE = int(os.environ['WORLD_SIZE'])
GLOBAL_RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
NB_NODE = int(os.environ['NB_NODE'])

print('w', WORLD_SIZE)
print('g', GLOBAL_RANK)
print('l', LOCAL_RANK)
print('n', NB_NODE)


if __name__ == '__main__':
    dist.init_process_group(
        backend='gloo',
        world_size=WORLD_SIZE,
        rank=LOCAL_RANK
    )
    print('LOCAL_RANK', LOCAL_RANK, 'initialized.')
    # groups = assign_groups()
    t = torch.Tensor([LOCAL_RANK])
    if LOCAL_RANK == 0:
        handle = dist.irecv(t, 1)
    else:
        handle = dist.isend(t, 0)

    handle.wait()

    # dist.broadcast_multigpu([t], src=LOCAL_RANK, group=groups[GLOBAL_RANK])

    print(t)
