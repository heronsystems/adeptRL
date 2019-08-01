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


def assign_groups():
    """
    All groups must be created on each node in the same order.
    See: https://pytorch.org/docs/stable/distributed.html#groups

    :return: Group, The group for this node to talk to the host.
    """
    local_size = WORLD_SIZE // NB_NODE
    print('local_size', local_size)
    groups = {}
    for rank in range(WORLD_SIZE):
        if rank == 0:
            continue

        local_rank = local_size % rank

        if local_rank == 0:
            continue
        else:
            grp = torch.distributed.new_group([0, rank])
            groups[rank] = grp
    return groups


if __name__ == '__main__':
    dist.init_process_group(
        backend='gloo',
        # init_method='file:///tmp/test_init',
        world_size=WORLD_SIZE,
        rank=LOCAL_RANK
    )
    print('LOCAL_RANK', LOCAL_RANK, 'initialized.')
    # groups = assign_groups()
    t = torch.Tensor([LOCAL_RANK])
    if LOCAL_RANK == 0:
        dist.recv(t)
    else:
        dist.send(t, 0)

    # dist.broadcast_multigpu([t], src=LOCAL_RANK, group=groups[GLOBAL_RANK])

    print(t)
