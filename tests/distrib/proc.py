import os
import torch
import torch.distributed as dist

WORLD_SIZE = int(os.environ['WORLD_SIZE'])
GLOBAL_RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
NB_NODE = int(os.environ['NB_NODE'])


def assign_groups():
    """
    All groups must be created on each node in the same order.
    See: https://pytorch.org/docs/stable/distributed.html#groups

    :return: Group, The group for this node to talk to the host.
    """
    local_size = WORLD_SIZE // NB_NODE
    host_comm_group = None
    for rank in range(WORLD_SIZE):
        local_rank = local_size % rank
        if local_rank == 0:
            continue
        else:
            grp = torch.distributed.new_group([0, rank])
            if rank == GLOBAL_RANK:
                host_comm_group = grp
    return host_comm_group


if __name__ == '__main__':
    dist.init_process_group(
        backend='nccl',
        init_method='file:///tmp/test_init',
        world_size=WORLD_SIZE,
        rank=LOCAL_RANK
    )
    print('LOCAL_RANK', LOCAL_RANK, 'initialized.')
    host_comm_group = assign_groups()
    t = torch.Tensor([LOCAL_RANK]).to('cuda:' + str(LOCAL_RANK))
    dist.broadcast_multigpu([t], src=LOCAL_RANK, group=host_comm_group)
    print(t)
