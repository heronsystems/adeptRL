import os
from threading import Thread

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
    nb_gpu = torch.cuda.device_count()
    print('Device Count', nb_gpu)

    dist.init_process_group(
        backend='nccl',
        world_size=WORLD_SIZE,
        rank=LOCAL_RANK
    )
    wh_group = dist.new_group([0, 1])
    hw_group = dist.new_group([0, 1])

    print('LOCAL_RANK', LOCAL_RANK, 'initialized.')

    def wh_loop():
        count = 0
        while count < 10:
            dist.barrier(wh_group)
            dist.broadcast(t_a, 1, wh_group)
            count += 1
            print(f'wh {count}')
        return True

    def hw_loop():
        count = 0
        while count < 5:
            dist.barrier(hw_group)
            dist.broadcast(t_b, 0, hw_group)
            count += 1
            print(f'hw {count}')
        return True

    if on_host():
        t_a = torch.tensor([1, 2, 3]).to('cuda:0')
        t_b = torch.tensor([1, 2, 3]).to('cuda:0')

    if on_worker():
        t_a = torch.tensor([4, 5, 6]).to('cuda:0')
        t_b = torch.tensor([4, 5, 6]).to('cuda:0')

    thread_wh = Thread(target=wh_loop)
    thread_hw = Thread(target=hw_loop)

    thread_wh.start()
    thread_hw.start()

    thread_wh.join()
    thread_hw.join()

    print('t_a', t_a, 'should be [4, 5, 6]')
    print('t_b', t_b, 'should be [1, 2, 3]')
