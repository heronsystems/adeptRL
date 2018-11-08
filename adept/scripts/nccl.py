import os
import torch
import torch.distributed as dist

def main(world_size: int, args):
    dist.init_process_group(backend='nccl',
                            init_method='file:///tmp/adept_init',
                            world_size=world_size)
    torch.cuda.set_device(args.gpu_id)
    print(dist.get_rank())
    print(dist.get_world_size())
    a = torch.tensor(5.)

    dist.all_reduce_multigpu([a])

    print(a)

if __name__ == '__main__':
    import argparse
    from adept.utils.script_helpers import add_base_args

    parser = argparse.ArgumentParser(description="AdeptRL Distributed")

    def add_args(parser):
        parser = parser.add_argument_group("Distributed Args")
        parser.add_argument(
            "--gpu-id", type=int, default=0,
            help="Which GPU to use for training (default: 0)"
        )

    parser = add_base_args(parser, add_args)
    args = parser.parse_args()

    main(os.environ["WORLD_SIZE"], args)
