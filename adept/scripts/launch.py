r"""
`adept.scripts.train_nccl` is a module that allows you to parallelize training
 across multiple machines and GPUs using NVIDIA's Collective Communications
 Library (NCCL). Adept does not use NCCL directly, but instead uses PyTorch's
torch.distributed API.

How to use this module:
1. Single-node multi-process distributed training

TODO command

This will spawn two processes, each using GPUs 0 and 1. In total, this means
there are four workers, each with 32 environment instances. 32 environments
* 4 workers = 128 total environments.

2. Multi-node multi-process distributed training

Node 0:
TODO command

Node 1:
TODO command

This will spawn two processes, each using GPUs 0 and 1, on Nodes 0 and 1.
"""
from argparse import ArgumentParser, REMAINDER
import os
import subprocess


if __name__ == '__main__':
    parser = ArgumentParser(
        description='AdeptRL distributed mode'
    )

    parser.add_argument(
        "--nb-node", type=int, default=1,
        help="The number of nodes to use for distributed training"
    )
    parser.add_argument(
        "--node-rank", type=int, default=0,
        help="The rank of the node for multi-node distributed training"
    )
    parser.add_argument(
        "--gpu-ids", type=int, nargs='+', default=0,
        help="Which GPU(s) to use for training (default: 0)"
    )
    parser.add_argument(
        "--master-address", type=str, default="127.0.0.1",
        help="Master node (rank 0)'s address should be either the IP "
             "address or the hostname of node 0, for single node "
             "multi-process training, the master address can be 127.0.0.1"
    )
    parser.add_argument(
        "--master-port", type=int, default=29500,
        help="Master node (rank 0)'s free port that needs to be used for "
             "communication during distributed training"
    )
    parser.add_argument(
        "--script", type=str,
        help="The full path of the script to be launched"
    )
    parser.add_argument("--script-args", nargs=REMAINDER)

    args = parser.parse_args()
    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    world_size = args.nb_node * len(args.gpu_ids)

    os_env = os.environ.copy()
    os_env["MASTER_ADDR"] = args.master_address
    os_env["MASTER_PORT"] = str(args.master_port)
    os_env["WORLD_SIZE"] = str(world_size)

    processes = []
    global_rank = 0
    for gpu_id in args.gpu_ids:
        os_env["RANK"] = gpu_id
        cmd = ["python", args.script, "--gpu-id {}".format(gpu_id)] + \
              args.script_args

        process = subprocess.Popen(cmd, env=os_env)
        processes.append(process)
        global_rank += 1

    for process in processes:
        process.wait()
