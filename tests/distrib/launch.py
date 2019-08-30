import os
import sys
import subprocess

NB_NODE = 1
NODE_RANK = 0
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"


def main(args):
    processes = []
    for local_rank in range(args.nb_proc):
        # each process's rank
        dist_rank = args.nb_proc * NODE_RANK + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        cmd = [
            sys.executable,
            "-u",
            args.script
        ]

        proc = subprocess.Popen(cmd, env=current_env)
        processes.append(proc)

    for process in processes:
        process.wait()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--script', default='exp_sync_broadcast.py')
    parser.add_argument('--nb-proc', type=int, default=2)
    args = parser.parse_args()

    dist_world_size = args.nb_proc * NB_NODE

    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = MASTER_ADDR
    current_env["MASTER_PORT"] = MASTER_PORT
    current_env["WORLD_SIZE"] = str(dist_world_size)
    current_env["NB_NODE"] = str(NB_NODE)

    main(args)
