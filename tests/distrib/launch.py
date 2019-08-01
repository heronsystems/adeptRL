import os
import sys
import subprocess

NB_PROC = 2
NB_NODE = 1
NODE_RANK = 0
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

dist_world_size = NB_PROC * NB_NODE

current_env = os.environ.copy()
current_env["MASTER_ADDR"] = MASTER_ADDR
current_env["MASTER_PORT"] = MASTER_PORT
current_env["WORLD_SIZE"] = str(dist_world_size)
current_env["NB_NODE"] = str(NB_NODE)


def main():
    processes = []
    for local_rank in range(NB_PROC):
        # each process's rank
        dist_rank = NB_PROC * NODE_RANK + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        cmd = [
            sys.executable,
            "-u",
            "proc.py"
        ]

        proc = subprocess.Popen(cmd, env=current_env)
        processes.append(proc)

    for process in processes:
        process.wait()


if __name__ == '__main__':
    main()
