# Docker & Nvidia-docker install
1. Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
2. Install CUDA
3. Then nvidia-docker https://github.com/NVIDIA/nvidia-docker (don't run the final line)
4. Clone the repo `git clone https://github.com/heronsystems/adeptRL`

It's not required to use the image that corresponds to your OS version since docker will just download it. BUT you MUST use the same CUDA version on the host and container.

If you want to use a Container other than Ubuntu 16.04 or CUDA version other than 9.1 change the `FROM nvidia/cuda` line of the Dockerfile and the Torch whl in the pip3 install.

# Building the image
From the adept root dir run the following:

`sudo docker build -t adept -f ./docker/Dockerfile .`

# Running the image
This command opens an interactive terminal to the scripts directory:

`docker run --runtime=nvidia --rm -it --cap-add=SYS_PTRACE --net=host -v /tmp/adept_logs:/tmp/adept_logs adept`

Docker containers have no permanent storage. The -v flag links a directory for use by the container. Link the desired directory with `-v <host>:<container>`. By default adept scripts log to `/tmp/adept_logs` so the default will place those under `/tmp/adept_logs` on the host.

Note: Currently the log timestamps created by the container are GMT. Special thanks to @robnagler in https://github.com/radiasoft/devops/issues/132 for the `--cap-add` argument. Openmpi 3 requires vader cma so the other solutions will not work

