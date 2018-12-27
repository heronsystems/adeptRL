# Set up nvidia-docker
1. Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
2. Install [CUDA 10](https://developer.nvidia.com/cuda-downloads)
3. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (don't run
 the final line)

It's not required to use the image that corresponds to your OS version since 
docker will just download it. However you must make sure the installed CUDA 
version matches the CUDA version of the docker image (`FROM 
nvidia/cuda:<CUDA_VERSION>-runtime-ubuntu18.04`).

# Build the image
```bash
git clone https://github.com/heronsystems/adeptRL
cd adeptRL
sudo docker build -t adept -f ./docker/Dockerfile .
```

# Run the image
`sudo docker run --runtime nvidia --rm -it --cap-add=SYS_PTRACE --net host -v 
/tmp/adept_logs:/tmp/adept_logs adept`

Notes:
* Docker containers have no permanent storage. The -v flag links a directory 
for use by the container. Link the desired directory with 
`-v <host>:<container>`. By default adept scripts log to `/tmp/adept_logs` so 
the default will place  those under `/tmp/adept_logs` on the host.
* Currently the log timestamps created by the container are GMT. Special 
thanks to @robnagler in https://github.com/radiasoft/devops/issues/132 for the 
`--cap-add` argument. Openmpi 3 requires vader cma so the other solutions will 
not work.

# Run adept
`python -m adept.scripts.local ActorCritic`

* See main readme for more information.
