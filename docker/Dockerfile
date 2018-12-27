# Install Ubuntu 18.04
# Other supported cuda images: https://hub.docker.com/r/nvidia/cuda/
FROM nvidia/cuda:10.0-runtime-ubuntu18.04

# Install basic dependencies
RUN apt-get update
RUN apt-get install -y python3 python3-pip wget

# OpenCV dependencies
RUN apt-get install -y libglib2.0-0

# MPI dependencies
RUN apt-get install -y ssh
RUN wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.2.tar.gz
RUN tar -xvf openmpi-3.1.2.tar.gz
WORKDIR /openmpi-3.1.2
RUN ./configure
RUN make -j 8 install
WORKDIR /

# Gym dependencies
RUN apt-get install -y cmake zlib1g-dev libjpeg-dev

# PyTorch 1.0
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl

# Set Python 3 as the default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install AdeptRL
COPY ./ /adeptRL
WORKDIR /adeptRL
RUN ls
RUN pip3 install .[all]
