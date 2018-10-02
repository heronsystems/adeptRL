# Supported cuda images: List at https://hub.docker.com/r/nvidia/cuda/

# Ubuntu 16.04 containers
# Cuda 8.0 with cudnn 7 in an ubuntu16.04 container
# FROM nvidia/cuda:8.0-cudnn7-runtime-ubuntu16.04
# Cuda 9.0 with cudnn 7 in an ubuntu16.04 container
# FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
# Cuda 9.1 with cudnn 7 in an ubuntu16.04 container
FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04
# Cuda 9.2 with cudnn 7 in an ubuntu16.04 container
# FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu16.04

# Ubuntu 18.04 containers
# Cuda 9.2 with cudnn 7 in an ubuntu18.04 container
# FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04

# Install basic deps
RUN apt-get update
RUN apt-get install -y python3 python3-pip wget

# python deps used by all packages
RUN pip3 install numpy

# Install opencv
RUN apt-get install -y libglib2.0-0 
RUN pip3 install opencv-python-headless # temp fix for setup.py not finding this package

# Install mpi
RUN apt-get install -y ssh # mpi communication dep
RUN wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.2.tar.gz
RUN tar -xvf openmpi-3.1.2.tar.gz
WORKDIR /openmpi-3.1.2
RUN ./configure
RUN make -j 8 install
WORKDIR /

# Torch Supported Version (0.4.0). List is at https://pytorch.org/get-started/previous-versions/
# Ubuntu16.04 python is 3.5
# cuda 8.0
# RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
# cuda 9.0
# RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
# cuda 9.1
RUN pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
# cuda 9.2
# RUN pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.0-cp35-cp35m-linux_x86_64.whl

# Ubuntu 18.04 python is 3.6
# cuda 9.2
# RUN pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.0-cp36-cp36m-linux_x86_64.whl

# Install gym[atari]
# deps from https://github.com/openai/gym
RUN apt-get install -y cmake zlib1g-dev libjpeg-dev
RUN pip3 install gym[atari]

# install packages so that docker caches them if the adept library is changed
# TODO: is there a better way of doing this?
RUN pip3 install absl-py tensorboardX cloudpickle
# additional deps aren't installed from the setup.py script
RUN pip3 install mpi4py pyinstrument

# copy adeptRL
COPY ./ /adeptRL

# Install adeptRL
WORKDIR /adeptRL
RUN python3 setup.py install

# Change into scripts
WORKDIR /adeptRL/adept/scripts

# Finally alias python3 -> python
RUN ln -s /usr/bin/python3 /usr/bin/python
