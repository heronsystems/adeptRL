FROM nvidia/cuda:10.0-runtime-ubuntu18.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip wget

RUN apt-get install -y libglib2.0-0

RUN apt-get install -y ssh # mpi communication dep
RUN wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.2.tar.gz
RUN tar -xvf openmpi-3.1.2.tar.gz
WORKDIR /openmpi-3.1.2
RUN ./configure
RUN make -j 8 install
WORKDIR /

RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN apt-get install -y cmake zlib1g-dev libjpeg-dev
# RUN pip3 install gym[atari]
