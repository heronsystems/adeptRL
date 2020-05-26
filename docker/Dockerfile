# Copyright (C) 2020 Heron Systems, Inc.
#
# This Dockerfile builds an image that:
# (1) installs all dependencies that should be necessary to develop and run
#     code for the current project.
#
# (2) supports persistence across Docker runs. After running connect.py for
#     the first time, you should notice a new folder /mnt/users/yourusername,
#     both on your host machine as well as your Docker instance. All files
#     that you wish to persist should be stored under folder. For convenience,
#     this Dockerfile sets up zsh under /mnt/users/yourusername and symlinks
#     necessary files to /home/yoursername (in the Docker instance) such that
#     history persists (so, ctrl-p works across Docker runs).
#
# Rather than running your own docker commands, it's recommended that you set
# up a docker image via the connect.py script. You'll almost always want to
# run something like:
#
# python connect.py --dockerfile ./Dockerfile --username gburdell \
#                   --email gburdell@heronsystems.com \
#                   --fullname "George P. Burdell"
#
# It's important to ensure that username, email, and fullname all match across
# all Docker instances that you wish to all be associated with the same user
# for persistence reasons.
FROM nvidia/cudagl:10.1-devel-ubuntu18.04

ARG USERNAME
ARG EMAIL
ARG FULLNAME

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        less \
        openssh-server \
        sudo \
        systemd \
        tmux \
        unzip \
        vim \
        wget \
        zsh && \
# ==================================================================
# useful developer tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3.6-tk \
        python3-distutils \
        cython \
        libopenmpi-dev \
        openmpi-bin \
        libsm6 \
        libxext6 \
        libxrender-dev \
        locales \
        mesa-utils

# ==================================================================
# Python goodies
# ------------------------------------------------------------------
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        black==19.10b0 \
        cloudpickle==1.4.1 \
        cmake==3.15.3 \
        Cython==0.29.13 \
        deepdiff==4.3.2 \
        defusedxml==0.6.0 \
        docopt==0.6.2 \
        flake8==3.7.8 \
        glances==3.1.4.1 \
        h5py==2.10.0 \
        imagesize==1.1.0 \
        ipdb==0.13.2 \
        jupyter==1.0.0 \
        magicattr==0.1.4 \
        matplotlib==3.1.1 \
        numpy==1.17.2 \
        opencv-python==4.1.1.26 \
        ordered-set==3.1.1 \
        pandas==0.25.1 \
        Pillow==6.1.0 \
        protobuf==3.9.2 \
        pycodestyle==2.5.0 \
        pyflakes==2.1.1 \
        ray==0.8.5 \
        requests==2.23.0 \
        scipy==1.3.1 \
        six==1.12.0 \
        scikit-learn==0.20.1 \
        scikit-optimize==0.7.4 \
        tabulate==0.8.7 \
        tensorboard==1.14.0 \
        tensorflow==1.14.0 \
        tensorboardX==2.0 \
        threadpoolctl==2.0.0 \
        torch==1.5.0 \
        torchvision==0.6.0 \
        tqdm==4.43.0

# ==================================================================
# cuDNN
# ==================================================================
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    mkdir -p /opt/cudnn7 && \
    cd /opt/cudnn7/ && \
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb && \
    dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb && \
    apt-get update && \
    $APT_INSTALL libcudnn7-dev && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# ==================================================================
# OpenAI Gym
# ------------------------------------------------------------------
# Dependencies for OpenAI Gym
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        ffmpeg \
        libosmesa6-dev \
        libsdl2-dev \
        patchelf \
        python-pyglet \
        python3-opengl \
        swig \
        xvfb && \
    $PIP_INSTALL \
        atari-py==0.2.6 \
        box2d==2.3.10 \
        cffi==1.14.0 \
        glfw==1.11.0 \
        pybullet==2.7.2 \
        gym==0.14.0

# ==================================================================
# matplotlib display dependencies
# ------------------------------------------------------------------
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        tcl-dev \
        tk-dev \
        python-tk \
        python3-tk && \
    ln -fs /usr/share/zoneinfo/US/Eastern /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# ==================================================================
# user setup
# ------------------------------------------------------------------
# Create a user and enable passwordless sudo
RUN useradd -m --shell /bin/zsh -r $USERNAME -u 1000 && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" | sudo EDITOR="tee -a" visudo

USER $USERNAME

# prezto
RUN cd /home/$USERNAME && \
    git clone --recursive https://github.com/sorin-ionescu/prezto.git "${ZDOTDIR:-$HOME}/.zprezto" && \
    echo $'#!/bin/zsh\nsetopt EXTENDED_GLOB\nfor rcfile in ${ZDOTDIR:-$HOME}/.zprezto/runcoms/^README.md(.N); do\n  ln -s $rcfile ${ZDOTDIR:-$HOME}/.${rcfile:t}\ndone\n' > .zsetup && \
    zsh .zsetup && \
# fzf
    git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && \
    printf "y\ny\ny\n" | ~/.fzf/install

# RUN locale-gen en_US.UTF-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LANGUAGE=en_US.UTF-8

ENV DBUS_FATAL_WARNINGS=0

ENV USERNAME=$USERNAME
ENV EMAIL=$EMAIL
ENV FULLNAME=$FULLNAME

ADD startup.sh /home/$USERNAME/startup.sh
ENTRYPOINT ["/bin/zsh", "-c", "cd /home/${USERNAME}; zsh -c 'sh /home/$USERNAME/startup.sh; rm -f /home/$USERNAME/startup.sh; tmux;'"]
