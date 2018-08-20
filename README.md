# adept

adept is a library designed to accelerate reinforcement learning research by providing:
* baseline reinforcement learning models and algorithms for PyTorch
* multi-GPU compute options
* access to various environments
* built-in tensorboard logging, model saving, reloading, evaluation, and rendering
* abstractions for building custom networks, agents, execution modes, and experiments
* proven hyperparameter defaults

This code is alpha, expect rough edges.

## Features
Agents / Networks
* Actor Critic with Generalized Advantage Estimation
* Stateful networks (ie. LSTMs)
* Batch Normalization for reinforcement learning

Execution Modes
* Local (Single-GPU, A2C)
* Towered (Multi-GPU, A3C-variant)
* Importance Weighted Actor Learner Architectures, [IMPALA](https://arxiv.org/pdf/1802.01561.pdf) (Faster Multi-GPU)

Environments
* OpenAI Gym
* StarCraft 2 (alpha)

We designed this library to be flexible and extensible. Plugging in novel research ideas should be doable.

## Major Dependencies
* gym
* PyTorch 0.4.x (excluding 0.4.1 due to the [unbind bug](https://github.com/pytorch/pytorch/pull/9995))
* Python 3.5+

## Installation
* Follow instructions for [PyTorch](https://pytorch.org/)  
* (Optional) Follow instructions for [StarCraft 2](https://github.com/Blizzard/s2client-proto#downloads)
* More optional dependencies in requirements.txt

```
# Remove mpi, sc2, profiler if you don't plan on using these features:
pip install adept[mpi,sc2,profiler]
```

## Performance
TODO

## Examples
If you write your own scripts, you can provide your own agents or networks, but we have some presets you can run out of the box.
If you pip installed, these scripts are on your classpath and can be run with the commands below.
If you cloned the repo, put a python in front of each command.

```
# Local Mode (A2C)
# We recommend 4GB+ GPU memory, 8GB+ RAM, 4+ Cores
local.py --env-id BeamRiderNoFrameskip-v4 --agent ActorCritic --vision-network FourConv --network-body LSTM

# Towered Mode (A3C Variant)
# We recommend 2x+ GPUs, 8GB+ GPU memory, 32GB+ RAM, 4+ Cores
towered.py --env-id BeamRiderNoFrameskip-v4 --agent ActorCritic --vision-network FourConv --network-body LSTM

# IMPALA (requires mpi4py and is resource intensive)
# We recommend 2x+ GPUs, 8GB+ GPU memory, 32GB+ RAM, 4+ Cores
mpirun -np 3 -H localhost:3 python -m mpi4py `which impala.py` -n 8

# To see a full list of options:
local.py -h
towered.py -h
impala.py -h
```

## API Reference
![architecture](images/architecture.png)
### Agents
An Agent acts on and observes the environment.
Currently only ActorCritic is supported. Other agents, such as DQN or ACER may be added later.
### Containers
Containers hold all of the application state. Each subprocess gets a container in Towered and IMPALA modes.
### Environments
Environments work using the OpenAI Gym wrappers.
### Experience Caches
An Experience Cache is a Rollout or Experience Replay that is written to after stepping and read before learning.
### Modules
Modules are generally useful PyTorch modules used in Networks.
### Networks
Networks are not PyTorch modules, they need to implement our abstract NetworkInterface or ModularNetwork classes.
A ModularNetwork consists of a trunk, body, and head.
The Trunk can consist of multiple networks for vision or discrete data. It flattens these into an embedding.
The Body network operates on the flattened embedding and would typically be an LSTM, Linear layer, or a combination.
The Head depends on the Environment and Agent and is created accordingly.

## Acknowledgements
We borrow pieces of OpenAI's (gym)[https://github.com/openai/gym] and (baselines)[https://github.com/openai/baselines] code.
We indicate where this is done.