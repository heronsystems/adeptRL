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
* StarCraft 2 (alpha, impala mode does not work with SC2 yet)

We designed this library to be flexible and extensible. Plugging in novel research ideas should be doable.

## Major Dependencies
* gym
* PyTorch 0.4.x (excluding 0.4.1 due to an [unbind bug](https://github.com/pytorch/pytorch/pull/9995))
* Python 3.5+

## Installation
* Follow instructions for [PyTorch](https://pytorch.org/)  
* (Optional) Follow instructions for [StarCraft 2](https://github.com/Blizzard/s2client-proto#downloads)

```
# Remove mpi, sc2, profiler if you don't plan on using these features:
pip install adeptRL[mpi,sc2,profiler]
```

## Performance
* Used to win a [Doom competition](http://vizdoom.cs.put.edu.pl/competition-cig-2018/competition-results) (Ben Bell / Marv2in)
* ~2500 training frames per second single-GPU performance on a Dell XPS 15" laptop (Geforce 1050Ti)  

| Env                         | ResNet18V2LSTM (ours) | IMPALA deep (paper) |
|-----------------------------|-----------------------|---------------------|
| BeamRiderNoFrameskip-v4     |             17058.533 |            32463.47 |
| BreakoutNoFrameskip-v4      |               546.467 |              787.34 |
| PongNoFrameskip-v4          |                    21 |               20.98 |
| QbertNoFrameskip-v4         |                4497.5 |           351200.12 |
| SeaquestNoFrameskip-v4      |                  8732 |              1753.2 |
| SpaceInvadersNoFrameskip-v4 |              1159.667 |            43595.78 |
* 30-episode average calculated every 1M training frames up to 50M training frames, then taking best

## Examples
If you write your own scripts, you can provide your own agents or networks, but we have some presets you can run out of the box.
Logs go to `/tmp/adept_logs/` by default.
The log directory contains the tensorboard file, saved models, and other metadata.

```
# Local Mode (A2C)
# We recommend 4GB+ GPU memory, 8GB+ RAM, 4+ Cores
python -m adept.scripts.local --env-id BeamRiderNoFrameskip-v4

# Towered Mode (A3C Variant, requires mpi4py)
# We recommend 2+ GPUs, 8GB+ GPU memory, 32GB+ RAM, 4+ Cores
python -m adept.scripts.towered --env-id BeamRiderNoFrameskip-v4

# IMPALA (requires mpi4py and is resource intensive)
# We recommend 2+ GPUs, 8GB+ GPU memory, 32GB+ RAM, 4+ Cores
mpiexec -n 3 python -m adept.scripts.impala --env-id BeamRiderNoFrameskip-v4

# StarCraft 2 (IMPALA not supported yet)
# Warning: much more resource intensive than Atari
python -m adept.scripts.local --env-id CollectMineralShards

# To see a full list of options:
python -m adept.scripts.local -h
python -m adept.scripts.towered -h
python -m adept.scripts.impala -h
```

## API Reference
![architecture](images/architecture.png)
### Agents
An Agent acts on and observes the environment.
Currently only ActorCritic is supported. Other agents, such as DQN or ACER may be added later.
### Containers
Containers hold all of the application state. Each subprocess gets a container in Towered and IMPALA modes.
### Environments
Environments run in subprocesses and send their observation, rewards, terminals, and infos to the host process.
They work pretty much the same way as OpenAI's code.
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
We borrow pieces of OpenAI's [gym](https://github.com/openai/gym) and [baselines](https://github.com/openai/baselines) code.
We indicate where this is done.