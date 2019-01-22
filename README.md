[![Gitter chat](https://badges.gitter.im/adeptRL.png)](https://gitter.im/adeptRL/Lobby)
[![Build Status](http://ci.heronsystems.com:12345/buildStatus/icon?job=adeptRL/master)](http://ci.heronsystems.com:12345/job/adeptRL/job/master/)
# adept

adept is a library designed to accelerate reinforcement learning research by 
providing:
* baseline reinforcement learning models and algorithms for PyTorch
* multi-GPU support
* access to various environments
* built-in tensorboard logging, model saving, reloading, evaluation, and 
rendering
* modular interface for using custom networks, agents, and environments
* proven hyperparameter defaults

This code is early-access, expect rough edges. Interfaces subject to change. 
We're happy to accept feedback and contributions.

Read More:
* [Features](#features)
* [Installation](#installation)
* [Performance](#performance)
* [Train an Agent](#train-an-agent)

Documentation:
* [API Overview](./docs/api_overview.md)
* ModularNetwork (coming soon)

Examples:
* [Custom network](./examples/custom_network.py)
* Custom submodule (coming soon)
* Custom agent (coming soon)
* Custom environment (coming soon)
* Resume training (coming soon)
* Evaluate a trained model (coming soon)

## Features
Agents / Networks
* Actor Critic with Generalized Advantage Estimation
* Stateful networks (ie. LSTMs)
* Batch norm

Execution Modes
* Local (Single-GPU, A2C)
* Towered (Multi-GPU, A3C-variant)
* Importance Weighted Actor Learner Architectures, 
[IMPALA](https://arxiv.org/pdf/1802.01561.pdf) (Faster Multi-GPU)

Environments
* OpenAI Gym
* StarCraft 2 (alpha, impala mode does not work with SC2 yet)

## Installation
Dependencies:
* gym
* PyTorch 1.x
* Python 3.5+
* We recommend CUDA 10, pytorch 1.0, python 3.6

From source:
* Follow instructions for [PyTorch](https://pytorch.org/)
* (Optional) Follow instructions for 
[StarCraft 2](https://github.com/Blizzard/s2client-proto#downloads)
```bash
git clone https://github.com/heronsystems/adeptRL
cd adeptRL
# Remove mpi, sc2, profiler if you don't plan on using these features:
pip install .[mpi,sc2,profiler]
```

From docker:
* [docker instructions](./docker/)

## Performance
* ~ 3,000 Steps/second = 12,000 FPS
  * Local Mode
  * 64 environments
  * GeForce 2080 Ti
  * Ryzen 2700x 8-core
* Used to win a 
[Doom competition](http://vizdoom.cs.put.edu.pl/competition-cig-2018/competition-results) 
(Ben Bell / Marv2in)
![architecture](images/benchmark.png)
* Up to 30 no-ops at start of each episode
* Evaluated on different seeds than trained on
* Architecture: Four Convs (F=32) followed by an LSTM (F=512)
* Reproduce with `python -m adept.app local --logdir ~/local64_benchmark --eval 
-y --env <env-id>`

## Train an Agent
Logs go to `/tmp/adept_logs/` by default. The log directory contains the 
tensorboard file, saved models, and other metadata.

```bash
# Local Mode (A2C)
# We recommend 4GB+ GPU memory, 8GB+ RAM, 4+ Cores
python -m adept.app local --env BeamRiderNoFrameskip-v4

# Towered Mode (A3C Variant, requires mpi4py)
# We recommend 2+ GPUs, 8GB+ GPU memory, 32GB+ RAM, 4+ Cores
python -m adept.app towered --env BeamRiderNoFrameskip-v4

# IMPALA (requires mpi4py and is resource intensive)
# We recommend 2+ GPUs, 8GB+ GPU memory, 32GB+ RAM, 4+ Cores
python -m adept.app impala --env BeamRiderNoFrameskip-v4

# StarCraft 2 (IMPALA not supported yet)
# Warning: much more resource intensive than Atari
python -m adept.app local --env CollectMineralShards

# To see a full list of options:
python -m adept.app -h
python -m adept.app help <command>
```

## Acknowledgements
We borrow pieces of OpenAI's [gym](https://github.com/openai/gym) and 
[baselines](https://github.com/openai/baselines) code. We indicate where this
 is done.
