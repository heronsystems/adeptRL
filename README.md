![banner](images/banner.png)

adept is a reinforcement learning framework designed to accelerate research 
by providing:
* a modular interface for using custom networks, agents, and environments
* baseline reinforcement learning models and algorithms for PyTorch
* multi-GPU support
* access to various environments
* built-in tensorboard logging, model saving, reloading, evaluation, and 
rendering
* proven hyperparameter defaults

This code is early-access, expect rough edges. Interfaces subject to change. 
We're happy to accept feedback and contributions.

### Read More
* [Installation](#installation)
* [Quickstart](#quickstart)
* [Features](#features)
* [Performance](#performance)

### Documentation
* [Architecture Overview](docs/api_overview.md)
* [ModularNetwork Overview](docs/modular_network.md)
* [Resume training](docs/resume_training.md)
* Evaluate a model
* Render environment

### Examples
* Custom Network ([stub](examples/custom_network_stub.py) | example)
* Custom SubModule ([stub](examples/custom_submodule_stub.py) | [example](adept/networks/net1d/lstm.py))
* Custom Agent ([stub](examples/custom_agent_stub.py) | [example](adept/agents/actor_critic.py))
* Custom Environment ([stub](examples/custom_environment_stub.py) | [example](adept/environments/openai_gym.py))

## Installation
**From source:**
```bash
git clone https://github.com/heronsystems/adeptRL
cd adeptRL
# Remove mpi, sc2, profiler if you don't plan on using these features:
pip install .[mpi,sc2,profiler]
```

**From docker:**
* [docker instructions](./docker/)

## Quickstart
**Train an Agent**
Logs go to `/tmp/adept_logs/` by default. The log directory contains the 
tensorboard file, saved models, and other metadata.

```bash
# Local Mode (A2C)
# We recommend 4GB+ GPU memory, 8GB+ RAM, 4+ Cores
python -m adept.app local --env BeamRiderNoFrameskip-v4

# Distributed Mode (A2C, requires NCCL)
# We recommend 2+ GPUs, 8GB+ GPU memory, 32GB+ RAM, 4+ Cores
python -m adept.app distrib --env BeamRiderNoFrameskip-v4

# Actor Learner Mode
# We recommend 2+ GPUs, 8GB+ GPU memory, 32GB+ RAM, 4+ Cores
python -m adept.app actorlearner

# To see a full list of options:
python -m adept.app -h
python -m adept.app help <command>
```

**Use your own Agent, Environment, Network, or SubModule**  
```python
"""
my_script.py

Train an agent on a single GPU.
"""
from adept.scripts.local import parse_args, main
from adept.network import NetworkModule,
from adept.agents import AgentModule, AgentRegistry
from adept.environments import EnvModule, EnvRegistry


class MyAgent(AgentModule):
    pass  # Implement


class MyEnv(EnvModule):
    pass  # Implement


class MyNet(NetworkModule):
    pass  # Implement


class MySubModule1D(SubModule1D):
    pass  # Implement


if __name__ == '__main__':
    adept.register_agent(MyAgent)
    adept.register_env(MyEnv)
    adept.register_custom_net(MyNet)
    adept.register_submodule(MySubModule1D)


    main(parse_args())
```
* Call your script like this: `python my_script.py --agent MyAgent`
* You can see all the args [here](adept/scripts/local.py) or how to implement
 the stubs in the examples section above.

## Features
### Scripts
**Local (Single-node, Single-GPU)**
* Best place to [start](adept/scripts/local.py) if you're trying to understand code.

**Distributed (Multi-node, Multi-GPU)**
* Uses NCCL backend to all-reduce gradients across GPUs without a parameter 
server or host process.
* Supports NVLINK and InfiniBand to reduce communication overhead
* InfiniBand untested since we do not have a setup to test on.

**Actor Learner (Single Node, Multi-GPU)**
* Our implementation uses GPU workers rather than CPU workers for forward 
passes.

### Agents
* Advantage Actor Critic, A2C ([paper](https://arxiv.org/pdf/1708.05144.pdf) | [code](adept/agents/actor_critic.py))
* Actor Critic Vtrace, IMPALA ([paper](https://arxiv.org/pdf/1802.01561.pdf) | [code](https://arxiv.org/pdf/1802.01561.pdf))

### Networks
* Modular Network Interface: supports arbitrary input and output shapes up to
 4D via a SubModule API.
* Stateful networks (ie. LSTMs)
* Batch normalization ([paper](https://arxiv.org/pdf/1502.03167.pdf))

### Environments
* Atari (OpenAI Gym)

## Performance
* ~ 3,000 Steps/second = 12,000 FPS (Atari)
  * Local Mode
  * 64 environments
  * GeForce 2080 Ti
  * Ryzen 2700x 8-core
* Used to win a 
[Doom competition](https://www.crowdai.org/challenges/visual-doom-ai-competition-2018-track-2) 
(Ben Bell / Marv2in)
![architecture](images/benchmark.png)
* Trained for 50M Steps / 200M Frames
* Up to 30 no-ops at start of each episode
* Evaluated on different seeds than trained on
* Architecture: [Four Convs](./adept/networks/net3d/four_conv.py) (F=32)
followed by an [LSTM](./adept/networks/net1d/lstm.py) (F=512)
* Reproduce with `python -m adept.app local --logdir ~/local64_benchmark --eval 
-y --nb-step 50e6 --env <env-id>`
