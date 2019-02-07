# API Overview
![architecture](../images/architecture.png)
### Containers
Containers hold all of the application state. Each subprocess gets a container 
in Distributed and IMPALA modes.
### Agents
An Agent acts on and observes the environment.
Currently only ActorCritic is supported. Other agents, such as DQN or ACER may 
be added later.
### Networks
Networks are not PyTorch modules, they need to implement our abstract 
NetworkModule or ModularNetwork classes. A ModularNetwork consists of a 
source nets, body, and heads.
### Environments
Environments run in subprocesses and send their observation, rewards,
terminals, and infos to the host process. They work pretty much the same way as 
OpenAI's code.
### Experience Caches
An Experience Cache is a Rollout or Experience Replay that is written to after 
stepping and read before learning.

Inheritance Tree:
* [Container](#container)
    * LocalWorker
    * DistribWorker
    * ImpalaHost
    * ImpalaWorkerCPU
    * ImpalaWorkerGPU
* [Agent](#agent)
    * ActorCritic
    * ActorCriticVTrace
    * ACER
    * ACERVTrace
* [ExperienceCache](#experiencecache)
    * ExperienceReplay
    * RolloutCache
* [Environment](#environment)
    * Gym Env
    * SC2Feat1DEnv (1d action space)
    * SC2Feat3DEnv (3d action space)
    * SC2RawEnv (new proto)
* [EnvironmentManager](#environmentmanager)
    * SimpleEnvManager (synchronous, same process, for debugging / rendering)
    * SubProcEnvManager (use torch.multiprocessing.Pipe)
* [Network](#network)
    * ModularNetwork
    * CustomNetwork
* [SubModule](#submodule)
    * SubModule1D
        * Identity1D
        * LSTM
    * SubModule2D
        * Identity2D
        * TransformerEnc
        * TransformerDec
    * SubModule3D
        * Identity3D
        * SC2LEEncoder - encoder from SC2LE paper
        * SC2LEDecoder - decoder from SC2LE paper
        * ConvLSTM
    * SubModule4D
        * Identity4D

### Container
* `agent:` [Agent](#agent)
* `env:` [EnvironmentManager](#environment)
* `exp_cache:` [ExperienceCache](#experiencecache)
* `network:` [Network](#network)
* `local_step_count: int`
* `reward_buffer: List[float], keep track of running rewards`
* `hidden_state: Dict[HSName, torch.Tensor], keep track of current LSTM states`
```python
def run(nb_step, initial_step=0): ...
"""
args:
    nb_step: int, number of steps to train for
return:
    self
"""
```

### Agent
* [network](#network)
```python
def observe(observations, rewards, terminals, infos): ...
"""
args:
    observations: Dict[ObsName, List[torch.Tensor]]
    rewards: List[int]
    terminals: List[Bool]
    infos: List[Any]
return:
    None
"""
def act(observation): ...
"""
legend:
    ObsName = str
    ActionName = str
args:
    observation: Dict[ObsName, torch.Tensor]
return:
    actions: Dict[ActionName, torch.Tensor],
    experience: Dict[ExpName, torch.Tensor]
"""
def compute_loss(experience, next_obs): ...
"""
args:
    experience: torch.Tensor
    next_obs: Dict[ObsName, torch.Tensor]
return:
    losses: Dict[LossName, torch.Tensor]
    metrics: Dict[MetricName, torch.Tensor]
"""
```

### EnvironmentManager
* `obs_preprocessor_gpu: ObsPreprocessor`
* `env_cls: type, environment class`
```python
def step(actions): ...
"""
args:
    actions: Dict[str, List[np.ndarray]]
return:
    Tuple[Observations, Rewards, Terminals, Infos]
"""
def reset(): ...
"""
description:
    Reset the environment to its initial state.
return:
    observation: Dict[ObsName, torch.Tensor]
"""
def close(): ...
"""
description:
    Close environments.
return:
    None
"""
```

### Environment
* [obs_preprocessor_cpu](#)
* [action_preprocessor](#)
```python
def step(action): ...
"""
args:
    action: Dict[str, np.ndarray]
return:
    Tuple[Observation, Reward, Terminal, Info]
"""
```

### ExperienceCache
```python
def read(): ...
def env_write(observations, rewards, terminals): ...
def agent_write(log_prob, entropy): ...
def write(observation, reward, terminal, value, log_prob, entropy, hidden_state): ...
def ready(): ...
def submit(): ...
def receive(): ...
```

### Network
```python
def forward(observation, hidden_state): ...
"""
args:
    observation: Dict[ObsName, torch.Tensor]
    hidden_state: Dict[HSName, torch.Tensor]
"""
```

### SubModule
* `input_shape: Tuple[int]`
```python
def forward(): ...
def output_shape(dim): ...
"""
args:
    dim: int, dimensionality of 
"""
```

### HiddenState
```python
def detach(): ...
```
