Inheritance Tree:
* [Container](#container)
    * LocalWorker
    * DistribWorker
    * ImpalaHost
    * ImpalaWorkerCPU
    * ImpalaWorkerGPU
* [Agent](#agent)
    * ActorCritic
    * VTrace
* [ExperienceCache](#experiencecache)
    * ExperienceReplay
    * RolloutBuffer
* [Environment](#environment)
    * GymEnv
    * SC2Feat1DEnv (1d action space)
    * SC2Feat3DEnv (3d action space)
    * SC2RawEnv (new proto)
* [EnvironmentManager](#environmentmanager)
    * SimpleEnvManager (synchronous, same process, for debugging / rendering)
    * SubProcEnvManager (use torch.multiprocessing.Pipe, default start method)
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

# Container
* [agent](#agent)
* [environment](#environment)
* [experience_cache](#experiencecache)
* `local_step_count: int`
* `reward_buffer: List[float], keep track of running rewards`
* `hidden_state: Dict[Id, torch.Tensor], keep track of current LSTM states`
```python
def run(nb_step, initial_step=0): ...
"""
args:
    nb_step: int, number of steps to train for
return:
    None
"""
```

# Agent
* [network](#network)
```python
@staticmethod
def output_space(action_space): ...
"""
legend:
    ActionName = str
    Shape = Tuple[*int]
args:
    action_space: Dict[ActionName, Shape]
    
"""

def step(observation): ...  # ?
"""
legend:
    ObsName = str
    ActionName = str
args:
    observation: Dict[ObsName, torch.Tensor]
return:
    policy_dict: Dict[ActionName, torch.Tensor]
"""
def compute_loss(): ...  # ?
```

# EnvironmentManager
* [obs_preprocessor_gpu](#)
* `env_cls: type, environment class`
```python
def step_train(policy_logits): ...
"""
args:
    policy_logits: Dict[ActionName, torch.Tensor]
return:
    Tuple[Obs, Reward, Done, Info, LogProb, Entropy]
"""
def step_eval(policy_logits): ...
"""
args:
    policy_logits: Dict[ActionName, torch.Tensor]
return:
    Tuple[Obs, Reward, Done, Info]
"""

```

# Environment
* [obs_preprocessor_cpu](#)
* [action_preprocessor](#)
```python

```

# ExperienceCache
```python
def read(): ...
def write(): ...
def ready(): ...
"""
return:
    bool, whether the cache is ready 
"""
```

# Network
```python
def forward(observation, hidden_state): ...
"""
args:
    observation: Dict[ObsName, torch.Tensor]
    hidden_state: Dict[
"""
```

# SubModule
* `input_shape: Tuple[int]`
```python
def forward(): ...
def output_shape(dim): ...
"""
args:
    dim: int, dimensionality of 
"""
```

# ObsPreprocessor
