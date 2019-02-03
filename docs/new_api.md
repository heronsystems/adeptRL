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
* [TrajectoryCache](#trajectorycache)
    * ExperienceReplay
    * RolloutCache
* [Environment](#environment)
    * GymEnv
    * SC2Feat1DEnv (1d action space)
    * SC2Feat3DEnv (3d action space)
    * SC2RawEnv (new proto)
* [EnvironmentManager](#environmentmanager)
    * SimpleEnvManager (synchronous, same process, for debugging / rendering)
    * SubProcEnvManager (use torch.multiprocessing.Pipe, default start method)
    * SelfPlayEnvManager
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
* `_agent:` [Agent](#agent)
* `_environment:` [Environment](#environment)

* `_network:` [Network](#network)
* `_local_step_count: int`
* `_reward_buffer: List[float], keep track of running rewards`
* `_hidden_state: Dict[Id, torch.Tensor], keep track of current LSTM states`
```python
def run(nb_step, initial_step=0): ...
"""
args:
    nb_step: int, number of steps to train for
return:
    None
"""
```

### Agent
* `_trajectory_cache:` [TrajectoryCache](#trajectorycache)
```python

@staticmethod
def output_space(action_space): ...
"""
args:
    action_space: Dict[ActionName, Shape]
    
"""
def observe(cls): ...
def compute_loss(prediction, trajectory): ...  # ?
```

### EnvironmentManager
* _obs_preprocessor_gpu: [ObsPreprocessor](#)
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

### Environment
* _obs_preprocessor_cpu: [ObsPreprocessor](#)
* _action_preprocessor: [ActionPreprocessor](#)
* `_action_space: Dict[]`
```python
def step(observation): ...
def reset(): ...
def close(): ...


```

### TrajectoryCache
```python
def read(): ...
def write(): ...
def ready(): ...
"""
return:
    bool, whether the cache is ready 
"""
```

### Network
```python
def forward(observation, hidden_state): ...
"""
args:
    observation: Dict[ObsName, torch.Tensor]
    hidden_state: Dict[
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

### ObsPreprocessor
