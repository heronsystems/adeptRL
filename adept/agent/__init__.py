from .base.agent_module import AgentModule
from .actor_critic import ActorCritic
from .dqn_rollout import DQNRollout

AGENT_REG = [
    ActorCritic, DQNRollout
]


