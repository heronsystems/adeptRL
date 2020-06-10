from .base.agent_module import AgentModule
from .actor_critic import ActorCritic
from .ppo import PPO

AGENT_REG = [
    ActorCritic,
    PPO
]
