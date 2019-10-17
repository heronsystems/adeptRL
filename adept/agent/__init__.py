from .base.agent_module import AgentModule
from .actor_critic import ActorCritic
from .dqn_rollout import DQNRollout
from .ddqn_rollout import DDQNRollout
from .qrddqn_rollout import QRDDQNRollout

AGENT_REG = [
    ActorCritic, DQNRollout, DDQNRollout, QRDDQNRollout
]


