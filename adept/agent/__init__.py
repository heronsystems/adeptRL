from .base.agent_module import AgentModule
from .actor_critic import ActorCritic
from .dqn_rollout import DQNRollout
from .dqn_replay import DQNReplay
from .ddqn_rollout import DDQNRollout
from .qrddqn_rollout import QRDDQNRollout
from .qrddqn_replay import QRDDQNReplay

AGENT_REG = [
    ActorCritic, DQNRollout, DQNReplay, DDQNRollout, QRDDQNRollout, QRDDQNReplay
]


