from .base.actor_module import ActorModule
from .ac_rollout import ACRolloutActorTrain
from .dqn import DQNRolloutActor, DDQNRolloutActor, QRDDQNRolloutActor
from .dqn import DQNReplayActor, QRDDQNReplayActor
from adept.actor.ac_eval import ACActorEval
from .impala import ImpalaHostActor, ImpalaWorkerActor

ACTOR_REG = [
    ACRolloutActorTrain,
    ACActorEval,
    ImpalaHostActor,
    ImpalaWorkerActor,
    DQNRolloutActor,
    DQNReplayActor,
    DDQNRolloutActor,
    QRDDQNRolloutActor,
    QRDDQNReplayActor
]
