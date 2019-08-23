from .base.actor_module import ActorModule
from .ac_rollout import ACRolloutActorTrain
from adept.actor.ac_eval import ACActorEval
from .impala import ImpalaActor

ACTOR_REG = [
    ACRolloutActorTrain,
    ACActorEval,
    # ImpalaActor
]
