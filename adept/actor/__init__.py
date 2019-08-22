from .base.actor_module import ActorModule
from .ac_rollout import ACRolloutActorTrain, ACRolloutActorEval
from .impala import ImpalaActor

ACTOR_REG = [
    ACRolloutActorTrain,
    ACRolloutActorEval,
    # ImpalaActor
]

ACTOR_EVAL_LOOKUP = {
    ACRolloutActorTrain.__name__: ACRolloutActorEval.__name__
}
