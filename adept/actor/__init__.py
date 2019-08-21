from ._actor import ActorModule
from .ac_rollout import ACRolloutActorTrain, ACRolloutActorEval

ACTOR_REG = [
    ACRolloutActorTrain,
    ACRolloutActorEval
]

ACTOR_EVAL_LOOKUP = {
    ACRolloutActorTrain.__name__: ACRolloutActorEval.__name__
}
