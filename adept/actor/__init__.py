from ._actor import ActorModule
from .ac_rollout import ACRolloutActorTrain, ACRolloutActorEval

ACTOR_TRAIN_REG = [
    ACRolloutActorTrain
]

ACTOR_EVAL_LOOKUP = {
    ACRolloutActorTrain.__name__: ACRolloutActorEval
}
