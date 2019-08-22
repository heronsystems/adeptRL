from adept.learner.base.learner_module import LearnerModule
from .ac_rollout import ACRolloutLearner
from .impala import ImpalaLearner

LEARNER_REG = [
    ACRolloutLearner  # , ImpalaLearner
]
