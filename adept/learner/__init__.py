from adept.learner.base.learner_module import LearnerModule
from .ac_rollout import ACRolloutLearner
from .dqn import DQNRolloutLearner, DQNReplayLearner, DDQNRolloutLearner, QRDDQNRolloutLearner
from .impala import ImpalaLearner

LEARNER_REG = [
    ACRolloutLearner, ImpalaLearner, DQNRolloutLearner, DQNReplayLearner, DDQNRolloutLearner, QRDDQNRolloutLearner
]
