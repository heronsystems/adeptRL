from .base import ExpModule
from .base import ExpSpecBuilder
from .impala_rollout import ImpalaRollout
from .replay import ExperienceReplay, PrioritizedExperienceReplay
from .rollout import ACRollout, Rollout

EXP_REG = [
    Rollout, ACRollout, ImpalaRollout # , ExperienceReplay, PrioritizedExperienceReplay
]
