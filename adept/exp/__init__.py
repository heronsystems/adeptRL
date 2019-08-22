from .base.exp_module import ExpModule
from .impala_rollout import ImpalaRollout
from .replay import ExperienceReplay, PrioritizedExperienceReplay
from .rollout import ACRollout

EXP_REG = [
    ACRollout, ImpalaRollout  # , ExperienceReplay, PrioritizedExperienceReplay
]
