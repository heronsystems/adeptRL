from .base import ExpModule
from .base import ExpSpecBuilder
from .replay import ExperienceReplay, PrioritizedExperienceReplay
from .rollout import Rollout

EXP_REG = [
    Rollout,  # , ExperienceReplay, PrioritizedExperienceReplay
]
