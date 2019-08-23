from .base import RewardNormModule
from .normalizers import Scale, Clip, Identity

REWARD_NORM_REG = [
    Scale, Clip, Identity
]
