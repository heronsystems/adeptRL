from adept.environments._plugin import AdeptEnvPlugin
from adept.environments._spaces import Spaces, Space
import numpy as np

from adept.preprocess.observation import ObsPreprocessor


class AdeptHanoiEnv(AdeptEnvPlugin):
    def __init__(self):
        # Observation Space
        # 3 stacks
        # n rings
        # n-bit encoding
        tower_space = Space((3, 8, 3), 0., 1., np.int32)
        observation_space = Spaces({'tower_space': tower_space})
        cpu_preprocessor = ObsPreprocessor([], observation_space)
        gpu_preprocessor = ObsPreprocessor([], observation_space)

        # Action Space
        # 0: A->B
        # 1: A->C
        # 2: B->A
        # 3: B->C
        # 4: C->A
        # 5: C->B
        action_space = Space((6,), 0., 1., np.float32)

        super(AdeptHanoiEnv, self).__init__(
            observation_space,
            action_space,
            cpu_preprocessor,
            gpu_preprocessor
        )

    def step(self, action):
        """
        :param action:
        :return: obs, reward, terminal, info
        """
        if action[0]:
            pass
        elif action[1]:
            pass
        elif action[2]:
            pass
        elif action[3]:
            pass
        elif action[4]:
            pass
        elif action[5]:
            pass
        else:
            raise NotImplementedError('Invalid action.')


class Hanoi:
    def __init__(self):
        pass
