
import numpy as np
import abc

class BasePotentialBasedReward:
    """
    Class for applying potential reward shaping to a set of observations. Potential based rewards
    are used to prevent the learning of suboptimal policies. The reward for executing a transition
    between states is the difference in value between the potential function applied to each state.
    This condition is sufficient to guarentee policy invariance

    For details, see

    "Policy invariance under reward transformations:
    Theory and application to reward shaping"
    https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    """

    def __init__(
        self,
        name: str,
        gamma: float,
        exponent_coefficient: float,
        minimum: float,
        maximum: float,
        absolute: bool,
        reward_base: float,
    ) -> None:

        """
        Parameters
        ----------
        name: str
            Name of the shaped reward
        gamma : float
            Discount factor needed for calculating potential-based reward shaping
        exponential_coefficient : float
            The coefficient of the exponent value. The smaller the value, the closer to linear
        minimum : float
            Minimum value to be given to the agent. This should match the minimum of the gym space
        maximum : float
            Maximum value to be given to the agent. This should match the maximum of the gym space
        absolute : bool
            If the absolute value should be taken during preprocessing
        reward_base : float
            Reward to use for the phi calculations (before the potential--not the actual reward provided)
        """
        self._name = name
        self._gamma = gamma
        self._exponential_coefficient = exponent_coefficient
        self._minimum = minimum
        self._maximum = maximum
        self._absolute = absolute
        self._reward_base = reward_base

        self._midpoint = (self._maximum - self._minimum) / 2 + self._minimum

    def __call__(self, observation, next_observation, action,) -> float:
        return self._potential_shaping_function(observation, next_observation)

    def name(self) -> str:
        return f"{type(self).__name__}_{self._name}"

    def _preprocess_absolute(self, x):
        return np.abs(x) if self._absolute else x

    def _preprocess_observation(self, x):
        return min(max(self._minimum, x), self._maximum)

    def _potential_shaping_function(self, current_observation, next_observation) -> float:
        return (self._gamma * self._phi(next_observation)) - self._phi(current_observation)

    @abc.abstractmethod
    def _phi(self, x) -> float:
        """
            Example phi function:
            return self._reward_base / (1 + np.exp(self._exponent_coefficient * (self._preprocess_observation(x) - self._midpoint)))
        """
        raise NotImplementedError