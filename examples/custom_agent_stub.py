"""
Use a custom agent.
"""

import typing
from typing import Dict, Tuple

from adept.agent import AgentModule
from adept.scripts.local import parse_args, main

if typing.TYPE_CHECKING:
    import torch
    from adept.network import NetworkModule
    from adept.rewardnorm import RewardNormModule


class MyCustomAgent(AgentModule):
    # You will be prompted for these when training script starts
    args = {"example_arg1": True, "example_arg2": 5}

    def __init__(self, reward_normalizer, action_space, spec_builder):
        super(MyCustomAgent, self).__init__(
            reward_normalizer, action_space,
        )

    @classmethod
    def from_args(
        cls,
        args,
        reward_normalizer: RewardNormModule,
        action_space: Dict[str, Tuple[int, ...]],
        spec_builder,
        **kwargs
    ):
        pass

    @property
    def exp_cache(self):
        """
        Experience cache, probably a RolloutCache or ExperienceReplay.

        :return: BaseExperience
        """
        pass

    @staticmethod
    def output_space(action_space: Dict[str, Tuple[int, ...]]):
        """Merge action space with any agent-based outputs to get an output_space."""
        pass

    def compute_loss(
        self, experience: Tuple[torch.Tensor], next_obs: Dict[str, torch.Tensor]
    ):
        """Compute losses."""
        pass

    def act(
        self,
        network: NetworkModule,
        obs: Dict[str, torch.Tensor],
        prev_internals: Dict[str, torch.Tensor],
    ):
        """Generate an action."""
        pass


if __name__ == "__main__":
    args = parse_args()

    main(args)

    # Call script like this to train agent:
    # python -m custom_agent_stub.py --agent MyCustomAgent
