"""
Use a custom agent.
"""
from adept.agents import AgentModule, AgentRegistry
from adept.scripts.local import parse_args, main


class MyCustomAgent(AgentModule):
    # You will be prompted for these when training script starts
    args = {
        'example_arg1': True,
        'example_arg2': 5
    }

    def __init__(
        self,
        network,
        device,
        reward_normalizer,
        gpu_preprocessor,
        engine,
        action_space,
        nb_env,
        *args,
        **kwargs
    ):
        super(MyCustomAgent, self).__init__(
            network,
            device,
            reward_normalizer,
            gpu_preprocessor,
            engine,
            action_space,
            nb_env
        )

    @classmethod
    def from_args(cls, args, network, device, reward_normalizer,
                  gpu_preprocessor, engine, action_space, **kwargs):
        """
        :param args: Dict[ArgName, Any]
        :param network: BaseNetwork
        :param device: torch.device
        :param reward_normalizer: Callable[[float], float]
        :param gpu_preprocessor: ObsPreprocessor
        :param engine: env_registry.Engines
        :param action_space: Dict[ActionKey, torch.Tensor]
        :param kwargs:
        :return: MyCustomAgent
        """
        pass

    @property
    def exp_cache(self):
        """
        Experience cache, probably a RolloutCache or ExperienceReplay.

        :return: BaseExperience
        """
        pass

    @staticmethod
    def output_space(action_space):
        """
        Merge action space with any agent-based outputs to get an output_space.

        ActionKey = str
        Shape = Tuple[*int]

        :param action_space: Dict[ActionKey, Shape]
        :return:
        """
        pass

    def compute_loss(self, experience, next_obs):
        """
        Compute losses.

        ObsKey = str
        LossKey = str

        :param experience: Tuple[*Any]
        :param next_obs: Dict[ObsKey, torch.Tensor]
        :return: Dict[LossKey, torch.Tensor (0D)]
        """
        pass

    def act(self, obs):
        """
        Generate an action.

        ObsKey = str
        ActionKey = str

        :param obs: Dict[ObsKey, torch.Tensor]
        :return: Dict[ActionKey, np.ndarray]
        """
        pass

    def act_eval(self, obs):
        """
        Generate an action in an evaluation.

        ObsKey = str
        ActionKey = str

        :param obs: Dict[ObsKey, torch.Tensor]
        :return: Dict[ActionKey, np.ndarray]
        """
        pass


if __name__ == '__main__':
    args = parse_args()
    agent_reg = AgentRegistry()
    agent_reg.register_agent(MyCustomAgent)

    main(args, agent_registry=agent_reg)

    # Call script like this to train agent:
    # python -m custom_agent_stub.py --agent MyCustomAgent
