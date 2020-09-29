"""
Use a custom environment.
"""
from adept.env import EnvModule
from adept.scripts.local import parse_args, main


class MyCustomEnv(EnvModule):
    # You will be prompted for these when training script starts
    args = {"example_arg1": True, "example_arg2": 5}
    ids = ["scenario1", "scenario2"]

    def __init__(self, action_space, cpu_ops, gpu_ops, *args, **kwargs):
        super(MyCustomEnv, self).__init__(action_space, cpu_ops, gpu_ops)

    @classmethod
    def from_args(cls, args, seed, **kwargs):
        """
        Construct from arguments. For convenience.

        :param args: Arguments object
        :param seed: Integer used to seed this environment.
        :param kwargs: Any custom arguments are passed through kwargs.
        :return: EnvModule instance.
        """
        pass

    def step(self, action):
        """
        Perform action.

        ActionID = str
        Observation = Dict[ObsKey, Any]
        Reward = np.ndarray
        Terminal = bool
        Info = Dict[Any, Any]

        :param action: Dict[ActionID, Any] Action dictionary
        :return: Tuple[Observation, Reward, Terminal, Info]
        """
        pass

    def reset(self, **kwargs):
        """
        Reset environment.

        ObsKey = str

        :param kwargs:
        :return: Dict[ObsKey, Any] Observation dictionary
        """
        pass

    def close(self):
        """
        Close any connections / resources.

        :return:
        """
        pass


if __name__ == "__main__":
    import adept

    adept.register_env(MyCustomEnv)
    main(parse_args())

    # Call script like this to train agent:
    # python -m custom_env_stub.py --env scenario1
