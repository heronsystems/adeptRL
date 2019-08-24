from collections import namedtuple
from itertools import chain

from adept.actor import ActorModule, ACActorEval
from adept.agent import AgentModule
from adept.env import EnvModule
from adept.exp import ExpModule
from adept.learner import LearnerModule
from adept.network import NetworkModule
from adept.network.base.submodule import SubModule
from adept.rewardnorm import RewardNormModule
from adept.utils.requires_args import RequiresArgsMixin

Topology = namedtuple('Topology', ['actor', 'learner', 'exp'])


def topology_lookup():
    from adept.actor import ImpalaActor
    from adept.learner import ImpalaLearner
    from adept.exp import ImpalaRollout
    return {
        'Impala': Topology(
            ImpalaActor.__name__,
            ImpalaLearner.__name__,
            ImpalaRollout.__name__
        )
    }


def agent_eval_lookup():
    from adept.agent import ActorCritic
    return {
        ActorCritic.__name__: ACActorEval.__name__
    }


def topology_eval_lookup():
    from adept.actor import ACActorEval
    return {
        'Impala': ACActorEval.__name__
    }


class Registry:
    def __init__(self):
        self._agent_class_by_id = {}
        self._actor_class_by_id = {}
        self._exp_class_by_id = {}
        self._learner_class_by_id = {}

        self._engine_ids_by_env_id_set = {}
        self._env_class_by_engine_id = {}
        self._reward_norm_class_by_id = {}

        self._net_class_by_id = {}
        self._submod_class_by_id = {}

        self._register_agents()
        self._register_actors()
        self._register_learners()
        self._register_exps()

        self._register_envs()
        self._register_reward_norms()

        self._register_networks()
        self._register_submodules()

        self._internal_modules = set(chain(
            [v.__name__ for v in self._agent_class_by_id.values()],
            [v.__name__ for v in self._actor_class_by_id.values()],
            [v.__name__ for v in self._exp_class_by_id.values()],
            [v.__name__ for v in self._learner_class_by_id.values()],
            [v.__name__ for v in self._env_class_by_engine_id.values()],
            [v.__name__ for v in self._reward_norm_class_by_id.values()],
            [v.__name__ for v in self._net_class_by_id.values()],
            [v.__name__ for v in self._submod_class_by_id.values()]
        ))

    # AGENT METHODS
    def register_agent(self, agent_class):
        """
        Use your own agent class.

        :param agent_class: adept.agent.AgentModule. Your custom class.
        :return:
        """
        assert issubclass(agent_class, AgentModule)
        agent_class.check_args_implemented()
        self._agent_class_by_id[agent_class.__name__] = agent_class
        return self

    def lookup_agent(self, agent_id):
        return self._agent_class_by_id[agent_id]

    def lookup_output_space(self, _id, action_space):
        """
        For a given or topology id, provide the shape of the outputs.

        :param _id: str, agent_id or topology_id
        :param action_space:
        :return:
        """
        topologies = topology_lookup()
        if _id in self._agent_class_by_id:
            return self._agent_class_by_id[_id].output_space(action_space)
        elif _id in topologies:
            actor_id = topologies[_id][0]
            return self._actor_class_by_id[actor_id].output_space(action_space)
        else:
            raise IndexError(f'Actor or Topology not found: {_id}')

    # ACTOR METHODS
    def register_actor(self, actor_class):
        """
        Use your own actor class.

        :param actor_class: adept.actor.ActorModule. Your custom class.
        :return:
        """
        assert issubclass(actor_class, ActorModule)
        actor_class.check_args_implemented()
        self._actor_class_by_id[actor_class.__name__] = actor_class
        return self

    def lookup_actor(self, actor_id):
        return self._actor_class_by_id[actor_id]

    def lookup_eval_actor(self, train_name):
        """
        Get the eval actor by training agent or actor name.

        :param train_name: Name of agent or actor class used for training
        :return: ActorModule
        """
        agent_lookup = agent_eval_lookup()
        topology_lookup = topology_eval_lookup()
        if train_name in agent_lookup:
            return self._actor_class_by_id[agent_lookup[train_name]]
        elif train_name in topology_lookup:
            return self._actor_class_by_id[topology_lookup[train_name]]
        else:
            raise IndexError(f'Unknown training agent or actor: {train_name}')

    # EXP METHODS
    def register_exp(self, exp_class):
        """
        Use your own exp cache.

        :param exp_class: adept.exp.ExpModule. Your custom class.
        :return:
        """
        assert issubclass(exp_class, ExpModule)
        exp_class.check_args_implemented()
        self._exp_class_by_id[exp_class.__name__] = exp_class
        return self

    def lookup_exp(self, exp_id):
        return self._exp_class_by_id[exp_id]

    # LEARNER METHODS
    def register_learner(self, learner_cls):
        """
        Use your own learner class.

        :param learner_cls: adept.learner.LearnerModule. Your custom class.
        :return:
        """
        assert issubclass(learner_cls, LearnerModule)
        learner_cls.check_args_implemented()
        self._learner_class_by_id[learner_cls.__name__] = learner_cls

    def lookup_learner(self, learner_id):
        return self._learner_class_by_id[learner_id]

    # ENV METHODS
    def register_env(self, env_module_class, env_id_set):
        """
        Register an environment class.

        EnvID = str

        :param env_module_class: EnvModule
        :param env_id_set: List[EnvID], list of environment ids supported by
        the provided module.
        :return: EnvRegistry
        """
        engine_id = env_module_class.__name__
        # TODO assert no duplicate env_ids
        assert issubclass(env_module_class, EnvModule)
        env_module_class.check_args_implemented()
        self._engine_ids_by_env_id_set[frozenset(env_id_set)] = engine_id
        self._env_class_by_engine_id[engine_id] = env_module_class
        return self

    def lookup_env(self, env_id):
        engine = self.lookup_engine(env_id)
        return self._env_class_by_engine_id[engine]

    def lookup_engine(self, env_id):
        eng = None
        for env_id_set, engine_id in self._engine_ids_by_env_id_set.items():
            if env_id in env_id_set:
                eng = engine_id
        if eng is None:
            raise Exception('Environment not registered: ' + env_id)
        return eng

    # REWARD NORM METHODS
    def register_reward_normalizer(self, normalizer_cls):
        assert issubclass(normalizer_cls, RewardNormModule)
        normalizer_cls.check_args_implemented()
        self._reward_norm_class_by_id[normalizer_cls.__name__] = normalizer_cls

    def lookup_reward_normalizer(self, reward_norm_id):
        return self._reward_norm_class_by_id[reward_norm_id]

    # NETWORK METHODS
    def register_network(self, net_cls):
        """
        Add your custom network.

        :param name: str
        :param net_cls: NetworkModule
        :return:
        """
        assert issubclass(net_cls, NetworkModule)
        net_cls.check_args_implemented()
        self._net_class_by_id[net_cls.__name__] = net_cls
        return self

    def lookup_network(self, net_name):
        """
        Get a NetworkModule by name.

        :param net_name: str
        :return: NetworkModule.__class__
        """
        return self._net_class_by_id[net_name]

    def register_submodule(self, submod_cls):
        """
        Add your own SubModule.

        :param name: str
        :param submod_cls: Submodule
        :return:
        """
        assert issubclass(submod_cls, SubModule)
        submod_cls.check_args_implemented()
        self._submod_class_by_id[submod_cls.__name__] = submod_cls
        return self

    def lookup_submodule(self, submodule_name):
        """
        Get a SubModule by name.

        :param submodule_name: str
        :return: SubModule.__class__
        """
        return self._submod_class_by_id[submodule_name]

    def lookup_modular_args(self, args):
        """
        :param args: Dict[name, Any]
        :return: Dict[str, Any]
        """
        return {
            **self.lookup_submodule(args.net1d).args,
            **self.lookup_submodule(args.net2d).args,
            **self.lookup_submodule(args.net3d).args,
            **self.lookup_submodule(args.net4d).args,
            **self.lookup_submodule(args.netbody).args,
            **self.lookup_submodule(args.head1d).args,
            **self.lookup_submodule(args.head2d).args,
            **self.lookup_submodule(args.head3d).args,
            **self.lookup_submodule(args.head4d).args,
        }

    def prompt_modular_args(self, args):
        """
        :param args: Dict[name, Any]
        :return: Dict[str, Any]
        """
        return RequiresArgsMixin._prompt(
            'ModularNetwork',
            self.lookup_modular_args(args)
        )

    def _register_agents(self):
        from adept.agent import AGENT_REG
        for agent in AGENT_REG:
            self.register_agent(agent)

    def _register_actors(self):
        from adept.actor import ACTOR_REG
        for actor in ACTOR_REG:
            self.register_actor(actor)

    def _register_learners(self):
        from adept.learner import LEARNER_REG
        for learner in LEARNER_REG:
            self.register_learner(learner)

    def _register_exps(self):
        from adept.exp import EXP_REG
        for exp in EXP_REG:
            self.register_exp(exp)

    def _register_envs(self):
        from adept.env import ENV_REG
        for env, id_list in ENV_REG:
            self.register_env(env, id_list)

    def _register_reward_norms(self):
        from adept.rewardnorm import REWARD_NORM_REG
        for rewardnorm in REWARD_NORM_REG:
            self.register_reward_normalizer(rewardnorm)

    def _register_networks(self):
        from adept.network import NET_REG
        for net in NET_REG:
            self.register_network(net)

    def _register_submodules(self):
        from adept.network import SUBMOD_REG
        for submod in SUBMOD_REG:
            self.register_submodule(submod)
