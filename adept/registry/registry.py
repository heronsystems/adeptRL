import os
import pickle
from glob import glob
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


def agent_eval_lookup():
    from adept.agent import ActorCritic
    return {
        ActorCritic.__name__: ACActorEval.__name__
    }


def actor_eval_lookup():
    from adept.actor import ImpalaWorkerActor
    return {
        ImpalaWorkerActor.__name__: ACActorEval.__name__
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

        self._internal_modules = set([k for k, v in self._iter_all_classes()])

    def _iter_all_classes(self):
        return chain(
            self._agent_class_by_id.items(),
            self._actor_class_by_id.items(),
            self._exp_class_by_id.items(),
            self._learner_class_by_id.items(),
            self._env_class_by_engine_id.items(),
            self._reward_norm_class_by_id.items(),
            self._net_class_by_id.items(),
            self._submod_class_by_id.items()
        )

    def save_extern_classes(self, log_id_dir):
        """
        Saves external classes to log id dir. This needs to be done for
        distributed topologies if using external classes.
        :return:
        """
        for k, v in self._iter_all_classes():
            if k not in self._internal_modules:
                if k in self._agent_class_by_id:
                    self._write_cls(v, log_id_dir, 'agent')
                elif k in self._actor_class_by_id:
                    self._write_cls(v, log_id_dir, 'actor')
                elif k in self._exp_class_by_id:
                    self._write_cls(v, log_id_dir, 'exp')
                elif k in self._learner_class_by_id:
                    self._write_cls(v, log_id_dir, 'learner')
                elif k in self._env_class_by_engine_id:
                    self._write_cls(v, log_id_dir, 'env')
                elif k in self._reward_norm_class_by_id:
                    self._write_cls(v, log_id_dir, 'reward_norm')
                elif k in self._net_class_by_id:
                    self._write_cls(v, log_id_dir, 'net')
                elif k in self._submod_class_by_id:
                    self._write_cls(v, log_id_dir, 'submod')
                else:
                    raise Exception('Unreachable.')

    def load_extern_classes(self, log_id_dir):
        def join(d):
            return os.path.join(log_id_dir, d)
        cls_dirs = [join('agent'), join('actor'), join('exp'), join('learner'),
                    join('env'), join('reward_norm'), join('net'),
                    join('submod')]
        for cls_dir in cls_dirs:
            if os.path.exists(cls_dir):
                dirname = os.path.split(cls_dir)[-1]
                for cls_path in glob(os.path.join(cls_dir, '*.cls')):
                    cls = self._load_cls(cls_path)
                    if 'agent' in dirname:
                        self.register_agent(cls)
                    elif 'actor' in dirname:
                        self.register_actor(cls)
                    elif 'exp' in dirname:
                        self.register_exp(cls)
                    elif 'learner' in dirname:
                        self.register_learner(cls)
                    elif 'env' in dirname:
                        self.register_env(cls)
                    elif 'reward_norm' in dirname:
                        self.register_reward_normalizer(cls)
                    elif 'net' in dirname:
                        self.register_network(cls)
                    elif 'submod' in dirname:
                        self.register_submodule(cls)
                    else:
                        raise Exception('Unreachable.')

    @staticmethod
    def _write_cls(cls, log_id_dir, dirname):
        os.makedirs(os.path.join(log_id_dir, dirname), exist_ok=True)
        filepath = os.path.join(log_id_dir, dirname, cls.__name__ + '.cls')
        with open(filepath, 'wb') as f:
            pickle.dump(cls, f)

    @staticmethod
    def _load_cls(cls_path):
        with open(cls_path, 'rb') as f:
            return pickle.load(f)

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
        if _id in self._agent_class_by_id:
            return self._agent_class_by_id[_id].output_space(action_space)
        elif _id in self._actor_class_by_id:
            return self._actor_class_by_id[_id].output_space(action_space)
        else:
            raise IndexError('Agent or Actor not found: {}'.format(_id))

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
        actor_lookup = actor_eval_lookup()
        if train_name in agent_lookup:
            return self._actor_class_by_id[agent_lookup[train_name]]
        elif train_name in actor_lookup:
            return self._actor_class_by_id[actor_lookup[train_name]]
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
    def register_env(self, env_module_class):
        """
        Register an environment class.

        EnvID = str

        :param env_module_class: EnvModule
        :return: EnvRegistry
        """
        engine_id = env_module_class.__name__
        # TODO assert no duplicate env_ids
        assert issubclass(env_module_class, EnvModule)
        env_module_class.check_args_implemented()
        env_module_class.check_ids_implemented()
        self._engine_ids_by_env_id_set[frozenset(env_module_class.ids)] = engine_id
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
        for env in ENV_REG:
            self.register_env(env)

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
