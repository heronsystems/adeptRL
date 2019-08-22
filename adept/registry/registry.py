from collections import namedtuple


class Registry:
    def __init__(self):
        pass

    def register_agent(self, agent_cls):
        pass

    def register_actor(self, actor_cls):
        pass

    def register_learner(self, learner_cls):
        pass

    def register_env(self):
        pass


def agent_eval_lookup():
    from adept.agent import ActorCritic
    from adept.actor import ACRolloutActorEval
    return {
        ActorCritic.__name__: ACRolloutActorEval.__name__
    }


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


