from adept.registry.registry import Registry

REGISTRY = Registry()


def register_agent(agent_cls):
    REGISTRY.register_agent(agent_cls)


def register_actor(actor_cls):
    REGISTRY.register_actor(actor_cls)


def register_learner(learner_cls):
    REGISTRY.register_learner(learner_cls)


def register_env():
    REGISTRY.register_env()
