
def register_agent(agent_cls):
    from adept.registry import REGISTRY
    REGISTRY.register_agent(agent_cls)


def register_actor(actor_cls):
    from adept.registry import REGISTRY
    REGISTRY.register_actor(actor_cls)


def register_exp(exp_cls):
    from adept.registry import REGISTRY
    REGISTRY.register_exp(exp_cls)


def register_learner(learner_cls):
    from adept.registry import REGISTRY
    REGISTRY.register_learner(learner_cls)


def register_env(env_cls, env_ids):
    from adept.registry import REGISTRY
    REGISTRY.register_env(env_cls, env_ids)


def register_reward_norm(rwd_norm_cls):
    from adept.registry import REGISTRY
    REGISTRY.register_reward_normalizer(rwd_norm_cls)


def register_network(network_cls):
    from adept.registry import REGISTRY
    REGISTRY.register_network(network_cls)


def register_submodule(submod_cls):
    from adept.registry import REGISTRY
    REGISTRY.register_submodule(submod_cls)
