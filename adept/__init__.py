from adept.registry import REGISTRY


def register_agent(agent_cls):
    REGISTRY.register_agent(agent_cls)


def register_actor(actor_cls):
    REGISTRY.register_actor(actor_cls)


def register_exp(exp_cls):
    REGISTRY.register_exp(exp_cls)


def register_learner(learner_cls):
    REGISTRY.register_learner(learner_cls)


def register_env(env_cls, env_ids):
    REGISTRY.register_env(env_cls, env_ids)


def register_reward_norm(rwd_norm_cls):
    REGISTRY.register_reward_normalizer(rwd_norm_cls)


def register_network(network_cls):
    REGISTRY.register_network(network_cls)


def register_submodule(submod_cls):
    REGISTRY.register_submodule(submod_cls)
