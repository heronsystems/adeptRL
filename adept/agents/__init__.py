from .actor_critic import ActorCritic
from .impala import ActorCriticVtrace
from ._base import EnvBase

AGENTS = {
    'ActorCritic': ActorCritic,
    'ActorCriticVtrace': ActorCriticVtrace
}
AGENT_ARGS = {
    'ActorCritic': lambda args: (
        args.nb_env, args.exp_length, args.discount, args.generalized_advantage_estimation, args.tau
    ),
    'ActorCriticVtrace': lambda args: (args.nb_env, args.exp_length, args.discount),
}
