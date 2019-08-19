from adept.agents.agent_module import AgentModule
from adept.agents.actor_critic import ActorCritic

AGENT_REG = [
    ActorCritic
]

AGENT_EVAL_LOOKUP = [

]


def agent_eval_lookup():
    from adept.actor import ACRolloutActorEval
    return {
        ActorCritic.__name__: ACRolloutActorEval
    }
