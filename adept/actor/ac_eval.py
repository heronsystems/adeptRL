from collections import OrderedDict

from adept.actor import ActorModule
from adept.actor.base.ac_helper import ACActorHelperMixin


class ACActorEval(ActorModule, ACActorHelperMixin):
    args = {}

    @classmethod
    def from_args(cls, args, action_space):
        return cls(action_space)

    @staticmethod
    def output_space(action_space):
        head_dict = {'critic': (1,), **action_space}
        return head_dict

    def process_predictions(self, preds, available_actions):
        actions = OrderedDict()

        for key in self.action_keys:
            logit = self.flatten_logits(preds[key])

            softmax = self.softmax(logit)
            action = self.select_action(softmax)

            actions[key] = action.cpu()
        return actions, {}

    @staticmethod
    def _exp_spec(rollout_len, batch_sz, obs_space, act_space, internal_space):
        return {}
