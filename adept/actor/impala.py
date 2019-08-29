from .base import ActorModule


class ImpalaWorkerActor(ActorModule):

    @staticmethod
    def output_space(action_space):
        pass

    def from_args(self, args, action_space):
        pass

    def process_predictions(self, preds, available_actions):
        pass
