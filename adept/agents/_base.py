import abc

import torch


class EnvBase(abc.ABC):
    @abc.abstractmethod
    def preprocess_logits(self, logits):
        """
        Do any last-minute processing on logits before process_logits does anything
        :param logits:
        :return:
        """
        raise NotImplementedError

    def process_logits(self, logits, obs, deterministic):
        raise NotImplementedError


class Agent(abc.ABC):
    """
    An Agent interacts with the environment and accumulates experience.
    """
    @property
    @abc.abstractmethod
    def exp_cache(self):
        """Get experience cache"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self):
        """Get experience cache"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def network(self):
        """Get network"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def internals(self):
        """A list of internals"""
        raise NotImplementedError

    @internals.setter
    @abc.abstractmethod
    def internals(self, new_internals):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def output_shape(action_space):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def act_eval(self, obs):
        raise NotImplementedError

    def obs_to_pathways(self, obs, device):
        visual_batch = []
        discrete_batch = []
        for channels in zip(*obs.values()):
            visual_channels = [
                channel_tensor.to(device).float()
                for channel_tensor in channels
                if (isinstance(channel_tensor, torch.Tensor) and (channel_tensor.dim() == 3))
            ]
            visual_tensor = torch.cat(visual_channels) if visual_channels else None

            discrete_channels = [
                channel_tensor.to(device).float()
                for channel_tensor in channels
                if (isinstance(channel_tensor, torch.Tensor) and (channel_tensor.dim() == 1))
            ]
            discrete_tensor = torch.cat(discrete_channels) if discrete_channels else None

            if visual_tensor is not None:
                visual_batch.append(visual_tensor)
            if discrete_tensor is not None:
                discrete_batch.append(discrete_tensor)
        return {
            'visual': torch.stack(visual_batch) if visual_batch else torch.tensor([]),
            'discrete': torch.stack(discrete_batch) if discrete_batch else torch.tensor([])
        }

    def observe(self, obs, rewards, terminals, infos):
        self.exp_cache.write_env(obs, rewards, terminals, infos)
        self.reset_internals(terminals)
        return rewards, terminals, infos

    def reset_internals(self, terminals):
        for i, terminal in enumerate(terminals):
            if terminal:
                self._reset_internals_at_index(i)

    def _reset_internals_at_index(self, env_idx):
        for k, v in self.network.new_internals(self.device).items():
            self.internals[k][env_idx] = v

    def detach_internals(self):
        for k, vs in self.internals.items():
            self.internals[k] = [v.detach() for v in vs]
