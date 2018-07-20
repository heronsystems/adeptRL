import abc


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


class TrainAgent(abc.ABC):
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
