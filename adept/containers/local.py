import torch
from time import time

from ._base import HasAgent, HasEnvironment, WritesSummaries, SavesModels, LogsAndSummarizesRewards


class Local(HasAgent, HasEnvironment, WritesSummaries, LogsAndSummarizesRewards, SavesModels):
    def __init__(
        self,
        agent,
        environment,
        make_optimizer,
        epoch_len,
        nb_env,
        logger,
        summary_writer,
        summary_frequency,
        saver
    ):
        super().__init__()
        self._agent = agent
        self._environment = environment
        self._optimizer = make_optimizer(self.network.parameters())
        self._epoch_len = epoch_len
        self._nb_env = nb_env
        self._logger = logger
        self._summary_writer = summary_writer
        self._saver = saver
        self._summary_frequency = summary_frequency

    @property
    def agent(self):
        return self._agent

    @property
    def environment(self):
        return self._environment

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def epoch_len(self):
        return self._epoch_len

    @property
    def nb_env(self):
        return self._nb_env

    @property
    def logger(self):
        return self._logger

    @property
    def summary_writer(self):
        return self._summary_writer

    @property
    def summary_frequency(self):
        return self._summary_frequency

    @property
    def saver(self):
        return self._saver

    def run(self, max_steps=float('inf'), initial_count=0):
        self.local_step_count = initial_count
        next_obs = self.environment.reset()
        self.start_time = time()
        while self.local_step_count < max_steps:
            obs = next_obs
            # Build rollout
            actions = self.agent.act(obs)
            next_obs, rewards, terminals, infos = self.environment.step(actions)
            self.agent.observe(obs, rewards, terminals, infos)

            # Perform state updates
            terminal_rewards, terminal_infos = self.update_buffers(rewards, terminals, infos)
            self.log_episode_results(terminal_rewards, terminal_infos, self.local_step_count, initial_count)
            self.write_reward_summaries(terminal_rewards, self.local_step_count)
            self.update_and_save_model(terminal_rewards, self.local_step_count, self.nb_env)

            # Learn
            if self.exp_cache.is_ready():
                self.learn(next_obs)

    def learn(self, next_obs):
        loss_dict, metric_dict = self.agent.compute_loss(self.exp_cache.read(), next_obs)
        total_loss = torch.sum(torch.stack(tuple(loss for loss in loss_dict.values())))

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.exp_cache.clear()
        self.agent.detach_internals()

        # write summaries
        self.write_summaries(total_loss, loss_dict, metric_dict, self.local_step_count)
