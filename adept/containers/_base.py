import abc
from time import time

import numpy as np
import torch


class CountsRewards(abc.ABC):
    """
    Maintains a buffer that increments and resets episode rewards for multiple environments.
    """
    # define lazy properties
    _episode_reward_buffer = None
    _local_step_count = 0
    _start_time = time()

    # ABSTRACT PROPS
    @property
    @abc.abstractmethod
    def nb_env(self):
        raise NotImplementedError

    # DEFINED PROPS
    @property
    def local_step_count(self):
        return self._local_step_count

    @local_step_count.setter
    def local_step_count(self, step_count):
        self._local_step_count = step_count

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, st):
        self._start_time = st

    @property
    def episode_reward_buffer(self):
        if self._episode_reward_buffer is None:
            self._episode_reward_buffer = torch.zeros(self.nb_env)
        return self._episode_reward_buffer

    @episode_reward_buffer.setter
    def episode_reward_buffer(self, updated_buffer):
        self._episode_reward_buffer = updated_buffer

    def update_buffers(self, rewards, terminals, infos):
        self.episode_reward_buffer += torch.tensor(rewards).float()
        self.local_step_count += self.nb_env
        terminal_rewards = []
        for ep_reward, done, info in zip(self.episode_reward_buffer, terminals, infos):
            if done and info:
                terminal_rewards.append(ep_reward.item())
                ep_reward.zero_()
        return terminal_rewards


class LogsRewards(CountsRewards, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def logger(self):
        raise NotImplementedError

    def log_episode_results(self, terminal_rewards, step_count, initial_step_count=0):
        if terminal_rewards:
            ep_reward = np.mean(terminal_rewards)
            self.logger.info(
                'train_frames: {} reward: {} avg_train_fps: {}'.format(
                    step_count,
                    ep_reward,
                    (step_count - initial_step_count) / (time() - self.start_time)
                )
            )
        return terminal_rewards


class LogsAndSummarizesRewards(LogsRewards, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def summary_writer(self):
        raise NotImplementedError

    def write_reward_summaries(self, terminal_rewards, step_count):
        if terminal_rewards:
            ep_reward = np.mean(terminal_rewards)
            self.summary_writer.add_scalar('reward', ep_reward, step_count)
        return terminal_rewards


class WritesSummaries(abc.ABC):
    _prev_summary_time = time()

    # ABSTRACT PROPS
    @property
    @abc.abstractmethod
    def network(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def summary_writer(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def summary_frequency(self):
        raise NotImplementedError

    # DEFINED PROPS
    @property
    def prev_summary_time(self):
        return self._prev_summary_time

    @prev_summary_time.setter
    def prev_summary_time(self, pst):
        self._prev_summary_time = pst

    def write_summaries(self, total_loss, loss_dict, metric_dict, step_count):
        cur_time = time()
        elapsed = cur_time - self.prev_summary_time

        if elapsed > self.summary_frequency:
            self.prev_summary_time = cur_time

            writer = self.summary_writer
            writer.add_scalar('macro_loss/total_loss', total_loss.item(), step_count)
            for l_name, loss in loss_dict.items():
                writer.add_scalar('loss/' + l_name, loss.item(), step_count)
            for m_name, metric in metric_dict.items():
                writer.add_scalar('metric/' + m_name, metric.item(), step_count)
            for p_name, param in self.network.named_parameters():
                p_name = p_name.replace('.', '/')
                writer.add_scalar(p_name, torch.norm(param).item(), step_count)
                if param.grad is not None:
                    writer.add_scalar(p_name + '.grad', torch.norm(param.grad).item(), step_count)


class SavesModels(abc.ABC):
    @property
    @abc.abstractmethod
    def epoch_len(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def network(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def optimizer(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def saver(self):
        raise NotImplementedError

    def update_and_save_model(self, terminal_rewards, step_count, delta_step_count):
        """
        :param terminal_rewards: list of episode rewards for all the envs that hit a terminal state
        :param step_count: current step count to check if epoch has been crossed
        :param delta_step_count: number of steps taken since last update
        :return:
        """
        # update buffer if new high score achieved
        if terminal_rewards:
            ep_reward = np.mean(terminal_rewards)
            self.saver.append_if_better(ep_reward, self.network, self.optimizer)

        # save models if we cross save freq threshold
        for i in range(delta_step_count):
            step_number = step_count + (i + 1)
            if step_number % self.epoch_len == 0:
                self.saver.write_state_dicts(step_number)
        return terminal_rewards


class HasAgent(abc.ABC):
    @property
    @abc.abstractmethod
    def agent(self):
        raise NotImplementedError

    @property
    def network(self):
        return self.agent.network

    @property
    def exp_cache(self):
        return self.agent.exp_cache


class HasEnvironment(abc.ABC):
    @property
    @abc.abstractmethod
    def environment(self):
        raise NotImplementedError


class MPIProc:
    def receive(self):
        raise NotImplementedError

    def submit(self):
        raise NotImplementedError


class AppliesGrads(abc.ABC):
    @property
    @abc.abstractmethod
    def optimizer(self):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_gradients(self, gradients):
        raise NotImplementedError
