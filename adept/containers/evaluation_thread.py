"""
Copyright (C) 2018 Heron Systems, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from ._base import HasAgent, LogsAndSummarizesRewards
import numpy as np
from threading import Thread
from time import time, sleep


class EvaluationThread(HasAgent, LogsAndSummarizesRewards):
    def __init__(self, agent, env, nb_env, logger, summary_writer, step_rate_limit):
        self._agent = agent
        self._environment = env
        self._nb_env = nb_env
        self._logger = logger
        self._summary_writer = summary_writer
        self._step_rate_limit = step_rate_limit
        self._thread = Thread(target=self._run)
        self._should_stop = False

    def start(self):
        self._thread.start()

    def stop(self):
        self._should_stop = True
        self._thread.join()

    def _run(self):
        next_obs = self.environment.reset()
        self.start_time = time()
        while not self._should_stop:
            sleep(1 / self._step_rate_limit)
            obs = next_obs
            actions = self.agent.act_eval(obs)
            next_obs, rewards, terminals, infos = self.environment.step(actions)

            self.agent.reset_internals(terminals)
            # Perform state updates
            terminal_rewards, terminal_infos = self.update_buffers(rewards, terminals, infos)
            self.log_episode_results(terminal_rewards, terminal_infos, self.local_step_count)
            self.write_reward_summaries(terminal_rewards, self.local_step_count)

    def log_episode_results(self, terminal_rewards, terminal_infos, step_count, initial_step_count=0):
        if terminal_rewards:
            ep_reward = np.mean(terminal_rewards)
            self.logger.info(
                'eval_frames: {} reward: {} avg_eval_fps: {}'.format(
                    step_count,
                    ep_reward,
                    (step_count - initial_step_count) / (time() - self.start_time)
                )
            )
        return terminal_rewards

    @property
    def agent(self):
        return self._agent

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, new_env):
        self._environment = new_env

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
    def summary_name(self):
        return 'reward/eval'