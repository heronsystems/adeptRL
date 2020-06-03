# Copyright (C) 2018 Heron Systems, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import random
from time import time

import queue
import ray
import threading
import torch


class RolloutQueuerAsync:
    def __init__(
        self,
        workers,
        num_rollouts,
        queue_max_size,
        timeout=15.0,
        shared_exp=None,
        p_draw_earlier_exp=0.1,
        draw_earlier_exp_if_rollout_queue_empty=True,
    ):
        self.workers = workers
        self.num_rollouts = num_rollouts
        self.queue_max_size = queue_max_size
        self.queue_timeout = timeout

        self.futures = [w.run.remote() for w in self.workers]
        self.future_inds = [w for w in range(len(self.workers))]
        self._should_stop = True
        self.rollout_queue = queue.Queue(self.queue_max_size)
        self._worker_wait_time = 0
        self._host_wait_time = 0

        # Here, we store an experience replay cache. When get() is invoked, we
        # probabilistically either get() from the queue or retrieve an earlier instance.
        # The advantage of this approach is that workers won't continue to produce
        # episodes in the background unless absolutely needed (this is to avoid) memory
        # from exploding.
        #
        # The idea is that workers can continue to push results to the queue, and we can
        # continue having a maximum queue length; and, workers can thus be idle when the
        # queue is really large.
        self.shared_exp = shared_exp
        self.p_draw_earlier_exp = p_draw_earlier_exp
        self.draw_earlier_exp_if_rollout_queue_empty = (
            draw_earlier_exp_if_rollout_queue_empty
        )

    def _background_queing_thread(self):
        while not self._should_stop:
            ready_ids, remaining_ids = ray.wait(self.futures, num_returns=1)

            # if ray returns an empty list that means all objects have been gotten
            # this should never happen?
            if len(ready_ids) == 0:
                print("WARNING: ray returned no ready rollouts")
            # otherwise rollout was returned
            else:
                for ready in ready_ids:
                    # get object and add to queue
                    rollouts = ray.get(ready)
                    # this will block if queue is at max size
                    self._add_to_queue(rollouts)

                # remove from futures
                self._idle_workers = []
                for ready in ready_ids:
                    index = self.futures.index(ready)
                    del self.futures[index]
                    self._idle_workers.append(self.future_inds[index])
                    del self.future_inds[index]

                # tell the worker(s) to start another rollout
                self._restart_idle_workers()

        # done, wait for all remaining to finish
        dones, not_dones = ray.wait(
            self.futures, len(self.futures), timeout=self.queue_timeout
        )

        if len(not_dones) > 0:
            print("WARNING: Not all rollout workers finished")

    def _add_to_queue(self, rollout):
        st = time()
        self.rollout_queue.put(rollout, timeout=self.queue_timeout)
        et = time()
        self._worker_wait_time += et - st

    def start(self):
        self._should_stop = False
        self.background_thread = threading.Thread(
            target=self._background_queing_thread
        )
        self.background_thread.start()

    def sample_from_earlier_exp(self):
        rollout = self.shared_exp.sample()
        return (
            rollout["rollout"],
            rollout["metas"]["terminal_rewards"],
            rollout["metas"]["terminal_infos"],
        )

    def draw_from_rollout_queue(self):
        datum = self.rollout_queue.get(True)
        rollout, terminal_rewards, terminal_infos = (
            datum["rollout"],
            datum["terminal_rewards"],
            datum["terminal_infos"],
        )

        return rollout, terminal_rewards, terminal_infos

    def get(self):
        st = time()
        # For each of our samples, sample from earlier experience if:
        # 1. The shared experience cache has elements AND
        #   a. We're asked randomly asked to draw from earlier experience OR
        #   b. The rollout queue is empty, and we're authorized to draw from earlier
        #      experience in the event that the rollout queue is empty
        # Otherwise, draw from the rollout_queue.
        rollouts = []
        terminal_rewards = []
        terminal_infos = []
        new_samples = []
        for _ in range(self.num_rollouts):
            if len(self.shared_exp) > 0 and (
                random.random() < self.p_draw_earlier_exp
                or (
                    self.rollout_queue.empty()
                    and self.draw_earlier_exp_if_rollout_queue_empty
                )
            ):
                sample = self.sample_from_earlier_exp()
            else:
                sample = self.draw_from_rollout_queue()
                new_samples.append(sample)

            rollouts.append(sample[0])
            terminal_rewards.append(sample[1])
            terminal_infos.append(sample[2])

        et = time()

        # Add all of the new samples to shared_exp; note that we don't do this as a part
        # of draw_from_rollout_queue() to avoid sample_from_earlier_exp() sampling from
        # a rollout that was just added.
        for sample in new_samples:
            self.shared_exp.start_next_rollout()
            self.shared_exp.write_exps([sample[0]])
            self.shared_exp.write_meta(
                {"terminal_rewards": sample[1], "terminal_infos": sample[2]}
            )

        self._host_wait_time += et - st

        return rollouts, terminal_rewards, terminal_infos

    def close(self):
        self._should_stop = True

        # try to join background thread
        self.background_thread.join()

    def metrics(self):
        return {
            "Host wait time": self._host_wait_time,
            "Worker wait time": self._worker_wait_time,
        }

    def _restart_idle_workers(self):
        del_inds = []
        for d_ind, w_ind in enumerate(self._idle_workers):
            worker = self.workers[w_ind]
            self.futures.append(worker.run.remote())
            self.future_inds.append(w_ind)
            del_inds.append(d_ind)

        for d in del_inds:
            del self._idle_workers[d]

    def __len__(self):
        return len(self.rollout_queue)
