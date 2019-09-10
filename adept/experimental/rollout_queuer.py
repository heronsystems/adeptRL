import queue
import ray
import threading
import torch


class RolloutQueuerAsync:
    def __init__(self, workers, num_rollouts, queue_max_size):
        self.workers = workers
        self.num_rollouts = num_rollouts
        self.queue_max_size = queue_max_size

        self.futures = [w.run.remote() for w in self.workers]
        self.future_inds = [w for w in range(len(self.workers))]
        self._should_stop = True
        self.rollout_queue = queue.Queue(self.queue_max_size)

    def _background_queing_thread(self):
        while not self._should_stop:
            ready_ids, remaining_ids = ray.wait(self.futures, num_returns=1)

            # if ray returns an empty list that means all objects have been gotten
            # this should never happen?
            if len(ready_ids) == 0:
                print('WARNING: ray returned no ready rollouts')
            # otherwise rollout was returned
            else:
                # get object and add to queue
                rollouts = ray.get(ready_ids)
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
        dones, not_dones = ray.wait(self.futures, len(self.futures), timeout=5.0)

        if len(not_dones) > 0:
            print('WARNING: Not all rollout workers finished')

    def _add_to_queue(self, rollouts):
        for r in rollouts:
            self.rollout_queue.put(r, timeout=5.0)

    def start(self):
        self._should_stop = False
        self.background_thread = threading.Thread(target=self._background_queing_thread)
        self.background_thread.start()

    def get(self):
        rollouts = [self.rollout_queue.get(True) for _ in range(self.num_rollouts)]

        # aggregate into batch
        batch = {}
        # TODO: this assumes all rollouts have the same keys
        for k in rollouts[0].keys():
            # cat over batch dimension
            if isinstance(rollouts[0][k], torch.Tensor):
                v_list = [r[k] for r in rollouts]
                agg = torch.cat(v_list, dim=1)
            elif isinstance(rollouts[0][k], dict):
                # cat all elements of dict
                agg = {}
                for r_key in rollouts[0][k].keys():
                    agg[r_key] = torch.cat([r[k][r_key] for r in rollouts], dim=1)
            batch[k] = agg

        return batch

    def stop(self):
        self._should_stop = True

        # try to join background thread
        self.background_thread.join()

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

