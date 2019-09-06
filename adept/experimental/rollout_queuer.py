import conncurrent.futures
import ray


class RolloutWorkerManager:
    def __init__(self, workers, min_batch, async_queue=False):
        self.workers = workers
        self.min_batch = min_batch
        self.async_queue = async_queue
        if not async_queue and min_batch != len(workers):
            raise AttributeError("If synchronous min_batch must equal the number of workers")

        self.futures = [w.run.remote() for w in self.workers]
        self.future_inds = [w for w in range(len(self.workers))]

    def _background_queing_thread(self):
        while not self._done:
            ready_ids, remaining_ids = ray.wait(self.futures, num_returns=1)

            # get object and add to queue
            rollouts = ray.get(ready_ids)
            self._add_to_queue(rollouts)

            # remove from futures
            self._idle_workers = []
            for ready in ready_ids:
                index = self.futures.index(ready)
                del self.futures[index]
                self._idle_workers.append(self.future_inds[index])
                del self.future_inds[index]

            # if async tell the worker(s) to start another rollout
            if self.async_queue:
                self._restart_idle_workers()

        # done, wait for all remaining to finish
        dones, not_dones = ray.wait(self.futures, len(self.futures), timeout=5)

        if len(not_dones) > 0:
            print('WARNING: Not all rollout workers finished')

    def get(self):
        batch = self._queue.wait()

    def resume(self):
        """
            Used when queue is synchronous to restart idle workers
        """
        assert self.async_queue
        self._restart_idle_workers()

    def stop(self):
        self._should_stop = True

    def _restart_idle_workers(self):
        del_inds = []
        for d_ind, w_ind in enumerate(self._idle_workers):
            worker = self.workers[w_ind]
            self.futures.append(worker.run.remote())
            self.future_inds.append(w_ind)
            del_inds.append(d_ind)

        for d in del_inds:
            del self._idle_workers[d]

  # def _batch_add_thread(
        # self, listen_source, worker_wait_time, worker_timestep,
        # max_items_in_queue
    # ):
        # """
            # A thread that listens to a mpi worker and
            # appends to the shared self.recieved_rollouts list
        # """
        # # iterative wait time to make sure the host doesn't get behind and not
        # # be able to recover
        # worker_wait_time.fill(0.0)
        # # setup rollout buffers
        # rollout_buffer = np.empty(self.rollout_flattener.total_size, np.float32)
        # # timestep is a np array so the value is correctly shared
        # worker_timestep.fill(0)

        # # thread loop as long as master says we're not done
        # while not self._threads_should_be_done:
            # # possible wait if host is slow
            # while len(
                # self.received_rollouts
            # ) > max_items_in_queue and not self._threads_should_be_done:
                # # random wait time to ensure that threads can uniformly add
                # # when max_queue size is reached
                # how_long_to_wait = float(np.random.uniform(0.1, 1))
                # time.sleep(how_long_to_wait)
                # worker_wait_time += how_long_to_wait

            # # break after possible waiting
            # if self._threads_should_be_done:
                # break

            # # setup listener for rollout, Recv is blocking but okay on thread
            # self.comm.Recv(
                # rollout_buffer, source=listen_source, tag=MpiMessages.SEND
            # )

            # # rollout is in buffer, copy into individual parts
            # unflattened_rollout = self.rollout_flattener.unflatten(
                # rollout_buffer
            # )
            # recv_timestep = unflattened_rollout.pop()

            # # to dict of tensors
            # rollout = {
                # # act on host must check terminal to determine if internals need to be reset, MUCH faster to do on cpu
                # k: torch.from_numpy(v).to(self.agent.device) if k != 'terminals' else torch.from_numpy(v)
                # for k, v in zip(
                    # self.sorted_keys,
                    # unflattened_rollout  # timestep has been popped
                # )
            # }
            # self.received_rollouts.append(rollout)

            # worker_timestep.fill(
                # recv_timestep[0]
            # )  # recv_timestep is a single value array

            # # upon recieving batch, send ack
            # self.comm.isend(
                # self._threaded_global_step(),
                # dest=listen_source,
                # tag=MpiMessages.SEND_ACK
            # )
        # print('Thread listening for {} exiting'.format(listen_source))

    # def _threaded_global_step(self):
        # return sum(self._thread_steps)[0]  # numpy array

    # def _saver_thread(self):
        # next_save_step = self.save_interval
        # try:
            # while not self._threads_should_be_done:
                # current_step = self._threaded_global_step()
                # if current_step > next_save_step:
                    # self.saver.save_state_dicts(
                        # self.network,
                        # int(current_step),
                        # optimizer=self.optimizer
                    # )
                    # next_save_step += self.save_interval
                # time.sleep(1)
        # except Exception as e:
            # print('Error saving', e)

        # # final save
        # self.saver.save_state_dicts(
            # self.network,
            # int(self._threaded_global_step()),
            # optimizer=self.optimizer
        # )

    # def run(
        # self,
        # num_rollouts_in_batch,
        # max_items_in_queue,
        # max_steps=float('inf'),
        # dynamic=False,
        # min_dynamic_batch=0
    # ):
        # from threading import Thread
        # size = self.comm.Get_size()

        # # setup a shared rollout list across threads
        # # python lists are GIL safe
        # self.received_rollouts = []
        # # shared done flag and steps
        # self._threads_should_be_done = False
        # self._thread_steps = [np.ones((1)) for i in range(1, size)]
        # self._thread_wait_times = [np.zeros((1)) for i in range(1, size)]
        # # start batch adding threads
        # threads = [
            # Thread(
                # target=self._batch_add_thread,
                # args=(
                    # i,
                    # self._thread_wait_times[i - 1],
                    # self._thread_steps[i - 1],
                    # max_items_in_queue,
                # )
            # ) for i in range(1, size)
        # ]
        # for t in threads:
            # t.start()

        # # start saver thread
        # saver_thread = Thread(target=self._saver_thread)
        # saver_thread.start()

        # # this runs a "busy" loop assuming that loss.backward takes longer than
        # # accumulating batches removing the need for the threads to notify of
        # # new batches
        # variable_buffer = np.empty(
            # self.variable_flattener.total_size, np.float32
        # )
        # start_time = time.time()
        # number_of_rollouts_waiting = 0
        # while self._threaded_global_step() < max_steps:
            # len_received_rollouts = len(self.received_rollouts)
            # # rollouts waiting?
            # if len_received_rollouts > 0:
                # # dynamic requires at least one rollout
                # if dynamic:
                    # do_batch = len_received_rollouts >= min_dynamic_batch
                    # # limit batch size to max of num_rollouts_in_batch
                    # batch_slice = min(
                        # (len_received_rollouts, num_rollouts_in_batch)
                    # )
                # else:
                    # do_batch = len_received_rollouts >= num_rollouts_in_batch
                    # batch_slice = num_rollouts_in_batch

                # if do_batch:
                    # try:
                        # number_of_rollouts_waiting += len_received_rollouts

                        # # pop everything from list starting first to last
                        # popped_rollouts = self.received_rollouts[0:batch_slice]
                        # # clear
                        # self.received_rollouts = self.received_rollouts[
                            # batch_slice:]

                        # # convert list of dict to dict of list
                        # rollouts = listd_to_dlist(popped_rollouts)

                        # loss_dict, metric_dict = self.agent.compute_loss(
                            # rollouts
                        # )
                        # total_loss = torch.sum(
                            # torch.stack(
                                # tuple(loss for loss in loss_dict.values())
                            # )
                        # )

                        # self.optimizer.zero_grad()
                        # total_loss.backward()
                        # self.optimizer.step()

                        # # write summaries
                        # self.write_summaries(
                            # total_loss, loss_dict, metric_dict,
                            # self._threaded_global_step()
                        # )

                        # # send new variables
                        # new_variables_flat = self.variable_flattener.flatten(
                            # self.get_parameters_numpy(), buffer=variable_buffer
                        # )
                        # self.comm.Ibcast(new_variables_flat, root=0)
                        # self.sent_parameters_count += 1

                        # if self.sent_parameters_count % self.summary_steps == 0:
                            # print('=' * 40)
                            # print(
                                # 'Train Per Second', self.sent_parameters_count /
                                # (time.time() - start_time)
                            # )
                            # print(
                                # 'Avg Queue Length', number_of_rollouts_waiting /
                                # self.sent_parameters_count
                            # )
                            # print(
                                # 'Current Queue Length',
                                # len(self.received_rollouts)
                            # )
                            # print(
                                # 'Thread wait times', [
                                    # '{:.2f}'.format(x[0])
                                    # for x in self._thread_wait_times
                                # ]
                            # )
                            # print('=' * 40)
                    # except Exception as e:
                        # import traceback
                        # print('Host error: {}'.format(e))
                        # print(traceback.format_exc())
                        # print('Trying to stop gracefully')
                        # break

        # # cleanup threads
        # print('Host stopping threaded batch recvs')
        # self._threads_should_be_done = True
        # [t.join() for t in threads]

        # # cleanup mpi
        # self.stop_mpi_workers()

        # # join saver after mpi workers are done to prevent infinite waiting
        # saver_thread.join()
