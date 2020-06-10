import zmq
import os
import time
from collections import deque

WORLD_SIZE = int(os.environ["WORLD_SIZE"])
GLOBAL_RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
NB_NODE = int(os.environ["NB_NODE"])
LOCAL_SIZE = WORLD_SIZE // NB_NODE

if __name__ == "__main__":
    if LOCAL_RANK == 0:
        context = zmq.Context()
        h_to_w = context.socket(zmq.PUBLISH)
        w_to_h = context.socket(zmq.PULL)

        h_to_w.bind("tcp://*:5556")
        w_to_h.bind("tcp://*:5557")

        step_count = 0
        nb_batch = 2
        # while step_count < 100:
        #     q, q_lookup = deque(), set()
        #     while len(q) < nb_batch:
        #         for i, hand in enumerate(handles):
        #             if i not in q_lookup:

    else:
        context = zmq.Context()
        h_to_w = context.socket(zmq.SUBSCRIBE)
        w_to_h = context.socket(zmq.PUSH)

        h_to_w.connect("tcp://localhost:5556")
        w_to_h.connect("tcp://localhost:5557")

        done = False

        while not done:

            print("worker received")

    time.sleep(1)

    # Host event loop
    # check for rollouts
    # batch rollouts
    # tell q workers to do another
    # learn on batch
    # send new model

    # Worker event loop
    # step the actor, write to exp
    # if new model, receive new model params
    # if exp ready, notify host
    # wait for host to be ready for another

    # Commands:
    # CALC_EXPS
    # GET_ROLLOUT_i
    # GET_
