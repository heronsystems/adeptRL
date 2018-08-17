import enum


class P2PCommunicationProtocol:
    def __init__(self, comm):
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

    @property
    def next_dest(self):
        raise NotImplementedError

    @property
    def next_source(self):
        raise NotImplementedError


class P2PDeterministic(P2PCommunicationProtocol):
    """
    Deterministically iterates the passed in order over and over.
    No partner switching. For partner switching see P2PDynamic
    """
    def __init__(self, *args, order):
        super().__init__(*args)
        self.order = order
        self.order_size = len(order)
        self._current_step = -1

    @property
    def next_dest(self):
        self._current_step = (self._current_step + 1) % self.order_size
        return (self.rank + self.order[self._current_step]) % self.size

    @property
    def next_source(self):
        return (self.rank - self.order[self._current_step]) % self.size


class P2PDynamic(P2PCommunicationProtocol):
    """
    Iterates the passed in order. But after each iteration changes the node layout
    so that each node will have new partners.
    All nodes must start with the same seed
    """
    def __init__(self, *args, order, shared_seed):
        super().__init__(*args)
        self.order = order
        self._seed = shared_seed
        self.random = np.random.RandomState(shared_seed)
        self._current_offset = self.random.randint(0, self.size - 1)
        self.order_size = len(order)
        self._current_step = -1

    @property
    def next_dest(self):
        self._current_step = (self._current_step + 1)
        if self._current_step % self.order_size == 0:
            self._current_offset = self.random.randint(0, self.size - 1)
        self._current_step %= self.order_size
        return (self.rank + self.order[self._current_step] + self._current_offset) % self.size

    @property
    def next_source(self):
        return (self.rank - self.order[self._current_step] - self._current_offset) % self.size


class P2PReversingRoundRobin(P2PCommunicationProtocol):
    def __init__(self, *args):
        super().__init__(*args)
        self._current_step = 0
        self._dir = True

    @property
    def next_dest(self):
        self._current_step += 1
        if self._current_step == self.size:
            self._dir = not self._dir
            self._current_step = 0

        if self._dir:
            return (self.rank + 1) % self.size
        else:
            return (self.rank - 1) % self.size

    @property
    def next_source(self):
        if self._dir:
            return (self.rank - 1) % self.size
        else:
            return (self.rank + 1) % self.size


class SquareDirections(enum.IntEnum):
    FORWARD = 0
    ACROSS = 1
    BACKWARD = 2


class P2PReversingSquare(P2PCommunicationProtocol):
    def __init__(self, *args):
        super().__init__(*args)
        self._dir = -1

    @property
    def next_dest(self):
        self._dir = self._next_dir()
        if self._dir == SquareDirections.FORWARD:
            return (self.rank + 1) % self.size
        elif self._dir == SquareDirections.ACROSS:
            return (self.rank + 2) % self.size
        else:  # SquareDirections.BACKWARD
            return (self.rank - 1) % self.size

    @property
    def next_source(self):
        if self._dir == SquareDirections.FORWARD:
            return (self.rank - 1) % self.size
        elif self._dir == SquareDirections.ACROSS:
            return (self.rank - 2) % self.size
        else:  # Direction is backward so listen to the node in front of me
            return (self.rank + 1) % self.size

    def _next_dir(self):
        return (self._dir + 1) % 3


def P2PBestProtocol(comm, shared_seed):
    size = comm.Get_size()
    if size == 2:
        return P2PDeterministic(comm, order=[1])
    if size == 3:
        return P2PDeterministic(comm, order=[2, 1])
    if size == 4:
        # 7.96[2, 1]
        # 7.96[2, 1, 3]
        # 7.96[2, 3, 1]
        # 7.97[3, 2, 1]
        return P2PDeterministic(comm, order=[3, 2, 1])
    # after 4 nodes it's normally best to get random new partners
    if size == 5:
        return P2PDynamic(comm, order=[3, 4, 1, 2], shared_seed=shared_seed)
        # 82.25[3, 4, 1, 2]
        # 82.25[3, 4, 2, 1]
    if size == 6:
        return P2PDynamic(comm, order=[1, 4, 2, 5, 3], shared_seed=shared_seed)
    #     113.47[1, 4, 2, 5, 3]
    #     113.47[1, 4, 5, 2, 3]
    #     113.47[4, 1, 2, 5, 3]
    #     113.47[4, 1, 5, 2, 3]
    if size == 7:
        return P2PDynamic(comm, order=[4, 6, 2, 5, 1, 3], shared_seed=shared_seed)
    #     117.69[4, 6, 2, 5, 1, 3]
    #     117.69[6, 4, 2, 5, 1, 3]
    if size == 8:
        # this is actually gossip grad https://arxiv.org/pdf/1803.05880.pdf
        return P2PDynamic(comm, order=[1, 2, 4], shared_seed=shared_seed)


from copy import deepcopy
import numpy as np


def propagation_model(num):
    # assuming we are node 0
    # there are (n-1)! / 2 different costs
    frontier = set(range(1, num))
    options = _recurse(frontier)
    options = options[:len(options) // 2]

    # there are still duplicates ie:

    # evaluate the options based on how much repetition there is
    costs = []
    reversing_costs = []
    for o in options:
        path_cost = [[] for _ in range(len(o) + 1)]
        reversing_path_cost = [[] for _ in range(len(o) + 1)]
        # iterate the path
        for current_step, relative_index in enumerate(o):
            # trace that path from this point on
            # this is assuming a non reversing pattern
            tmp_index = relative_index
            path_to_this_one = o[current_step + 1:]
            path_to_this_one.extend(o[:current_step + 1])
            for second_step, second_path in enumerate(path_to_this_one):
                second_index = (tmp_index + second_path) % len(path_cost)
                path_cost[second_index].append(0.5 ** (second_step + 1))
                tmp_index = second_index

            # this is a reversing pattern
            tmp_index = relative_index
            path_to_this_one = o[current_step + 1:]
            path_to_this_one.extend(o[:current_step + 1:-1])
            for second_step, second_path in enumerate(path_to_this_one):
                second_index = (tmp_index + second_path) % len(path_cost)
                reversing_path_cost[second_index].append(0.5 ** (second_step + 1))
                tmp_index = second_index
        costs.append(path_cost)
        reversing_costs.append(reversing_path_cost)

    zipped = []
    for i in range(len(options)):
        zipped.append((options[i] + options[i][:-1], [sum(c) for c in costs[i]]))
        zipped.append((options[i] + options[i][::-1][1:], [sum(c) for c in reversing_costs[i]]))
    sort_zip = sorted(zipped, key=lambda x: (x[1][0], np.max(x[1])))
    for thing in sort_zip:
        if np.all(np.asarray(thing[1]) < 1):
            # print('solution')
            print(thing[0])
            print(thing[1], np.sum(thing[1]), np.prod(thing[1]), np.std(thing[1]))

    return options


def _recurse(frontier, depth=0):
    if len(frontier) > 1:
        completions = []
        for choice in list(frontier):
            new_frontier = deepcopy(frontier)  # type: set
            new_frontier.remove(choice)
            possible_completions = _recurse(new_frontier, depth + 1)
            if isinstance(possible_completions, list):
                for pc in possible_completions:
                    thing = [choice]
                    thing.extend(pc)
                    completions.append(thing)
            else:  # max depth returns int
                completions.append([choice, possible_completions])
        return completions
    else:
        return list(frontier)[0]


def eval_cost(num, initial_values, final_value, path, nsteps):
    cost = 0
    current_values = deepcopy(initial_values)
    for n in range(nsteps):
        # iterate the path
        rand_partner = np.random.randint(0, num)
        for p in path:
            # all sends are async
            # p += rand_partner
            tmp_values = deepcopy(current_values)
            for i in range(num):
                recv_index = (i + p) % num
                current_values[recv_index] = (tmp_values[i] + tmp_values[recv_index]) / 2
                cost += 1
                if np.all(np.isclose(current_values, final_value)):
                    return cost
    return cost


def mixture_model(num, ntrials, nsteps=100):
    # assuming we are node 0
    # there are (n-1)! / 2 different costs
    options = []
    for possible_num in range(2, num + 1):
        frontier = set(range(1, possible_num))
        os = _recurse(frontier)
        if not isinstance(os, list):
            os = [[os]]
        options.extend(os)
    options.append([1, 3, 4, 2, 4, 3])

    costs = [0] * len(options)
    for n in range(ntrials):
        node_values = np.asarray(np.random.randint(1, 100, size=num), dtype=np.float32)
        # node_values = list(range(num))
        # node_values = [0] + [1] + [0] * (num - 2)
        true_value = np.mean(node_values)

        for o_ind, o in enumerate(options):
            cost = eval_cost(num, node_values, true_value, o, nsteps)
            costs[o_ind] += cost
    zipped = list(zip(costs, options))
    sort_zip = sorted(zipped, key=lambda x: (x[0], len(x[1])))
    for c, o in sort_zip:
        print(c / ntrials, o)


if __name__ == '__main__':
    # propagation_model(3)
    mixture_model(8, 1000)
