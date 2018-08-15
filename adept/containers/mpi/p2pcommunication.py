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


class P2PRoundRobin(P2PCommunicationProtocol):
    @property
    def next_dest(self):
        return (self.rank + 1) % self.size

    @property
    def next_source(self):
        return (self.rank - 1) % self.size


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


def P2PBestProtocol(comm):
    size = comm.Get_size()
    if size == 2:
        return P2PRoundRobin(comm)
    if size == 3:
        return P2PReversingRoundRobin(comm)
    if size == 4:
        return P2PReversingSquare(comm)
