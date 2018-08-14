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
        if self.rank == self.size - 1:
            return 0
        else:
            return self.rank + 1

    @property
    def next_source(self):
        if self.rank == 0:
            return self.size - 1
        else:
            return self.rank - 1


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
            if self.rank == self.size - 1:
                return 0
            else:
                return self.rank + 1
        else:
            if self.rank == 0:
                return self.size - 1
            else:
                return self.rank - 1

    @property
    def next_source(self):
        if self._dir:
            if self.rank == 0:
                return self.size - 1
            else:
                return self.rank - 1
        else:
            if self.rank == self.size - 1:
                return 0
            else:
                return self.rank + 1


def P2PBestProtocol(comm):
    size = comm.Get_size()
    if size == 2:
        return P2PRoundRobin(comm)
    if size == 3:
        return P2PReversingRoundRobin(comm)
    # if size == 4:
    #     return P2PReversingSquare(comm)
