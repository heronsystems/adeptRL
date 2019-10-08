
import time
from collections import namedtuple

import ray


@ray.remote(num_cpus=2)
class Worker:
    State = namedtuple("State", ['asdf'])
    def __init__(self):
        pass

    def sleep(self, t):
        time.sleep(t)
        print(f'slept for {t}')

    def sleep5(self):
        time.sleep(5)

    def sleep10(self):
        time.sleep(10)


def main():
    ray.init(num_cpus=4)
    remote_worker = Worker.remote()

    t_zero = time.time()

    f5 = remote_worker.sleep5.remote()
    f10 = remote_worker.sleep10.remote()

    ray.wait([f5, f10], num_returns=2)
    print('delta', time.time() - t_zero)


def main_async():
    import asyncio
    from ray.experimental import async_api
    ray.init(num_cpus=4)
    remote_worker = Worker.remote()
    loop = asyncio.get_event_loop()

    t_zero = time.time()

    tasks = [async_api.as_future(remote_worker.sleep.remote(i)) for i in range(1,3)]
    loop.run_until_complete(
        asyncio.gather(tasks)
    )

    print('delta', time.time() - t_zero)


if __name__ == '__main__':
    main()
