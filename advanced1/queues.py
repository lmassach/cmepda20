#!/usr/bin/env python3
import multiprocessing as mp
import random
import time


def f(pos, name):
    time.sleep(random.random())
    msg = f"Hello {name}"
    output.put((pos, msg))


if __name__ == "__main__":
    output = mp.Queue()
    processes = [mp.Process(target=f, args=(x, "World")) for x in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    results = [output.get() for _ in range(4)]
    print(results)
