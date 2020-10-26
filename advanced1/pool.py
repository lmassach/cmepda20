#!/usr/bin/env python3
import multiprocessing as mp
import os
import time
import random


def f(x):
    print(f"PID = {os.getpid()}, PPID = {os.getppid()}, x = {x}, step = 1")
    time.sleep(random.random() / 10)
    print(f"PID = {os.getpid()}, PPID = {os.getppid()}, x = {x}, step = 2")
    return x ** 3


if __name__ == "__main__":
    pool = mp.Pool(processes=4)
    results = pool.map(f, range(1, 7))
    print("PROVA")
    print(results)

    print()
    print()
    results = pool.map_async(f, range(1, 7))
    print("PROVA")
    print(results.get())
