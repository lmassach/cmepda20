#!/usr/bin/env python3
from multiprocessing import Process
import os


def f(name):
    print()
    print(f"Function {name} with PID {os.getpid()} parent {os.getppid()}")
    print(f"Still {name} with PID {os.getpid()}")
    if not name.endswith("-bis"):
        f(f"{name}-bis")


if __name__ == "__main__":
    print(f"Main process with PID {os.getpid()}")
    f("zero")
    p = Process(target=f, args=("one",))
    p.start()
    p.join()
    print("END")
