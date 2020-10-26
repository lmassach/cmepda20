#!/usr/bin/env python3
import threading
import time


def f(wait):
    # time.sleep(wait)
    for x in range(int(wait * 1e8)):
        x = 2 * x
    print(f"Thread {threading.current_thread().name} waiting {wait} s")


if __name__ == "__main__":
    t1 = threading.Thread(target=f, name="t1", args=(1,))
    t2 = threading.Thread(target=f, name="t2", args=(.1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
