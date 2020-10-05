#!/usr/bin/env python3
"""Timer for assegnamento1.py

Usage: ./assegnamento1_timer.py [args for assegnamento1.py ...]
"""

import subprocess
import re
import math
import time
import sys
import os
import statistics

MAX_RUNS = 100
MAX_TIME = 15

RE_RUNTIME = re.compile(r'Done in ([0-9\.]+) seconds')


if __name__ == '__main__':
    start_time = time.time()
    times = []
    for i in range(MAX_RUNS):
        if os.name == 'nt':
            args = ['python', 'assegnamento1.py']
        else:
            args = ['./assegnamento1.py']
        args.extend(sys.argv[1:])
        cp = subprocess.run(args, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, encoding='utf8')
        times.append(float(RE_RUNTIME.search(cp.stdout).group(1)))
        if time.time() - start_time > MAX_TIME:
            break

    avg = statistics.mean(times)
    std = statistics.stdev(times, avg)
    print(f"t = {avg:.3f} +- {std:.3f} s ({len(times)} runs)")
