"""Writes `sorted.pstats`: a cProfile capture of a Python function that sorts
twice through the builtin `sorted`, whose key callback is Python.

Run from this directory as
`python3 -c "exec(compile(open('sorted.py').read(), 'sorted.py', 'exec'))"`,
so that the capture names its functions by this file's relative name.
"""

import cProfile


def key(value):
    return -value


def workload():
    for _ in range(2):
        assert sorted([1, 2, 3], key=key) == [3, 2, 1]


profiler = cProfile.Profile()
profiler.runcall(workload)
profiler.dump_stats("sorted.pstats")
