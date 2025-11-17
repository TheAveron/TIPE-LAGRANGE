"""profiler: the profilers decorators to use for profiling functions

functions:

- profile: for profiling not async functions
- profile_async: for profiling async functions"""

import cProfile
import io
import pstats
import tracemalloc


def profile_async(fnc):
    """Profile: a decorator for any functions to profile them"""

    async def inner(*args, **kwargs):
        print(f"Profiling asyncroneous {fnc.__name__}...")

        pr = cProfile.Profile()
        pr.enable()
        retval = await fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def profile(fnc):
    """profile: a decorator for any functions to profile them"""

    def inner(*args, **kwargs):
        print(f"Profiling {fnc.__name__}...")

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def profile_with_memory(fnc):
    """profile: a decorator for any functions to profile them with time and memory usage"""

    def inner(*args, **kwargs):
        print(f"Profiling with memory {fnc.__name__}...")
        tracemalloc.start()

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(25)
        print(s.getvalue())

        print(f"Current memory usage: {current / 1024:.2f} KB")
        print(f"Peak memory usage: {peak / 1024:.2f} KB")

        return retval

    return inner
