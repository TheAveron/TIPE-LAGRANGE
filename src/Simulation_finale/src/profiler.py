"""profiler: the profilers decorators to use for profiling functions

functions:

- profile: for profiling not async functions
- profile_async: for profiling async functions"""

import cProfile
import io
import pstats
import tracemalloc
import threading
import time
import matplotlib.pyplot as plt


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

        ps.print_stats(100)
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
        ps.print_stats()
        print(s.getvalue())

        print(f"Current memory usage: {current / 1024:.2f} KB")
        print(f"Peak memory usage: {peak / 1024:.2f} KB")

        return retval

    return inner


def profile_with_memory_graph(fnc):
    """Decorator to profile execution time and memory usage, with memory graph"""

    def inner(*args, **kwargs):
        print(f"Profiling with memory {fnc.__name__}...")

        tracemalloc.start()

        memory_samples = []
        time_samples = []
        start_time = time.time()
        stop_event = threading.Event()

        def sample_memory():
            while not stop_event.is_set():
                current, peak = tracemalloc.get_traced_memory()
                memory_samples.append(current / 1024)  # KB
                time_samples.append(time.time() - start_time)
                time.sleep(0.01)  # sampling interval (10ms)

        sampler_thread = threading.Thread(target=sample_memory)
        sampler_thread.start()

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()

        stop_event.set()
        sampler_thread.join()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        print(s.getvalue())

        print(f"Current memory usage: {current / 1024:.2f} KB")
        print(f"Peak memory usage: {peak / 1024:.2f} KB")

        # Plot memory usage
        plt.figure()
        plt.plot(time_samples, memory_samples)
        plt.xlabel("Time (s)")
        plt.ylabel("Memory usage (KB)")
        plt.title(f"Memory allocation over time: {fnc.__name__}")
        plt.grid(True)
        plt.show()

        return retval

    return inner
