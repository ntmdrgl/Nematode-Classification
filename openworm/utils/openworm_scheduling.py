"""Bounded reading and a separate MATLAB pool, without ordered-result stalls."""
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import ExitStack
import multiprocessing


def staged_results(prepare, jobs, workers, rebuild, rebuild_workers=4):
    """Yield (original index, values) as ready; keep at most 4 rebuilds running.

    Preparation returns (values, None) for exact rows or (None, rebuild_args).
    Bound outstanding read results and queued rebuild descriptors separately.
    Single-reader mode uses one thread so it can keep finding MATLAB jobs while
    previous rebuilds run; multiple readers retain per-process archive caching.
    """
    if workers < 1 or rebuild_workers < 1:
        raise ValueError("Worker counts must be positive")
    jobs = iter(enumerate(jobs))
    reads, rebuilds = {}, {}
    waiting = deque()
    exhausted = False
    queue_limit = 2 * rebuild_workers
    with ExitStack() as resources:
        if workers == 1:
            readers = ThreadPoolExecutor(max_workers=1)
        else:
            readers = ProcessPoolExecutor(max_workers=workers,
                                          mp_context=multiprocessing.get_context("spawn"))
        resources.callback(readers.shutdown, wait=True, cancel_futures=True)
        builders = ThreadPoolExecutor(max_workers=rebuild_workers)
        resources.callback(builders.shutdown, wait=True, cancel_futures=True)
        try:
            while not exhausted or reads or rebuilds or waiting:
                while waiting and len(rebuilds) < queue_limit:
                    index, arguments = waiting.popleft()
                    rebuilds[builders.submit(rebuild, *arguments)] = index
                while (not exhausted and len(reads) < workers
                       and len(waiting) + len(rebuilds) < queue_limit):
                    try:
                        index, arguments = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    reads[readers.submit(prepare, *arguments)] = index
                pending = (*reads, *rebuilds)
                if not pending:
                    continue
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    if future in reads:
                        index = reads.pop(future)
                        values, arguments = future.result()
                        if arguments is None:
                            yield index, values
                        else:
                            waiting.append((index, arguments))
                    else:
                        index = rebuilds.pop(future)
                        yield index, future.result()
        finally:
            for future in (*reads, *rebuilds):
                future.cancel()
