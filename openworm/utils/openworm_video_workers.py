"""Bounded per-record processes and parent-only progress for archive operations."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from queue import Empty
import time

from tqdm import tqdm

from openworm.openworm_dataset import PROGRESS_FORMAT


def _pack_job(function, index, arguments, directory, events, slot):
    last_update = 0.
    def report(done, total):
        nonlocal last_update
        now = time.monotonic()
        if done == 0 or done == total or now - last_update >= .25:
            events.put((index, slot, done, total))
            last_update = now
    return function(index, arguments, directory, report)


def packed_records(function, jobs, workers, directory, overall, release=None, poll=None):
    """Jobs are (record pair, arguments); yield small (index, part path, metric).

    At most one outstanding compressed record per worker plus the record being
    merged is buffered. Refill workers before yielding a completed record so
    compression continues while the parent copies compressed bytes to the NPZ.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    source_jobs = jobs
    jobs = iter(enumerate(jobs))
    active = {}
    def start(index, slot, record):
        bar = tqdm(total=0, desc=f"{record[0]}/{record[1]}", unit="frame", position=slot + 1,
                   leave=False, smoothing=0, bar_format=PROGRESS_FORMAT, disable=overall.disable)
        active[slot] = index, bar
    def advance(index, slot, done, total):
        if slot in active and active[slot][0] == index:
            bar = active[slot][1]
            bar.total = total
            bar.update(max(0, done - bar.n))
    def close(slot):
        _, bar = active.pop(slot)
        bar.close()
    try:
        if workers == 1:
            for index, (record, arguments) in jobs:
                start(index, 0, record)
                try:
                    result = function(index, arguments, directory,
                                      lambda done, total: advance(index, 0, done, total))
                finally:
                    close(0)
                    if release is not None:
                        release(arguments)
                if poll is not None:
                    poll()
                yield result
            return
        context = multiprocessing.get_context("spawn")
        with context.Manager() as manager:
            events = manager.Queue()
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
            pending = {}
            free = list(reversed(range(workers)))
            exhausted = False
            next_index = 0
            def submit_available():
                nonlocal exhausted, next_index
                while free and not exhausted:
                    try:
                        if hasattr(source_jobs, "next_nowait"):
                            record, arguments = source_jobs.next_nowait()
                            index = next_index
                            next_index += 1
                        else:
                            index, (record, arguments) = next(jobs)
                    except Empty:
                        break
                    except StopIteration:
                        exhausted = True
                        break
                    slot = free.pop()
                    start(index, slot, record)
                    pending[pool.submit(_pack_job, function, index, arguments, directory, events, slot)] = (index, slot, arguments)
            def drain():
                while True:
                    try:
                        advance(*events.get_nowait())
                    except Empty:
                        break
            try:
                while pending or not exhausted:
                    submit_available()
                    if pending:
                        done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                    else:
                        time.sleep(.05)
                        done = ()
                    drain()
                    if poll is not None:
                        poll()
                    for future in done:
                        _, slot, arguments = pending.pop(future)
                        try:
                            result = future.result()
                        finally:
                            close(slot)
                            free.append(slot)
                            if release is not None:
                                release(arguments)
                        submit_available()
                        yield result
            finally:
                for future in pending:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                if release is not None:
                    for _, _, arguments in pending.values():
                        release(arguments)
                drain()
                if poll is not None:
                    poll()
    finally:
        for _, bar in active.values():
            bar.close()
