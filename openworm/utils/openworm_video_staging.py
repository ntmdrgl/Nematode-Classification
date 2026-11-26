"""One sequential source copier feeding independent SSD-backed export workers."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
from pathlib import Path
from queue import Empty, Full, Queue
import shutil
import tempfile
import threading
import time

import numpy as np
from tqdm import tqdm

from openworm.openworm_dataset import PROGRESS_FORMAT
from openworm.openworm_video import is_mask_export, process_job

COPY_BYTES = 8 * 1024 ** 2


class StagingCancelled(Exception):
    pass


def remove_job(directory, run_directory):
    """Only remove one directory owned by this run, never user/source folders."""
    directory, run_directory = Path(directory).resolve(), Path(run_directory).resolve()
    if directory.parent != run_directory or not directory.name.startswith("job-"):
        raise ValueError(f"Refusing to delete a directory outside this staging run: {directory}")
    if directory.exists():
        shutil.rmtree(directory)


def copy_video(source, destination, stop, callback):
    """Copy raw file bytes sequentially; publish the staged path after close."""
    source, destination = Path(source), Path(destination)
    partial = destination.with_name(destination.name + ".copying")
    try:
        with source.open("rb", buffering=0) as original, partial.open("xb") as staged:
            before = os.fstat(original.fileno())
            callback(0, before.st_size)
            buffer = bytearray(COPY_BYTES)
            copied, last_update = 0, 0.
            while True:
                if stop.is_set():
                    raise StagingCancelled()
                size = original.readinto(buffer)
                if not size:
                    break
                if staged.write(memoryview(buffer)[:size]) != size:
                    raise OSError("Incomplete staging write")
                copied += size
                now = time.monotonic()
                if now - last_update >= .25:
                    callback(copied, before.st_size)
                    last_update = now
            after = os.fstat(original.fileno())
            if (copied != before.st_size or after.st_size != before.st_size
                    or after.st_mtime_ns != before.st_mtime_ns):
                raise OSError(f"Source video changed while being staged: {source}")
        if stop.is_set():
            raise StagingCancelled()
        os.replace(partial, destination)
        callback(copied, before.st_size)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def produce(records, output_dir, compression, directory, ready, copy_events, capacity, stop, finished):
    """The only thread that discovers/copies source video files during the run."""
    def enqueue(item):
        while not stop.is_set():
            try:
                ready.put(item, timeout=.1)
                return True
            except Full:
                pass
        return False
    try:
        for job_id, record in enumerate(records):
            if stop.is_set():
                break
            owned, acquired = None, False
            try:
                output = Path(output_dir) / record.parent.name / (record.name + ".npz")
                # Skip before source discovery/copying; completed exports cost
                # no reads from the external video drive.
                if output.is_file() and is_mask_export(output, compression):
                    if not enqueue(("skipped", job_id, record, output, None, None)):
                        break
                    continue
                videos = sorted(path for path in record.glob("*.hdf5")
                                if path.is_file() and not path.name.lower().endswith("_features.hdf5"))
                if not videos:
                    if not enqueue(("skipped", job_id, record, output, None, None)):
                        break
                    continue
                if len(videos) != 1:
                    raise ValueError(f"Expected exactly one video HDF5 in {record}; found {len(videos)}")
                original = videos[0].resolve()
                while not stop.is_set():
                    if capacity.acquire(timeout=.1):
                        acquired = True
                        break
                if not acquired:
                    break
                owned = Path(tempfile.mkdtemp(prefix=f"job-{job_id:06d}-", dir=directory))
                staged = owned / original.name
                copy_video(original, staged, stop,
                           lambda done, total: copy_events.put((job_id, record, done, total)))
                if not enqueue(("ready", job_id, record, staged, original, owned)):
                    break
                # The consumer now owns this file and its capacity reservation.
                owned, acquired = None, False
            except StagingCancelled:
                break
            except Exception as error:
                if not enqueue(("failed", job_id, record, error, None, None)):
                    break
            finally:
                if owned is not None:
                    remove_job(owned, directory)
                if acquired:
                    capacity.release()
    except BaseException as error:
        # Propagate unexpected producer/cleanup failures to the parent instead
        # of silently treating an incomplete source scan as a successful run.
        enqueue(("fatal", -1, None, error, None, None))
    finally:
        finished.set()


def run_staged(records, output_dir, workers, compression, staging_dir):
    records = list(map(Path, records))
    counts = dict(saved=0, skipped=0, failed=0)
    if not records:
        print("Saved 0; skipped 0; failed 0.")
        return 0
    workers = min(workers, len(records))
    staging_dir = Path(staging_dir).resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    active = {}
    ready, copy_events = Queue(maxsize=1), Queue()
    capacity = threading.BoundedSemaphore(workers + 1)
    stop, finished = threading.Event(), threading.Event()
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="openworm-video-", dir=staging_dir) as directory, \
            context.Manager() as manager, \
            tqdm(total=len(records), desc="Video records", unit="record", position=0,
                 smoothing=0, bar_format=PROGRESS_FORMAT) as overall, \
            tqdm(total=0, desc="Staging video", unit="B", unit_scale=True, position=workers + 1,
                 leave=False, smoothing=0, bar_format=PROGRESS_FORMAT) as copying:
        events = manager.Queue()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
        available = list(reversed(range(workers)))
        pending = {}
        copy_id = None
        producer = threading.Thread(target=produce,
                                    args=(records, output_dir, compression, directory, ready,
                                          copy_events, capacity, stop, finished), daemon=True)

        def drain():
            nonlocal copy_id
            while True:
                try:
                    job_id, slot, done, total = events.get_nowait()
                except Empty:
                    break
                if slot in active and active[slot][0] == job_id:
                    bar = active[slot][1]
                    bar.total = total
                    bar.update(max(0, done - bar.n))
            while True:
                try:
                    job_id, record, done, total = copy_events.get_nowait()
                except Empty:
                    break
                if copy_id != job_id:
                    copy_id = job_id
                    copying.reset(total=total)
                    copying.set_description(f"Staging {record.parent.name}/{record.name}")
                copying.update(max(0, done - copying.n))
                copying.set_postfix_str("Staged" if done == total else "Copying", refresh=False)

        def receive(item):
            kind, job_id, record, value, original, owned = item
            if kind == "fatal":
                raise RuntimeError(f"Staging producer failed: {value}") from value
            if kind == "failed":
                counts["failed"] += 1
                tqdm.write(f"FAILED {record}: {value}")
                overall.update(1)
            elif kind == "skipped":
                counts["skipped"] += 1
                overall.update(1)
            else:
                slot = available.pop()
                bar = tqdm(total=0, desc=f"{record.parent.name}/{record.name}", unit="image",
                           position=slot + 1, leave=False, smoothing=0, bar_format=PROGRESS_FORMAT,
                           disable=overall.disable)
                active[slot] = (job_id, bar)
                try:
                    future = pool.submit(process_job, job_id, slot, record, output_dir, compression,
                                         events, value, original)
                except BaseException:
                    bar.close()
                    active.pop(slot)
                    available.append(slot)
                    remove_job(owned, directory)
                    capacity.release()
                    raise
                pending[future] = (slot, record, owned)

        producer.start()
        try:
            while pending or not finished.is_set() or not ready.empty():
                # A full process pool leaves at most one completed copy queued.
                # This keeps staging bounded instead of copying the dataset ahead.
                while available:
                    try:
                        item = ready.get(timeout=.1 if not pending else 0)
                    except Empty:
                        break
                    receive(item)
                done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED) if pending else ((), ())
                drain()
                for future in done:
                    slot, record, owned = pending.pop(future)
                    try:
                        status, output, count = future.result()
                        if status == "saved":
                            with np.load(output, allow_pickle=False) as saved:
                                codec = saved["compression"].item()
                            tqdm.write(f"Saved {output}: {count:,} mask images, {codec}")
                        counts[status] += 1
                    except Exception as error:
                        counts["failed"] += 1
                        tqdm.write(f"FAILED {record}: {error}")
                    finally:
                        # Worker completion means its HDF5 handle is closed.
                        active.pop(slot)[1].close()
                        available.append(slot)
                        remove_job(owned, directory)
                        capacity.release()
                    overall.update(1)
                overall.set_postfix_str(f"processing {len(pending)}/{workers}; staged {ready.qsize()}", refresh=False)
        finally:
            stop.set()
            producer.join()
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            # Do not remove staged files while running workers could read them.
            for slot, record, owned in pending.values():
                remove_job(owned, directory)
            while True:
                try:
                    item = ready.get_nowait()
                except Empty:
                    break
                if item[0] == "ready":
                    remove_job(item[5], directory)
            for _, bar in active.values():
                bar.close()
            drain()
    if sum(counts.values()) != len(records):
        raise RuntimeError("Staging stopped before every input record was accounted for")
    print(f"Saved {counts['saved']}; skipped {counts['skipped']}; failed {counts['failed']}.")
    return 1 if counts["failed"] else 0
