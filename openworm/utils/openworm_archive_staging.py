"""Bounded sequential input staging shared by video assembly and trimming."""
from pathlib import Path
from queue import Empty, Full, Queue
import tempfile
import threading

from tqdm import tqdm

from openworm.utils.openworm_video_staging import StagingCancelled, copy_video, remove_job


def stage_file(record, arguments, directory, stop, callback):
    source = Path(arguments[0])
    destination = Path(directory) / source.name
    copy_video(source, destination, stop, callback)
    return (destination, *arguments[1:])


class StagedJobs:
    """One copier, one ready job, at most workers+1 staged inputs.

    The consumer releases an input only after its worker closes it. Close the
    process iterator before this context, so cancellation never deletes an input
    still open by a worker. Only this run's unique staging subfolder is removed.
    """
    def __init__(self, jobs, directory, workers, overall, prepare=stage_file):
        self.jobs, self.directory, self.workers = jobs, Path(directory).resolve(), workers
        self.overall, self.prepare = overall, prepare
        self.ready, self.events = Queue(maxsize=1), Queue()
        self.capacity = threading.BoundedSemaphore(workers + 1)
        self.stop, self.finished = threading.Event(), threading.Event()
        self.owned, self.lock = {}, threading.Lock()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="openworm-archive-stage-", dir=self.directory)
        self.run = Path(self.temporary.name)
        self.copy_id = None
        self.bar = tqdm(total=0, desc="Staging input", unit="B", unit_scale=True,
                        position=self.workers + 1, leave=False, disable=self.overall.disable)
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()
        return self

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.ready.put(item, timeout=.1)
                return True
            except Full:
                pass
        return False

    def _produce(self):
        try:
            for index, (record, arguments) in enumerate(self.jobs):
                if self.stop.is_set():
                    break
                acquired, directory = False, None
                try:
                    while not self.stop.is_set():
                        if self.capacity.acquire(timeout=.1):
                            acquired = True
                            break
                    if not acquired:
                        break
                    directory = Path(tempfile.mkdtemp(prefix=f"job-{index:06d}-", dir=self.run))
                    prepared = self.prepare(record, arguments, directory, self.stop,
                                            lambda done, total: self.events.put((index, record, done, total)))
                    with self.lock:
                        self.owned[str(prepared[0])] = directory
                    if not self._put((record, prepared, None)):
                        with self.lock:
                            self.owned.pop(str(prepared[0]), None)
                        break
                    acquired, directory = False, None
                finally:
                    if directory is not None:
                        remove_job(directory, self.run)
                    if acquired:
                        self.capacity.release()
        except StagingCancelled:
            pass
        except BaseException as error:
            self._put((None, None, error))
        finally:
            self.finished.set()

    def poll(self):
        while True:
            try:
                index, record, done, total = self.events.get_nowait()
            except Empty:
                break
            if self.copy_id != index:
                self.copy_id = index
                self.bar.reset(total=total)
                self.bar.set_description(f"Staging {record[0]}/{record[1]}")
            self.bar.update(max(0, done - self.bar.n))

    def __iter__(self):
        while not self.finished.is_set() or not self.ready.empty():
            self.poll()
            try:
                record, arguments, error = self.ready.get(timeout=.1)
            except Empty:
                continue
            if error is not None:
                raise error
            yield record, arguments
        self.poll()

    def next_nowait(self):
        self.poll()
        try:
            record, arguments, error = self.ready.get_nowait()
        except Empty:
            if self.finished.is_set() and self.ready.empty():
                raise StopIteration
            raise
        if error is not None:
            raise error
        return record, arguments

    def release(self, arguments):
        with self.lock:
            directory = self.owned.pop(str(arguments[0]), None)
        if directory is not None:
            try:
                remove_job(directory, self.run)
            finally:
                self.capacity.release()

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        self.poll()
        self.bar.close()
        # Callers have shut down workers before exiting this context.
        self.temporary.cleanup()
