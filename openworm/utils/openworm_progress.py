"""Display active MATLAB rebuild progress in the parent process."""
import multiprocessing
from queue import Empty, Queue
import threading

from tqdm import tqdm

from openworm.openworm_dataset import PROGRESS_FORMAT


class RebuildProgress:
    def __init__(self, overall, workers):
        self.overall = overall
        self.workers = workers
        self.manager = None
        self.queue = None
        self.bars = {}
        self.finished = 0
        self.record = ""
        self.lock = threading.Lock()
        self.error = None

    def __enter__(self):
        if self.workers > 1:
            self.manager = multiprocessing.get_context("spawn").Manager()
            self.queue = self.manager.Queue()
        else:
            self.queue = Queue()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def _postfix(self):
        self.overall.set_postfix_str(
            f"MATLAB active: {len(self.bars)}, rebuilt: {self.finished}; {self.record}")

    def wrote_record(self, record):
        with self.lock:
            self.record = f"wrote {record[0]}/{record[1]}"
            self._postfix()

    def _event(self, event):
        video, stage, done, total = event
        if video not in self.bars and stage not in ("complete", "failed"):
            used = {slot for _, slot in self.bars.values()}
            slot = next(i for i in range(1, self.workers + 1) if i not in used)
            # Video names can be long; strain/record identifies the active job.
            from pathlib import Path
            path = Path(video)
            bar = tqdm(total=total, desc=f"MATLAB {path.parent.parent.name}/{path.parent.name}",
                       unit="frame", position=slot, leave=False, smoothing=0,
                       bar_format=PROGRESS_FORMAT, disable=self.overall.disable)
            self.bars[video] = (bar, slot)
        if video in self.bars:
            bar, _ = self.bars[video]
            bar.set_postfix_str(stage, refresh=False)
            bar.update(max(0, done - bar.n))
            if stage in ("complete", "failed"):
                bar.close()
                del self.bars[video]
        if stage == "complete":
            self.finished += 1
        self._postfix()

    def _monitor(self):
        try:
            while True:
                try:
                    event = self.queue.get(timeout=1)
                except Empty:
                    # Keep elapsed times moving during MATLAB startup/pool
                    # creation and between its frame progress messages.
                    with self.lock:
                        for bar, _ in self.bars.values():
                            bar.refresh()
                        self.overall.refresh()
                    continue
                if event is None:
                    break
                with self.lock:
                    self._event(event)
        except Exception as error:
            self.error = error

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.queue.put(None)
            self.thread.join()
            for bar, _ in self.bars.values():
                bar.close()
        finally:
            if self.manager is not None:
                self.manager.shutdown()
        if self.error is not None and exc_type is None:
            raise self.error
