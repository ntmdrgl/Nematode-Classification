"""MATLAB capacity, replenishment and correct output keys under uneven runtimes."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import openworm.openworm_trim as trim
import openworm.utils.openworm_alignment as alignment
from openworm.utils.openworm_scheduling import staged_results
import test_openworm_alignment as fixtures


def prepare_job(number, needs_matlab):
    return (None, (number,)) if needs_matlab else (f"ordinary-{number}", None)


class SchedulingTests(unittest.TestCase):
    def test_fills_four_slots_and_replaces_finished_job_while_first_is_running(self):
        for readers in (1, 2):
            with self.subTest(readers=readers):
                barrier = threading.Barrier(4, timeout=15)
                fifth_started = threading.Event()
                lock = threading.Lock()
                active, started = set(), []
                maximum = [0]
                anchor = [None]
                def rebuild(number):
                    with lock:
                        active.add(number)
                        started.append(number)
                        ordinal = len(started)
                        if ordinal == 1:
                            anchor[0] = number
                        maximum[0] = max(maximum[0], len(active))
                    try:
                        # Preparation futures may complete out of record order.
                        # Synchronize the first four actual starts, holding the
                        # first one until a fifth job takes a freed slot.
                        if ordinal <= 4:
                            barrier.wait()
                        if number == anchor[0]:
                            if not fifth_started.wait(15):
                                raise RuntimeError("Fifth MATLAB job did not start while the first was running")
                        if ordinal == 5:
                            with lock:
                                self.assertIn(anchor[0], active)
                            fifth_started.set()
                        return f"built-{number}"
                    finally:
                        with lock:
                            active.remove(number)
                jobs = [(0, True), *[(n, False) for n in range(100, 106)],
                        *[(n, True) for n in range(1, 8)]]
                actual = list(staged_results(prepare_job, iter(jobs), readers, rebuild))
                self.assertEqual(maximum[0], 4)
                self.assertEqual(set(started), set(range(8)))
                held_index = next(i for i, (number, build) in enumerate(jobs) if build and number == anchor[0])
                self.assertGreater(actual.index((held_index, f"built-{anchor[0]}")), 0)
                expected = {i: f"built-{n}" if build else f"ordinary-{n}"
                            for i, (n, build) in enumerate(jobs)}
                self.assertEqual(dict(actual), expected)

    def test_writes_later_record_without_waiting_and_keeps_original_keys(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixtures.AlignmentTests().fixtures(root, offset=2.5)
            later_written = threading.Event()
            write = trim.write_array
            def record_write(archive, key, values):
                write(archive, key, values)
                if key == "data_000001":
                    later_written.set()
            def rebuild(video, times, matlab, frame_workers, pool_type, progress_callback):
                number = int(video.parent.name.split("_")[-1])
                progress_callback("Starting MATLAB", 0, 12)
                if number == 2 and not later_written.wait(15):
                    raise RuntimeError("Writer stalled behind the first MATLAB job")
                progress_callback("complete", 12, 12)
                return np.full((len(times), 16), number, dtype=float)
            with patch.object(trim, "MIN_SAMPLES", 3), \
                    patch.object(trim, "write_array", side_effect=record_write), \
                    patch.object(alignment, "rebuild_record", side_effect=rebuild):
                output, = trim.trim_datasets(root, dataset="matlab_features_new", video_dir=root / "videos")
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(archive["labels"].tolist(), ["c", "d"])
                self.assertEqual(archive["record_ids"].tolist(), ["record_2", "record_3"])
                self.assertEqual(archive["record_keys"].tolist(), ["data_000000", "data_000001"])
                np.testing.assert_array_equal(archive["data_000000"], np.full((5, 16), 2.))
                np.testing.assert_array_equal(archive["data_000001"], np.full((5, 16), 3.))
                self.assertLess(archive.files.index("data_000001"), archive.files.index("data_000000"))

    def test_failed_rebuild_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixtures.AlignmentTests().fixtures(root, offset=2.5)
            output = root / "openworm_matlab_features_new_trimmed_890_dataset.npz"
            output.write_bytes(b"previous completed output")
            with patch.object(trim, "MIN_SAMPLES", 3), \
                    patch.object(alignment, "rebuild_record", side_effect=ValueError("rebuild failed")):
                with self.assertRaisesRegex(ValueError, "rebuild failed"):
                    trim.trim_datasets(root, dataset="matlab_features_new", video_dir=root / "videos")
            self.assertEqual(output.read_bytes(), b"previous completed output")
            self.assertEqual(list(root.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
