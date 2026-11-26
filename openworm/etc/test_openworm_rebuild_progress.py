"""Parallel frame options and live progress while ordered output is blocked."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import openworm.utils.openworm_alignment as alignment
import openworm.openworm_trim as trim
from openworm.openworm_dataset import ordered_results
from openworm.utils.openworm_progress import RebuildProgress
import test_openworm_alignment as fixtures


def waiting_job(events, observed):
    events.put(("strain/record_1/worm.hdf5", "Starting MATLAB", 0, 12))
    if not observed.wait(timeout=15):
        raise RuntimeError("Parent did not display progress while awaiting the result")
    events.put(("strain/record_1/worm.hdf5", "Calculating features", 6, 12))
    events.put(("strain/record_1/worm.hdf5", "complete", 12, 12))
    return 42


class RebuildProgressTests(unittest.TestCase):
    def video(self, root):
        path = root / "worm.hdf5"
        with h5py.File(path, "w") as source:
            source.create_dataset("mask", shape=(12, 1, 1), dtype="u1")
        return path

    def test_pool_options_and_streamed_progress(self):
        for pool, matlab_pool in (("threads", "Threads"), ("processes", "Processes")):
            with self.subTest(pool=pool), tempfile.TemporaryDirectory() as d:
                video = self.video(Path(d))
                events = []
                expected = np.ones((5, 16))
                times = np.arange(5) * .04
                with patch.object(alignment.subprocess, "Popen") as launch, \
                        patch("scipy.io.loadmat", return_value={}), \
                        patch("openworm.openworm_matlab_features.matlab_executable", return_value="matlab"), \
                        patch("openworm.openworm_matlab_features.preprocess_extracted", return_value=(times, expected, None)) as preprocess:
                    process = launch.return_value.__enter__.return_value
                    process.stdout = iter(["startup log\n", "Progress: 50.0% (6/12)\n"])
                    process.wait.return_value = 0
                    actual = alignment.rebuild_record(video, times, frame_workers=3, pool_type=pool,
                                                      progress_callback=lambda *event: events.append(event))
                self.assertIn(f"inf, 32, 3, '{matlab_pool}');", launch.call_args.args[0][-1])
                self.assertEqual(events, [("Starting MATLAB", 0, 12), ("Calculating features", 6, 12),
                                          ("Resampling to reference grid", 12, 12), ("complete", 12, 12)])
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(preprocess.call_args.kwargs["sample_times"], times)

    def test_matlab_failure_keeps_output_tail(self):
        with tempfile.TemporaryDirectory() as d:
            video = self.video(Path(d))
            with patch.object(alignment.subprocess, "Popen") as launch, \
                    patch("openworm.openworm_matlab_features.matlab_executable", return_value="matlab"):
                process = launch.return_value.__enter__.return_value
                process.stdout = iter(["license unavailable\n"])
                process.wait.return_value = 1
                with self.assertRaisesRegex(ValueError, "license unavailable"):
                    alignment.rebuild_record(video, np.arange(5) * .04)

    def test_invalid_frame_count_terminates_matlab(self):
        with tempfile.TemporaryDirectory() as d:
            video = self.video(Path(d))
            with patch.object(alignment.subprocess, "Popen") as launch, \
                    patch("openworm.openworm_matlab_features.matlab_executable", return_value="matlab"):
                process = launch.return_value.__enter__.return_value
                process.stdout = iter(["Progress: 50.0% (6/99)\n"])
                with self.assertRaisesRegex(ValueError, "Unexpected MATLAB frame progress"):
                    alignment.rebuild_record(video, np.arange(5) * .04)
                process.terminate.assert_called_once()

    def test_progress_arrives_while_ordered_result_is_blocked(self):
        with tqdm(total=1, disable=True) as overall:
            with RebuildProgress(overall, workers=2) as display:
                observed = display.manager.Event()
                handle = display._event
                def acknowledge(event):
                    handle(event)
                    observed.set()
                with patch.object(display, "_event", side_effect=acknowledge):
                    self.assertEqual(list(ordered_results(waiting_job, [(display.queue, observed)], 2)), [42])
            self.assertEqual(display.finished, 1)
            self.assertEqual(display.bars, {})
            self.assertFalse(display.thread.is_alive())

    def test_pipeline_forwards_options_and_preserves_reference_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixtures.AlignmentTests().fixtures(root, offset=2.5)
            def rebuild(video, times, matlab, frame_workers, pool_type, progress_callback):
                self.assertEqual(frame_workers, 3)
                self.assertEqual(pool_type, "processes")
                progress_callback("Starting MATLAB", 0, 12)
                progress_callback("Calculating features", 6, 12)
                progress_callback("complete", 12, 12)
                return np.ones((len(times), 16))
            with patch.object(trim, "MIN_SAMPLES", 3), \
                    patch.object(alignment, "rebuild_record", side_effect=rebuild) as run:
                output, = trim.trim_datasets(root, dataset="matlab_features_new", video_dir=root / "videos",
                                            frame_workers=3, pool_type="processes")
            self.assertEqual(run.call_count, 2)
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(archive["labels"].tolist(), ["c", "d"])
                self.assertEqual(archive["data_000000"].shape, (5, 16))

    def test_failed_rebuild_closes_its_progress_bar(self):
        with tempfile.TemporaryDirectory() as d:
            video = self.video(Path(d))
            from queue import Queue
            events = Queue()
            with patch.object(alignment, "alignment_plan", return_value=(None, np.arange(5) * .04)), \
                    patch.object(alignment, "rebuild_record", side_effect=ValueError("broken")):
                with self.assertRaisesRegex(ValueError, "broken"):
                    trim.aligned_record("unused", "key", 12, 5, 5, "features", video, None,
                                        progress_queue=events)
            self.assertEqual(events.get_nowait(), (str(video), "failed", 0, 0))

    def test_invalid_options_fail_before_opening_archives(self):
        for options, message in ((dict(frame_workers=0), "frame_workers"),
                                 (dict(pool_type="invalid"), "pool type"),
                                 (dict(dataset="eigenworms", frame_workers=2), "requires")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                trim.trim_datasets("unused", **options)

    def test_cli_passes_frame_options(self):
        with patch.object(sys, "argv", ["trim", "--dataset", "matlab_features_new", "--frame-workers", "3", "--pool-type", "processes"]), \
                patch.object(trim, "trim_datasets") as run:
            trim.main()
        self.assertEqual(run.call_args.args[-2:], (3, "processes"))


if __name__ == "__main__":
    unittest.main()
