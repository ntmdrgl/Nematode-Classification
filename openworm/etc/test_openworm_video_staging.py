"""Sequential source copies, parallel SSD processing, provenance and cleanup."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import openworm.openworm_video as video
import openworm.utils.openworm_video_staging as staging
import test_openworm_video as fixtures


def cleanup_job(job_id, slot, record, output_dir, compression, events, video_file, source_file):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    if job_id == 0:
        (output_dir / "first-staged-path").write_text(str(video_file))
    else:
        deadline = time.monotonic() + 15
        marker = output_dir / "first-staged-path"
        while not marker.exists() or Path(marker.read_text()).exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Finished record's staged file was retained until run end")
            time.sleep(.01)
    return "skipped", output_dir / "unused.npz", 0


def overlap_job(job_id, slot, record, output_dir, compression, events, video_file, source_file):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    if job_id == 0:
        (output_dir / "first-processing").touch()
        deadline = time.monotonic() + 15
        while not (output_dir / "second-copied").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Staging failed to overlap record processing")
            time.sleep(.01)
    return video.process_job(job_id, slot, record, output_dir, compression, events, video_file, source_file)


class StagingTests(unittest.TestCase):
    def fixture(self, root, record="record_1"):
        return fixtures.VideoTests().fixture(root, record=record, dtype=">u2")

    def test_staged_outputs_equal_direct_and_preserve_original_source_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            records = [self.fixture(root, f"record_{i}")[0] for i in range(3)]
            stage = root / "ssd"
            stage.mkdir(); (stage / "keep.txt").write_text("user-owned")
            self.assertEqual(video.run_records(records, root / "direct", 1, "bzip2"), 0)
            self.assertEqual(video.run_records(records, root / "staged", 2, "bzip2", staging_dir=stage), 0)
            for record in records:
                with np.load(root / "direct/n2" / (record.name + ".npz"), allow_pickle=False) as expected, \
                        np.load(root / "staged/n2" / (record.name + ".npz"), allow_pickle=False) as actual:
                    self.assertEqual(set(actual.files), set(expected.files))
                    for key in expected.files:
                        np.testing.assert_array_equal(actual[key], expected[key])
                    self.assertEqual(actual["source_file"].item(), str((record / "worm.hdf5").resolve()))
                self.assertTrue((record / "worm.hdf5").exists())
                self.assertTrue((record / "worm_features.hdf5").exists())
            self.assertEqual([p.name for p in stage.iterdir()], ["keep.txt"])

    def test_staged_worker_never_discovers_original_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); record, values = self.fixture(root)
            original = (record / "worm.hdf5").resolve()
            stage = root / "stage.hdf5"
            staging.copy_video(original, stage, threading.Event(), lambda *args: None)
            with patch.object(Path, "glob", side_effect=AssertionError("Worker touched source record")):
                _, output, _ = video.process_record(record, root / "out", compression="bzip2",
                                                    video_file=stage, source_file=original)
            with np.load(output, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["mask"], values)
                self.assertEqual(archive["source_file"].item(), str(original))

    def test_completed_outputs_skipped_before_source_discovery_or_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); record, _ = self.fixture(root)
            video.process_record(record, root / "out", compression="bzip2")
            with patch.object(staging, "copy_video", side_effect=AssertionError("Unexpected source copy")), \
                    patch.object(Path, "glob", side_effect=AssertionError("Unexpected source discovery")):
                self.assertEqual(video.run_records([record], root / "out", 2, "bzip2", staging_dir=root / "ssd"), 0)
            self.assertEqual(list((root / "ssd").iterdir()), [])

    def test_source_copy_is_sequential_and_overlaps_processing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            records = [self.fixture(root, f"record_{i}")[0] for i in range(2)]
            output = root / "out"; output.mkdir()
            original_copy = staging.copy_video
            calls, active, maximum = [], [0], [0]
            def copy_one(source, destination, stop, callback):
                active[0] += 1
                maximum[0] = max(maximum[0], active[0])
                try:
                    calls.append(source.parent.name)
                    if source.parent.name == "record_1":
                        deadline = time.monotonic() + 15
                        while not (output / "first-processing").exists():
                            if time.monotonic() > deadline:
                                raise RuntimeError("Processing did not start during staging")
                            time.sleep(.01)
                    result = original_copy(source, destination, stop, callback)
                    if source.parent.name == "record_1":
                        (output / "second-copied").touch()
                    return result
                finally:
                    active[0] -= 1
            with patch.object(staging, "copy_video", side_effect=copy_one), \
                    patch.object(staging, "process_job", overlap_job):
                self.assertEqual(video.run_records(records, output, 2, "bzip2", staging_dir=root / "ssd"), 0)
            self.assertEqual(calls, ["record_0", "record_1"])
            self.assertEqual(maximum[0], 1)
            self.assertEqual(list((root / "ssd").iterdir()), [])

    def test_staged_file_is_deleted_as_soon_as_its_job_finishes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            records = [self.fixture(root, f"record_{i}")[0] for i in range(2)]
            with patch.object(staging, "process_job", cleanup_job):
                self.assertEqual(video.run_records(records, root / "out", 2, "bzip2", staging_dir=root / "ssd"), 0)
            self.assertEqual(list((root / "ssd").iterdir()), [])
            for record in records:
                self.assertTrue((record / "worm.hdf5").exists())

    def test_bad_mask_and_copy_failure_clean_staging_and_continue(self):
        for failure in ("mask", "copy"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                records = [self.fixture(root, f"record_{i}")[0] for i in range(2)]
                if failure == "mask":
                    with h5py.File(records[0] / "worm.hdf5", "a") as source:
                        del source["mask"]
                original_copy = staging.copy_video
                def copy(source, destination, stop, callback):
                    if failure == "copy" and source.parent.name == "record_0":
                        destination.with_name(destination.name + ".copying").write_bytes(b"unfinished")
                        raise OSError("simulated copy failure")
                    return original_copy(source, destination, stop, callback)
                with patch.object(staging, "copy_video", side_effect=copy):
                    self.assertEqual(video.run_records(records, root / "out", 2, "bzip2", staging_dir=root / "ssd"), 1)
                self.assertTrue((root / "out/n2/record_1.npz").exists())
                self.assertFalse((root / "out/n2/record_0.npz").exists())
                self.assertEqual(list((root / "ssd").iterdir()), [])
                for record in records:
                    self.assertTrue((record / "worm.hdf5").exists())

    def test_cancelled_copy_preserves_source_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / "source.hdf5"; target = root / "copy.hdf5"
            values = bytes(range(256)) * 100
            source.write_bytes(values)
            stop = threading.Event()
            def cancel(done, total):
                if done: stop.set()
            with patch.object(staging, "COPY_BYTES", 1024), self.assertRaises(staging.StagingCancelled):
                staging.copy_video(source, target, stop, cancel)
            self.assertEqual(source.read_bytes(), values)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name(target.name + ".copying").exists())

    def test_parent_interrupt_waits_for_workers_then_cleans_staged_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); record, _ = self.fixture(root)
            before = (record / "worm.hdf5").read_bytes()
            with patch.object(staging, "wait", side_effect=KeyboardInterrupt()), self.assertRaises(KeyboardInterrupt):
                video.run_records([record], root / "out", 1, "bzip2", staging_dir=root / "ssd")
            self.assertEqual((record / "worm.hdf5").read_bytes(), before)
            self.assertEqual(list((root / "ssd").iterdir()), [])
            self.assertEqual(list((root / "out").rglob("*.part")), [])

    def test_unexpected_producer_cleanup_failure_is_reported_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); record, _ = self.fixture(root)
            with patch.object(staging, "copy_video", side_effect=OSError("copy failed")), \
                    patch.object(staging, "remove_job", side_effect=OSError("cleanup failed")), \
                    self.assertRaisesRegex(RuntimeError, "Staging producer failed"):
                video.run_records([record], root / "out", 1, "bzip2", staging_dir=root / "ssd")
            self.assertTrue((record / "worm.hdf5").exists())
            self.assertEqual(list((root / "ssd").iterdir()), [])

    def test_cleanup_refuses_source_or_user_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); original = root / "original"; original.mkdir()
            (original / "keep").write_text("untouched")
            stage = root / "run"; stage.mkdir()
            with self.assertRaisesRegex(ValueError, "Refusing"):
                staging.remove_job(original, stage)
            self.assertTrue((original / "keep").exists())

    def test_cli_defaults_and_configurable_staging_folder(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); record, _ = self.fixture(root)
            for options, expected in ((["--staging-dir", str(root / "ssd")], root / "ssd"),
                                      (["--no-staging"], None)):
                with patch.object(sys, "argv", ["video", "--input-dir", str(record), *options]), \
                        patch.object(video, "run_records", return_value=0) as run:
                    self.assertEqual(video.main(), 0)
                self.assertEqual(run.call_args.args[2:], (24, "bzip2"))
                self.assertEqual(run.call_args.kwargs["staging_dir"], expected)


if __name__ == "__main__":
    unittest.main()
