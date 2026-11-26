"""Raw mask extraction, dataset selection, and safe concurrent outputs."""
import sys
import os
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import openworm.openworm_video as video


def identity_job(job_id, slot, record, output_dir, compression, events):
    """Two jobs must reach their independent processes before either can finish."""
    directory = Path(output_dir)
    (directory / f"{job_id}.pid").write_text(str(os.getpid()))
    deadline = time.monotonic() + 15
    while len(list(directory.glob("*.pid"))) < 2:
        if time.monotonic() >= deadline:
            raise RuntimeError("Second worker did not start concurrently")
        time.sleep(.01)
    events.put((job_id, slot, 1, 1))
    return "skipped", directory / "unused.npz", 0


class VideoTests(unittest.TestCase):
    def fixture(self, root, strain="n2", record="record_1", shape=(5, 3, 4), dtype="u1"):
        directory = root / strain / record
        directory.mkdir(parents=True)
        values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        with h5py.File(directory / "worm.hdf5", "w") as source:
            source.create_dataset("mask", data=values)
            source["mask"].attrs["fps"] = 25.
            source["mask"].attrs["expected_fps"] = 30
            source.create_dataset("timestamp/raw", data=np.arange(shape[0]) + 100)
            source.create_dataset("timestamp/time", data=5 + np.arange(shape[0]) * .04)
            source.create_dataset("full_data", data=np.full((13, 3, 4), 255, dtype="u1"))
        with h5py.File(directory / "worm_features.hdf5", "w") as source:
            source.create_dataset("mask", data=np.zeros((1, 1, 1), dtype="u1"))
        return directory, values

    def test_mask_exact_pixels_shape_dtype_metadata_and_bounded_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, values = self.fixture(root, dtype=">u2")
            events = []
            with patch.object(video, "READ_BYTES", 48):
                status, output, count = video.process_record(record, root / "output", lambda *event: events.append(event))
            self.assertEqual((status, count), ("saved", 5))
            self.assertEqual(events, [(0, 5), (2, 5), (4, 5), (5, 5)])
            with np.load(output, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["mask"], values)
                self.assertEqual(archive["mask"].dtype, values.dtype)
                self.assertEqual(archive["strain"].item(), "n2")
                self.assertEqual(archive["record_id"].item(), "record_1")
                self.assertEqual(archive["fps"].item(), 25.)
                self.assertEqual(archive["source_dataset"].item(), "mask")
                np.testing.assert_array_equal(archive["frame_number"], np.arange(5) + 100)
                np.testing.assert_array_equal(archive["time_s"], 5 + np.arange(5) * .04)
                self.assertEqual(archive["expected_fps"].item(), 30)
                self.assertNotIn("full_data", archive.files)

    def test_never_falls_back_to_full_data(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, _ = self.fixture(root)
            with h5py.File(record / "worm.hdf5", "a") as source:
                del source["mask"]
            with self.assertRaisesRegex(ValueError, "Missing /mask"):
                video.process_record(record, root / "output")
            self.assertFalse((root / "output/n2/record_1.npz").exists())

    def test_existing_output_skipped_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, _ = self.fixture(root)
            _, output, _ = video.process_record(record, root / "output")
            before = output.read_bytes()
            with patch.object(video.h5py, "File", side_effect=AssertionError("Must skip before reading")):
                self.assertEqual(video.process_record(record, root / "output")[0], "skipped")
            self.assertEqual(output.read_bytes(), before)

    def test_old_full_data_export_is_replaced_with_all_mask_images(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, values = self.fixture(root)
            output = root / "output/n2/record_1.npz"
            output.parent.mkdir(parents=True)
            np.savez_compressed(output, full_data=np.ones((1, 3, 4), dtype="u1"))
            self.assertEqual(video.process_record(record, root / "output")[0], "saved")
            with np.load(output, allow_pickle=False) as archive:
                self.assertNotIn("full_data", archive.files)
                np.testing.assert_array_equal(archive["mask"], values)

    def test_failed_regeneration_preserves_old_export(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, _ = self.fixture(root)
            output = root / "output/n2/record_1.npz"
            output.parent.mkdir(parents=True)
            np.savez_compressed(output, full_data=np.ones((1, 3, 4), dtype="u1"))
            before = output.read_bytes()
            def interrupt(done, total):
                if done: raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                video.process_record(record, root / "output", interrupt)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(list(output.parent.glob("*.part")), [])

    def test_trailing_timestamps_are_not_saved_as_images(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, values = self.fixture(root)
            with h5py.File(record / "worm.hdf5", "a") as source:
                del source["timestamp/raw"]
                del source["timestamp/time"]
                source.create_dataset("timestamp/raw", data=[100, 101, 101, 103, 104, 105])
                source.create_dataset("timestamp/time", data=[5., 5.04, 5.04, 5.12, 5.16, 5.20])
            _, output, count = video.process_record(record, root / "output")
            with np.load(output, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["mask"], values)
                np.testing.assert_array_equal(archive["frame_number"], [100, 101, 101, 103, 104])
                self.assertEqual(len(archive["time_s"]), count)

    def test_features_only_record_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, _ = self.fixture(root)
            (record / "worm.hdf5").unlink()
            self.assertEqual(video.process_record(record, root / "output")[0], "skipped")
            self.assertFalse((root / "output").exists())

    def test_multiple_videos_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, _ = self.fixture(root)
            (record / "another.hdf5").touch()
            with self.assertRaisesRegex(ValueError, "exactly one video"):
                video.process_record(record, root / "output")

    def test_interruption_cleans_temporary_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, _ = self.fixture(root)
            def interrupt(done, total):
                if done: raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                video.process_record(record, root / "output", interrupt)
            self.assertEqual(list((root / "output").rglob("*.part")), [])
            self.assertEqual(list((root / "output").rglob("*.npz")), [])

    def test_invalid_image_shape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, _ = self.fixture(root, shape=(3, 4))
            with self.assertRaisesRegex(ValueError, "numeric 3D"):
                video.process_record(record, root / "output")

    def test_parallel_records_and_failure_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            records = [self.fixture(root, record=f"record_{i}")[0] for i in range(3)]
            with h5py.File(records[-1] / "worm.hdf5", "a") as source:
                del source["mask"]
            self.assertEqual(video.run_records(records, root / "output", workers=2), 1)
            for record in records[:2]:
                with np.load(root / "output/n2" / (record.name + ".npz"), allow_pickle=False) as archive:
                    self.assertEqual(archive["mask"].shape, (5, 3, 4))
            self.assertEqual(len(list((root / "output").rglob("*.npz"))), 2)
            self.assertEqual(list((root / "output").rglob("*.part")), [])

    def test_each_codec_preserves_pixels_dtype_shape_and_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, expected = self.fixture(root, dtype=">u2")
            for name, (method, _) in video.CODECS.items():
                with self.subTest(codec=name):
                    _, output, _ = video.process_record(record, root / name, compression=name)
                    with np.load(output, allow_pickle=False) as archive:
                        np.testing.assert_array_equal(archive["mask"], expected)
                        self.assertEqual(archive["mask"].dtype, expected.dtype)
                        self.assertEqual(archive["compression"].item(), name)
                        self.assertEqual(archive["compression_strategy"].item(), name)
                        np.testing.assert_array_equal(archive["time_s"], 5 + np.arange(5) * .04)
                    with ZipFile(output) as archive:
                        self.assertEqual(archive.getinfo("mask.npy").compress_type, method)

    def test_best_compares_complete_archives_and_keeps_smallest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, expected = self.fixture(root, shape=(8, 16, 16))
            original_replace = video.os.replace
            seen = {}
            def compare_before_publish(source, destination):
                paths = list(destination.parent.glob("*.part"))
                self.assertEqual(len(paths), len(video.CODECS))
                for path in paths:
                    with np.load(path, allow_pickle=False) as archive:
                        codec = archive["compression"].item()
                        np.testing.assert_array_equal(archive["mask"], expected)
                        seen[codec] = path.stat().st_size
                with np.load(source, allow_pickle=False) as archive:
                    self.assertEqual(seen[archive["compression"].item()], min(seen.values()))
                original_replace(source, destination)
            with patch.object(video.os, "replace", side_effect=compare_before_publish):
                _, output, _ = video.process_record(record, root / "output")
            self.assertEqual(set(seen), set(video.CODECS))
            self.assertEqual(output.stat().st_size, min(seen.values()))
            self.assertEqual(list(output.parent.glob("*.part")), [])

    def test_switching_compression_regenerates_previous_export(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, expected = self.fixture(root)
            _, output, _ = video.process_record(record, root / "output", compression="deflate")
            self.assertEqual(video.process_record(record, root / "output", compression="bzip2")[0], "saved")
            with ZipFile(output) as archive:
                self.assertEqual(archive.getinfo("mask.npy").compress_type, video.CODECS["bzip2"][0])
            np.testing.assert_array_equal(np.load(output, allow_pickle=False)["mask"], expected)

    def test_old_deflate_mask_export_is_recompressed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, expected = self.fixture(root)
            output = root / "output/n2/record_1.npz"
            output.parent.mkdir(parents=True)
            np.savez_compressed(output, mask=expected, source_dataset="mask")
            self.assertEqual(video.process_record(record, root / "output")[0], "saved")
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(archive["compression_strategy"].item(), "best")
                np.testing.assert_array_equal(archive["mask"], expected)

    def test_failure_of_candidate_preserves_old_output_and_cleans_every_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, expected = self.fixture(root)
            output = root / "output/n2/record_1.npz"
            output.parent.mkdir(parents=True)
            np.savez_compressed(output, mask=expected, source_dataset="mask")
            before = output.read_bytes()
            original_open = ZipFile.open
            def fail_lzma(archive, name, *args, **kwargs):
                if archive.compression == video.CODECS["lzma"][0] and name == "mask.npy":
                    raise OSError("simulated LZMA failure")
                return original_open(archive, name, *args, **kwargs)
            with patch.object(ZipFile, "open", fail_lzma):
                with self.assertRaisesRegex(OSError, "simulated LZMA failure"):
                    video.process_record(record, root / "output")
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(list(output.parent.glob("*.part")), [])

    def test_invalid_compression_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "compression"):
            video.process_record(Path("unused"), Path("unused"), compression="lossy")

    def test_workers_use_independent_processes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            records = [root / f"record_{i}" for i in range(2)]
            with patch.object(video, "process_job", identity_job):
                self.assertEqual(video.run_records(records, root, workers=2, compression="bzip2"), 0)
            pids = {int(p.read_text()) for p in root.glob("*.pid")}
            self.assertEqual(len(pids), 2)
            self.assertNotIn(os.getpid(), pids)

    def test_process_outputs_equal_serial_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            records = []
            for i in range(3):
                record, _ = self.fixture(root, record=f"record_{i}", dtype=">u2")
                records.append(record)
            self.assertEqual(video.run_records(records, root / "serial", workers=1, compression="bzip2"), 0)
            self.assertEqual(video.run_records(records, root / "parallel", workers=2, compression="bzip2"), 0)
            for record in records:
                with np.load(root / "serial/n2" / (record.name + ".npz"), allow_pickle=False) as expected, \
                        np.load(root / "parallel/n2" / (record.name + ".npz"), allow_pickle=False) as actual:
                    self.assertEqual(expected.files, actual.files)
                    for key in expected.files:
                        np.testing.assert_array_equal(actual[key], expected[key])

    def test_worker_throttles_intermediate_progress_and_preserves_endpoints(self):
        from queue import Queue
        events = Queue()
        def fake_record(record, output, callback, compression):
            self.assertEqual(compression, "bzip2")
            for done in range(4):
                callback(done, 3)
            return "skipped", Path("unused"), 0
        with patch.object(video, "process_record", side_effect=fake_record), \
                patch.object(video.time, "monotonic", side_effect=[0., .1, .2, .3]):
            video.process_job(5, 2, Path("record_5"), Path("output"), "bzip2", events)
        self.assertEqual(events.get_nowait(), (5, 2, 0, 3))
        self.assertEqual(events.get_nowait(), (5, 2, 3, 3))
        self.assertTrue(events.empty())

    def test_single_record_cli_discovers_only_requested_record(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record, _ = self.fixture(root)
            with patch.object(sys, "argv", ["openworm_video.py", "--input-dir", str(record), "--output-dir", str(root / "output"), "--workers", "3"]), \
                    patch.object(video, "run_records", return_value=0) as run:
                self.assertEqual(video.main(), 0)
            self.assertEqual(run.call_args.args, ([record], root / "output", 3, "bzip2"))
            self.assertEqual(run.call_args.kwargs["staging_dir"], Path(tempfile.gettempdir()) / "openworm-video-staging")


if __name__ == "__main__":
    unittest.main()
