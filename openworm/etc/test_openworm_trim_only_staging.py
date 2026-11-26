"""SSD staging and final-only mask video output from all input forms."""
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZIP_BZIP2

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import openworm.openworm_dataset as dataset
import openworm.openworm_trim as trim
import openworm.utils.openworm_video_trim as video_trim
from openworm.utils.openworm_archive_staging import StagedJobs
from openworm.utils.openworm_video_workers import packed_records
import test_openworm_video_dataset as fixtures


def tiny_pack(index, arguments, directory, report):
    assert Path(arguments[0]).read_bytes() == b"source"
    report(1, 1)
    output = Path(directory) / f"{index}.part"
    output.write_bytes(b"packed")
    return index, output, 0.


class TrimOnlyTests(unittest.TestCase):
    def fixtures(self, root):
        fixtures.VideoTrimTests().fixtures(root)
        raw_root = root / "raw"
        for path in (root / "video").glob("*/record_*.npz"):
            with np.load(path, allow_pickle=False) as source, ZipFile(path) as zipped:
                arrays = {key: source[key] for key in source.files}
                method = zipped.getinfo("mask.npy").compress_type
            raw = raw_root / path.parent.name / path.stem / "worm.hdf5"
            raw.parent.mkdir(parents=True)
            with h5py.File(raw, "w") as source:
                images = source.create_dataset("mask", data=arrays["mask"], compression="gzip")
                images.attrs["fps"] = arrays["fps"].item()
                source.create_dataset("timestamp/raw", data=arrays["frame_number"])
                source.create_dataset("timestamp/time", data=arrays["time_s"])
                source.create_dataset("full_data", data=np.zeros((1, 2, 3), dtype="u1"))
            arrays["source_file"] = np.asarray(str(raw.resolve()))
            with ZipFile(path, "w", compression=method) as output:
                for key, value in arrays.items():
                    dataset.write_array(output, key, value)
        dataset.build_datasets(root, "video")
        return raw_root

    def compare(self, expected_path, actual_path, hdf5=False):
        with np.load(expected_path, allow_pickle=False) as expected, np.load(actual_path, allow_pickle=False) as actual:
            self.assertEqual(set(expected.files), set(actual.files))
            for key in expected.files:
                if hdf5 and key.startswith("compression_"):
                    self.assertEqual(actual[key].item(), "bzip2")
                    continue
                np.testing.assert_array_equal(actual[key], expected[key])
                self.assertEqual(actual[key].dtype, expected[key].dtype)

    def test_staged_records_archive_and_hdf5_have_same_alignment_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = self.fixtures(root)
            ssd = root / "ssd"; ssd.mkdir(); (ssd / "keep.txt").write_text("user-owned")
            with patch.object(trim, "MIN_SAMPLES", 3):
                expected, = trim.trim_datasets(root, root / "baseline", dataset="video", video_source="archive")
                for source in ("records", "archive", "hdf5"):
                    for workers in (1, 2):
                        with self.subTest(source=source, workers=workers):
                            actual, = trim.trim_datasets(root, root / f"{source}-{workers}", dataset="video",
                                                        workers=workers, video_source=source, video_dir=raw,
                                                        staging_dir=ssd)
                            self.compare(expected, actual, source == "hdf5")
                            self.assertEqual([p.name for p in ssd.iterdir()], ["keep.txt"])

    def test_correctly_aligned_export_reuses_compressed_pixels(self):
        from test_openworm_parallel_archives import payload
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            source = root / "video/c/record_2.npz"
            values = np.arange(30, dtype=">u2").reshape(5, 2, 3)
            fixtures.mask_export(root, "c", "record_2", values, ZIP_BZIP2,
                                 np.arange(5), 10 + np.arange(5) / 25)
            features = root / "openworm/c/record_2/worm_features.hdf5"
            with h5py.File(features, "w") as hdf:
                table = np.zeros(5, dtype=[("timestamp", "f8")])
                table["timestamp"] = np.arange(5)
                hdf.create_dataset("features_timeseries", data=table).attrs["fps"] = 25.
            arguments = (source, ("c", "record_2"), features, 5, 5, None, "exact", .02)
            with patch.object(video_trim, "copy_images", side_effect=AssertionError("Recompressed aligned stack")):
                _, result, error = video_trim.pack_trim_export_record(0, arguments, root, lambda *args: None)
            self.assertEqual(error, 0.)
            self.assertEqual(payload(source, "mask.npy"), payload(result, "data_000000.npy"))
            with np.load(result, allow_pickle=False) as output:
                self.assertEqual(output["data_000000"].dtype, values.dtype)
                np.testing.assert_array_equal(output["data_000000"], values)
                np.testing.assert_array_equal(output["source_index_000000"], np.arange(5))

    def test_nearest_default_handles_slow_and_missing_source_frames(self):
        for gap in (False, True):
            with self.subTest(gap=gap), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                features = root / "worm_features.hdf5"
                table = np.zeros(50, dtype=[("timestamp", "f8")])
                table["timestamp"] = np.arange(50)
                with h5py.File(features, "w") as source:
                    source.create_dataset("features_timeseries", data=table).attrs["fps"] = 1 / .0416
                raw = np.arange(50)
                if gap:
                    raw = raw[(raw < 20) | (raw >= 30)]
                times = raw * .0416
                values = np.repeat(raw.astype("u1"), 6).reshape(-1, 2, 3)
                path = fixtures.mask_export(root, "a", "record_1", values, ZIP_BZIP2, raw, times)
                from openworm.utils.openworm_alignment import reference_timestamps, sample_count
                n = sample_count(reference_timestamps(features)[1])
                args = (path, ("a", "record_1"), features, n, n, None, "nearest", None)
                _, result, error = video_trim.pack_trim_export_record(0, args, root, lambda *a: None)
                target = np.arange(n) * .04
                distances = abs(times[:, None] - target[None, :])
                with np.load(result, allow_pickle=False) as output:
                    selected = output["source_index_000000"]
                    valid = selected >= 0
                    np.testing.assert_allclose(distances[selected[valid], np.flatnonzero(valid)],
                                               distances[:, valid].min(axis=0), atol=1e-12)
                    np.testing.assert_array_equal(output["data_000000"][valid], values[selected[valid]])
                    self.assertEqual(len(selected), n)
                    self.assertGreater(error, .02)
                    if gap:
                        self.assertTrue((~valid).any())
                        self.assertFalse(output["data_000000"][~valid].any())
                        self.assertTrue(np.isnan(output["alignment_error_s_000000"][~valid]).all())
                        self.assertLess(error, .021)
                    self.assertLess(len(np.unique(selected[valid])), valid.sum())
                for alignment, limit in (("nearest", .02), ("exact", None)):
                    with self.assertRaisesRegex(ValueError, "exceeds"):
                        video_trim.prepare_export_record(*args[:-2], alignment, limit)

    def test_gap_rules_match_matlab_features_including_quarter_second_boundary(self):
        from openworm.openworm_matlab_features import FEATURES, preprocess_extracted
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            features = root / "worm_features.hdf5"
            table = np.zeros(100, dtype=[("timestamp", "f8")])
            table["timestamp"] = np.arange(100)
            with h5py.File(features, "w") as source:
                source.create_dataset("features_timeseries", data=table).attrs["fps"] = 100.
            raw = np.array([0, 24, 49, 75, 99])
            times = 10 + raw / 100
            values = np.arange(1, 31, dtype=">u2").reshape(5, 2, 3)
            path = fixtures.mask_export(root, "a", "record_1", values, ZIP_BZIP2, raw, times)
            args = (path, ("a", "record_1"), features, 25, 25, None, "nearest", None)
            indices, metadata = video_trim.prepare_export_record(*args)
            extracted = dict(feature_names=FEATURES, feature_values=np.ones((5, 16)),
                             observed_mask=np.ones((5, 16), bool), time_s=times, frame_number=raw)
            _, matlab, statuses = preprocess_extracted(extracted, sample_times=metadata["target_time_s"])
            np.testing.assert_array_equal(metadata["frame_status"], statuses[:, 0])
            black = indices < 0
            np.testing.assert_array_equal(black, matlab[:, 0] == 0)
            self.assertFalse(black[6])  # An actual frame at the gap endpoint stays real.
            self.assertTrue(black[7:19].all())  # 0.25 s and 0.26 s gaps remain black.
            self.assertEqual(metadata["frame_status"][1], "interpolated")  # 0.24 s gap fills.
            _, result, _ = video_trim.pack_trim_export_record(0, args, root, lambda *a: None)
            with np.load(result, allow_pickle=False) as output, ZipFile(result) as zipped:
                image = output["data_000000"]
                self.assertEqual(image.dtype, values.dtype)
                self.assertFalse(image[black].any())
                self.assertEqual(image[~black].tobytes(), values[indices[~black]].tobytes())
                self.assertEqual(zipped.getinfo("data_000000.npy").compress_type, ZIP_BZIP2)

    def test_unbracketed_and_entirely_missing_targets_are_black(self):
        for all_black in (False, True):
            with self.subTest(all_black=all_black), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                features = root / "worm_features.hdf5"
                table = np.zeros(101, dtype=[("timestamp", "f8")])
                table["timestamp"] = np.arange(101)
                with h5py.File(features, "w") as source:
                    source.create_dataset("features_timeseries", data=table).attrs["fps"] = 100.
                raw = np.array([10, 90]) if all_black else np.arange(10, 91)
                values = np.ones((len(raw), 2, 3), dtype="u1")
                path = fixtures.mask_export(root, "a", "record_1", values, ZIP_BZIP2, raw, raw / 100)
                args = (path, ("a", "record_1"), features, 26, 26, None, "nearest", None)
                _, result, error = video_trim.pack_trim_export_record(0, args, root, lambda *a: None)
                with np.load(result, allow_pickle=False) as output:
                    black = output["source_index_000000"] < 0
                    self.assertTrue(black[:3].all())
                    self.assertTrue(black[-3:].all())
                    self.assertFalse(output["data_000000"][black].any())
                    self.assertTrue(np.isnan(output["time_s_000000"][black]).all())
                    np.testing.assert_array_equal(output["frame_number_000000"][black], -1)
                    np.testing.assert_allclose(output["target_time_s_000000"], np.arange(26) * .04, atol=1e-12)
                    if all_black:
                        self.assertTrue(black.all())
                        self.assertEqual(error, 0.)
                    else:
                        self.assertFalse(black[3:-3].any())

    def test_gap_pixels_match_across_staged_records_archives_hdf5_and_workers(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw_root = self.fixtures(root)
            for label, record in (("c", "record_2"), ("d", "record_3")):
                raw = np.r_[np.arange(7), np.arange(17, 20)]
                values = np.repeat((raw + 1).astype("u1"), 6).reshape(-1, 2, 3)
                fixtures.mask_export(root, label, record, values, ZIP_BZIP2, raw, 10 + raw / 30)
                with h5py.File(raw_root / label / record / "worm.hdf5", "w") as source:
                    source.create_dataset("mask", data=values).attrs["fps"] = 30.
                    source.create_dataset("timestamp/raw", data=raw)
                    source.create_dataset("timestamp/time", data=10 + raw / 30)
            dataset.build_datasets(root, "video")
            with patch.object(trim, "MIN_SAMPLES", 3):
                for kind in ("records", "archive", "hdf5"):
                    for workers in (1, 2):
                        with self.subTest(kind=kind, workers=workers):
                            path, = trim.trim_datasets(root, root / f"{kind}-{workers}", dataset="video",
                                                       video_source=kind, video_dir=raw_root,
                                                       staging_dir=root / "ssd", workers=workers)
                            with np.load(path, allow_pickle=False) as output, ZipFile(path) as zipped:
                                self.assertEqual(output["labels"].tolist(), ["c", "d"])
                                self.assertEqual(output["record_ids"].tolist(), ["record_2", "record_3"])
                                for i in range(2):
                                    np.testing.assert_array_equal(output[f"source_index_{i:06d}"], [6, -1, -1, -1, -1])
                                    np.testing.assert_array_equal(output[f"data_{i:06d}"][:, 0, 0], [7, 0, 0, 0, 0])
                                    self.assertEqual(output[f"frame_status_{i:06d}"].tolist(), ["real"] + ["zeroed"] * 4)
                                    self.assertEqual(zipped.getinfo(f"data_{i:06d}.npy").compress_type, ZIP_BZIP2)
                                np.testing.assert_allclose(output["time_s"], np.arange(5) * .04)
                            self.assertEqual(list((root / "ssd").iterdir()), [])

    def test_records_only_reselects_frames_without_combined_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            (root / "openworm_video_dataset.npz").unlink()
            with patch.object(trim, "MIN_SAMPLES", 3), patch.object(
                    video_trim, "copy_images", wraps=video_trim.copy_images) as select:
                path, = trim.trim_datasets(root, root / "final", dataset="video", video_source="records",
                                           staging_dir=root / "ssd")
            self.assertEqual(select.call_count, 2)
            with np.load(path, allow_pickle=False) as output:
                self.assertEqual(output["labels"].tolist(), ["c", "d"])
                for index in range(2):
                    np.testing.assert_array_equal(output[f"source_index_{index:06d}"], [6, 7, 8, 10, 11])
                    np.testing.assert_array_equal(output[f"data_{index:06d}"][:, 0, 0], [6, 7, 8, 10, 11])
            self.assertFalse((root / "openworm_video_dataset.npz").exists())
            self.assertEqual(list((root / "ssd").iterdir()), [])

    def test_hdf5_cli_creates_only_trimmed_output_without_any_video_npzs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = self.fixtures(root)
            (root / "openworm_video_dataset.npz").unlink()
            shutil.rmtree(root / "video")
            output = root / "external-final"
            with patch.object(trim, "MIN_SAMPLES", 3), patch.object(sys, "argv", [
                    "trim", "--dataset", "video", "--video-source", "hdf5", "--data-dir", str(root),
                    "--video-dir", str(raw), "--output-dir", str(output), "--staging-dir", str(root / "ssd"), "--workers", "2"]):
                trim.main()
            path = output / "openworm_video_trimmed_890_dataset.npz"
            self.assertEqual([p.name for p in output.iterdir()], [path.name])
            self.assertFalse((root / "openworm_video_dataset.npz").exists())
            self.assertFalse((root / "video").exists())
            with np.load(path, allow_pickle=False) as result, ZipFile(path) as zipped:
                self.assertEqual(result["labels"].tolist(), ["c", "d"])
                self.assertEqual(result["data_000000"].shape, (5, 2, 3))
                self.assertEqual(zipped.getinfo("data_000000.npy").compress_type, ZIP_BZIP2)
            self.assertEqual(list((root / "ssd").iterdir()), [])
            self.assertEqual(len(list(raw.glob("*/record_*/*.hdf5"))), 4)

    def test_direct_hdf5_keeps_non_native_pixel_dtype_and_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = self.fixtures(root)
            values = np.arange(120, dtype=">u2").reshape(20, 2, 3)
            with h5py.File(raw / "c/record_2/worm.hdf5", "a") as source:
                del source["mask"]
                source.create_dataset("mask", data=values).attrs["fps"] = 30.
            with patch.object(trim, "MIN_SAMPLES", 3):
                path, = trim.trim_datasets(root, root / "final", dataset="video", video_source="hdf5",
                                           video_dir=raw, staging_dir=root / "ssd", workers=2)
            with np.load(path, allow_pickle=False) as result:
                expected = values[[6, 7, 8, 10, 11]]
                self.assertEqual(result["data_000000"].dtype, values.dtype)
                np.testing.assert_array_equal(result["data_000000"], expected)
                self.assertEqual(result["data_000000"].tobytes(), expected.tobytes())

    def test_dataset_trim_video_entrypoint_does_not_build_untrimmed_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = self.fixtures(root)
            (root / "openworm_video_dataset.npz").unlink()
            with patch.object(trim, "MIN_SAMPLES", 3):
                path, = dataset.build_datasets(root, "video", workers=2, trim_video=True,
                                               video_source="hdf5", video_dir=raw,
                                               output_dir=root / "final", staging_dir=root / "ssd")
            self.assertEqual(path.name, "openworm_video_trimmed_890_dataset.npz")
            self.assertFalse((root / "openworm_video_dataset.npz").exists())
            self.assertEqual(list((root / "ssd").iterdir()), [])

    def test_dataset_assembly_staging_preserves_compression_and_source_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixtures(root)
            expected = root / "expected.npz"
            shutil.copyfile(root / "openworm_video_dataset.npz", expected)
            actual, = dataset.build_datasets(root, "video", workers=2, output_dir=root / "combined", staging_dir=root / "ssd")
            self.compare(expected, actual)
            with ZipFile(expected) as left, ZipFile(actual) as right:
                for key in ("data_000000.npy", "data_000001.npy", "data_000002.npy", "data_000003.npy"):
                    self.assertEqual(left.getinfo(key).compress_type, right.getinfo(key).compress_type)
            self.assertEqual(list((root / "ssd").iterdir()), [])

    def test_staged_worker_failure_preserves_existing_final_and_cleans_ssd(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixtures(root)
            with patch.object(trim, "MIN_SAMPLES", 3):
                path, = trim.trim_datasets(root, root / "final", dataset="video", video_source="records", staging_dir=root / "ssd")
                before = path.read_bytes()
                source = root / "video/c/record_2.npz"
                with np.load(source, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                arrays["time_s"][8] = arrays["time_s"][7]
                np.savez_compressed(source, **arrays)
                with self.assertRaises(ValueError):
                    trim.trim_datasets(root, root / "final", workers=2, dataset="video", video_source="records", staging_dir=root / "ssd")
                self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list((root / "ssd").iterdir()), [])
            self.assertEqual(list((root / "final").glob("*.part")), [])
            self.assertTrue(source.exists())

    def test_archive_staging_extracts_records_instead_of_copying_whole_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixtures(root)
            original = video_trim.append_entry
            copied = []
            def append(output, source, key, *args, **kwargs):
                copied.append(key)
                return original(output, source, key, *args, **kwargs)
            with patch.object(trim, "MIN_SAMPLES", 3), patch.object(video_trim, "append_entry", side_effect=append):
                trim.trim_datasets(root, root / "final", workers=2, dataset="video", video_source="archive", staging_dir=root / "ssd")
            self.assertEqual(copied, ["data_000002.npy", "data_000003.npy"])
            self.assertEqual(list((root / "ssd").iterdir()), [])

    def test_stager_keeps_processing_completions_while_next_copy_is_waiting(self):
        with tempfile.TemporaryDirectory() as d, tqdm(total=2, disable=True) as overall:
            root = Path(d); parts = root / "parts"; parts.mkdir()
            first = []
            def prepare(record, arguments, directory, stop, callback):
                if record[1] == "record_1":
                    deadline = time.monotonic() + 15
                    while first[0].exists():
                        if time.monotonic() > deadline:
                            raise RuntimeError("Completed staged input was not released while waiting for another copy")
                        time.sleep(.01)
                path = Path(directory) / "input.npz"
                path.write_bytes(b"source")
                if not first: first.append(path)
                return path,
            jobs = [(('a', f'record_{i}'), ()) for i in range(2)]
            with StagedJobs(jobs, root / "ssd", 2, overall, prepare) as feed:
                completed = packed_records(tiny_pack, feed, 2, parts, overall, feed.release, feed.poll)
                try:
                    self.assertEqual({i for i, _, _ in completed}, {0, 1})
                finally:
                    completed.close()
            self.assertEqual(list((root / "ssd").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
