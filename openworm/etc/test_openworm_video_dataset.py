"""Compressed 3D mask datasets and acquisition-time alignment to skeletons."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZipFile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import openworm.openworm_dataset as dataset
import openworm.openworm_trim as trim
import openworm.utils.openworm_video_dataset as video_dataset
import test_openworm_dataset as feature_fixtures


def mask_export(root, strain, record_id, images, method=ZIP_BZIP2, raw=None, times=None):
    path = root / "video" / strain / (record_id + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    arrays = dict(mask=images, source_dataset="mask", strain=strain, record_id=record_id,
                  frame_number=np.arange(n) if raw is None else raw,
                  time_s=10 + np.arange(n) / 30 if times is None else times,
                  fps=30., source_file="original/worm.hdf5")
    with ZipFile(path, "w", compression=method, compresslevel=9 if method != ZIP_LZMA else None) as archive:
        for key, value in arrays.items():
            dataset.write_array(archive, key, value)
    return path


class VideoDatasetTests(unittest.TestCase):
    def video_fixtures(self, root):
        expected = {}
        for i, (strain, count, method) in enumerate((("a", 6, ZIP_BZIP2), ("b", 9, ZIP_LZMA), ("c", 11, ZIP_DEFLATED))):
            values = np.arange(count * 6, dtype=">u2").reshape(count, 2, 3)
            mask_export(root, strain, "record_1", values, method)
            expected[strain] = (values, method)
        return expected

    def test_ragged_3d_pixels_dtype_ids_and_codecs_preserved_serial_and_parallel(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            expected = self.video_fixtures(root)
            before = None
            for workers in (1, 2):
                path, = dataset.build_datasets(root, "video", workers)
                with np.load(path, allow_pickle=False) as archive, ZipFile(path) as zipped:
                    self.assertEqual(archive["labels"].tolist(), ["a", "b", "c"])
                    self.assertEqual(archive["record_ids"].tolist(), ["record_1"] * 3)
                    self.assertEqual(archive["array_axes"].tolist(), ["frame", "height", "width"])
                    for i, strain in enumerate(expected):
                        key = str(archive["record_keys"][i])
                        values, method = expected[strain]
                        np.testing.assert_array_equal(archive[key], values)
                        self.assertEqual(archive[key].dtype, values.dtype)
                        self.assertEqual(archive[key].ndim, 3)
                        self.assertEqual(zipped.getinfo(key + ".npy").compress_type, method)
                        np.testing.assert_array_equal(archive[f"frame_number_{i:06d}"], np.arange(len(values)))
                    current = {key: archive[key].copy() for key in archive.files}
                if before is not None:
                    self.assertEqual(before.keys(), current.keys())
                    for key in before:
                        np.testing.assert_array_equal(before[key], current[key])
                before = current

    def test_streamed_selection_repeats_source_frames_without_changing_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            values = np.arange(18, dtype="u1").reshape(3, 2, 3)
            source = mask_export(root, "a", "record_1", values)
            output = root / "selected.npz"
            indices = np.array([0, 0, 1, 2, 2], dtype="u8")
            with ZipFile(source) as incoming, ZipFile(output, "w", compression=ZIP_BZIP2) as outgoing:
                with patch.object(video_dataset, "BUFFER_BYTES", 7):
                    video_dataset.copy_images(incoming, "mask", outgoing, "data_000000", indices)
            with np.load(output, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["data_000000"], values[indices])

    def test_all_mode_includes_video_with_or_without_new_matlab(self):
        for new in (False, True):
            with self.subTest(new=new), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                feature_fixtures.DatasetTests().fixtures(root)
                if not new:
                    shutil.rmtree(root / "matlab_features_new")
                values = np.arange(42, dtype="u1").reshape(7, 2, 3)
                mask_export(root, "cb5", "record_123", values)
                paths = dataset.build_datasets(root)
                self.assertEqual(len(paths), 6 if new else 5)
                self.assertEqual(paths[-1].name, "openworm_video_dataset.npz")
                with np.load(paths[-1], allow_pickle=False) as archive:
                    self.assertEqual(archive["labels"].tolist(), ["cb5"])
                    self.assertEqual(archive["record_ids"].tolist(), ["record_123"])
                    np.testing.assert_array_equal(archive["data_000000"], values)

    def test_bad_labels_or_old_full_data_preserve_existing_output(self):
        for invalid in ("labels", "full_data"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.video_fixtures(root)
                output, = dataset.build_datasets(root, "video")
                before = output.read_bytes()
                path = root / "video/a/record_1.npz"
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                if invalid == "labels":
                    arrays["strain"] = "wrong label"
                else:
                    arrays["full_data"] = arrays.pop("mask")
                np.savez_compressed(path, **arrays)
                with self.assertRaises(ValueError):
                    dataset.build_datasets(root, "video")
                self.assertEqual(output.read_bytes(), before)
                self.assertEqual(list(root.glob("*.part")), [])

    def test_truncated_payload_is_rejected_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.video_fixtures(root)
            path = root / "video/a/record_1.npz"
            with ZipFile(path) as archive:
                entries = {name: archive.read(name) for name in archive.namelist()}
            entries["mask.npy"] = entries["mask.npy"][:-5]
            with ZipFile(path, "w", compression=ZIP_BZIP2) as archive:
                for name, value in entries.items():
                    archive.writestr(name, value)
            with self.assertRaisesRegex(ValueError, "Truncated"):
                dataset.build_datasets(root, "video")
            self.assertFalse((root / "openworm_video_dataset.npz").exists())
            self.assertEqual(list(root.glob("*.part")), [])


class VideoTrimTests(unittest.TestCase):
    def fixtures(self, root, duplicate=False):
        for name, width in trim.ALL_DATASETS:
            arrays = {f"data_{i:06d}": np.zeros((n, width)) for i, n in enumerate((2, 3, 5, 7))}
            np.savez_compressed(root / f"openworm_{name}_dataset.npz", labels=list("abcd"),
                                record_ids=[f"record_{i}" for i in range(4)], record_keys=list(arrays),
                                feature_names=[f"f_{i}" for i in range(width)], **arrays)
        expected = {}
        for i, n in enumerate((2, 3, 5, 7)):
            strain, record = "abcd"[i], f"record_{i}"
            raw = np.arange(20)
            if duplicate:
                raw = np.r_[raw[:7], 6, raw[7:]]
            values = np.repeat(np.arange(len(raw), dtype="u1"), 6).reshape(len(raw), 2, 3)
            method = (ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA, ZIP_BZIP2)[i]
            mask_export(root, strain, record, values, method, raw, 10 + raw / 30)
            folder = root / "openworm" / strain / record
            folder.mkdir(parents=True)
            frames = np.arange(6, 6 + int(np.ceil((n - 1) * 30 / 25)) + 1)
            table = np.zeros(len(frames), dtype=[("timestamp", "f8")])
            table["timestamp"] = frames
            with h5py.File(folder / "worm_features.hdf5", "w") as source:
                ds = source.create_dataset("features_timeseries", data=table)
                ds.attrs["fps"] = 30.
            expected[strain] = (values, method, raw)
        dataset.build_datasets(root, "video")
        return expected

    def test_nearest_alignment_same_reference_records_shape_pixels_and_codecs(self):
        for workers in (1, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                expected = self.fixtures(root)
                with patch.object(trim, "MIN_SAMPLES", 3):
                    path, = trim.trim_datasets(root, dataset="video", workers=workers)
                with np.load(path, allow_pickle=False) as archive, ZipFile(path) as zipped:
                    self.assertEqual(archive["labels"].tolist(), ["c", "d"])
                    self.assertEqual(archive["record_ids"].tolist(), ["record_2", "record_3"])
                    for i, strain in enumerate("cd"):
                        indices = np.array([6, 7, 8, 10, 11])
                        values, method, raw = expected[strain]
                        np.testing.assert_array_equal(archive[f"data_{i:06d}"], values[indices])
                        self.assertEqual(archive[f"data_{i:06d}"].shape, (5, 2, 3))
                        self.assertEqual(archive[f"data_{i:06d}"].dtype, np.dtype("u1"))
                        self.assertEqual(zipped.getinfo(f"data_{i:06d}.npy").compress_type, method)
                        np.testing.assert_array_equal(archive[f"source_index_{i:06d}"], indices)
                        np.testing.assert_array_equal(archive[f"frame_number_{i:06d}"], raw[indices])
                        np.testing.assert_allclose(archive[f"time_s_{i:06d}"], 10 + raw[indices] / 30)
                        np.testing.assert_allclose(archive[f"target_time_s_{i:06d}"], 10.2 + np.arange(5) * .04)
                        self.assertLessEqual(np.max(abs(archive[f"alignment_error_s_{i:06d}"])), .02)
                    np.testing.assert_allclose(archive["time_s"], np.arange(5) * .04)

    def test_check_only_and_strict_failure_do_not_change_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            with patch.object(trim, "MIN_SAMPLES", 3):
                self.assertEqual(trim.trim_datasets(root, dataset="video", check_alignment=True), [])
                path, = trim.trim_datasets(root, dataset="video")
                before = path.read_bytes()
                for options in (dict(video_alignment="exact"), dict(max_frame_error=.005)):
                    with self.subTest(options=options), self.assertRaisesRegex(ValueError, "exceeds"):
                        trim.trim_datasets(root, dataset="video", **options)
                    self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(root.glob("*.part")), [])

    def test_duplicate_acquisition_rows_map_back_to_original_image_indices(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            expected = self.fixtures(root, duplicate=True)
            with patch.object(trim, "MIN_SAMPLES", 3):
                path, = trim.trim_datasets(root, dataset="video")
            with np.load(path, allow_pickle=False) as archive:
                indices = np.array([6, 8, 9, 11, 12])
                np.testing.assert_array_equal(archive["source_index_000000"], indices)
                np.testing.assert_array_equal(archive["data_000000"], expected["c"][0][indices])
                np.testing.assert_array_equal(archive["frame_number_000000"], [6, 7, 8, 10, 11])

    def test_missing_embedded_timestamps_are_recovered_from_explicit_video_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            expected = self.fixtures(root)
            source = root / "video/c/record_2.npz"
            with np.load(source, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files if key not in ("frame_number", "time_s")}
            np.savez_compressed(source, **arrays)
            folder = root / "raw/c/record_2"
            folder.mkdir(parents=True)
            with h5py.File(folder / "worm.hdf5", "w") as raw:
                raw.create_dataset("mask", data=expected["c"][0])
                raw.create_dataset("timestamp/raw", data=np.arange(20))
                raw.create_dataset("timestamp/time", data=10 + np.arange(20) / 30)
            dataset.build_datasets(root, "video")
            with patch.object(trim, "MIN_SAMPLES", 3):
                output, = trim.trim_datasets(root, dataset="video", video_dir=root / "raw")
            with np.load(output, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["source_index_000000"], [6, 7, 8, 10, 11])
                np.testing.assert_array_equal(archive["data_000000"], expected["c"][0][[6, 7, 8, 10, 11]])

    def test_exact_mode_succeeds_for_coincident_reference_frames(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            for name, width in trim.ALL_DATASETS:
                path = root / f"openworm_{name}_dataset.npz"
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                arrays["data_000000"] = arrays["data_000000"][:1]
                np.savez_compressed(path, **arrays)
            path = root / "openworm/a/record_0/worm_features.hdf5"
            with h5py.File(path, "a") as reference:
                table = reference["features_timeseries"][:1]
                del reference["features_timeseries"]
                reference.create_dataset("features_timeseries", data=table).attrs["fps"] = 30.
            with patch.object(trim, "MIN_SAMPLES", 0):
                path, = trim.trim_datasets(root, dataset="video", video_alignment="exact")
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["labels"].tolist(), list("abcd"))
                for i in range(4):
                    self.assertEqual(archive[f"data_{i:06d}"].shape, (1, 2, 3))
                    np.testing.assert_allclose(archive[f"alignment_error_s_{i:06d}"], 0., atol=1e-8)

    def test_conflicting_timestamps_and_missing_records_are_rejected(self):
        for failure in ("conflict", "missing"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixtures(root)
                source = root / "video/c/record_2.npz"
                if failure == "missing":
                    source.unlink()
                else:
                    with np.load(source, allow_pickle=False) as archive:
                        arrays = {key: archive[key] for key in archive.files}
                    arrays["time_s"][8] = arrays["time_s"][7]
                    np.savez_compressed(source, **arrays)
                dataset.build_datasets(root, "video")
                with patch.object(trim, "MIN_SAMPLES", 3), self.assertRaises(ValueError):
                    trim.trim_datasets(root, dataset="video")
                self.assertFalse((root / "openworm_video_trimmed_890_dataset.npz").exists())
                self.assertEqual(list(root.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
