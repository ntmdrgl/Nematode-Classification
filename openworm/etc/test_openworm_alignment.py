"""Reference record selection and physical acquisition-time alignment."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import openworm.utils.openworm_alignment as alignment
import openworm.openworm_trim as trim
from openworm.openworm_matlab_features import FEATURES, preprocess_extracted


class AlignmentTests(unittest.TestCase):
    def fixtures(self, root, offset=2):
        for dataset, width in trim.DATASETS:
            lengths = [2, 3, 5, 7] if dataset != "matlab_features_new" else [12] * 4
            arrays = {f"data_{i:06d}": np.repeat(np.arange(n, dtype=float)[:, None], width, axis=1)
                      for i, n in enumerate(lengths)}
            np.savez_compressed(root / f"openworm_{dataset}_dataset.npz",
                                labels=["a", "b", "c", "d"], record_ids=[f"record_{i}" for i in range(4)],
                                record_keys=list(arrays), feature_names=FEATURES if dataset == "matlab_features_new" else [f"f_{j}" for j in range(width)], **arrays)
        for i, (strain, count) in enumerate(zip("abcd", [2, 3, 5, 7])):
            directory = root / "openworm" / strain / f"record_{i}"
            directory.mkdir(parents=True)
            with h5py.File(directory / "worm_features.hdf5", "w") as f:
                table = np.zeros(count, dtype=[("timestamp", "f8")])
                table["timestamp"] = offset + np.arange(count)
                ds = f.create_dataset("features_timeseries", data=table)
                ds.attrs["fps"] = 25.
            directory = root / "videos" / strain / f"record_{i}"
            directory.mkdir(parents=True)
            with h5py.File(directory / "worm.hdf5", "w") as f:
                f.create_dataset("mask", shape=(12, 1, 1), dtype="u1")
                f.create_dataset("timestamp/raw", data=np.arange(12))
                f.create_dataset("timestamp/time", data=10 + np.arange(12) * .04)

    def test_same_reference_records_and_shifted_rows_serial_and_parallel(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            with patch.object(trim, "MIN_SAMPLES", 3):
                for workers in (1, 2):
                    output, = trim.trim_datasets(root, workers=workers, dataset="matlab_features_new", video_dir=root / "videos")
                    with np.load(output, allow_pickle=False) as archive:
                        self.assertEqual(archive["labels"].tolist(), ["c", "d"])
                        self.assertEqual(archive["record_ids"].tolist(), ["record_2", "record_3"])
                        for key in archive["record_keys"]:
                            np.testing.assert_array_equal(archive[key][:, 0], [2, 3, 4, 5, 6])
                            self.assertEqual(archive[key].shape, (5, 16))
                        np.testing.assert_allclose(archive["time_s"], np.arange(5) * .04)

    def test_fractional_offset_requires_original_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root, offset=2.5)
            features = root / "openworm/c/record_2/worm_features.hdf5"
            video = root / "videos/c/record_2/worm.hdf5"
            indices, times = alignment.alignment_plan(features, video, 5, 12, 5)
            self.assertIsNone(indices)
            np.testing.assert_allclose(times, 10.1 + np.arange(5) * .04)
            with patch.object(alignment, "rebuild_record", return_value=np.ones((5, 16))) as rebuild:
                values = trim.aligned_record(root / "openworm_matlab_features_new_dataset.npz",
                                             "data_000002", 12, 5, 5, features, video, None)
            np.testing.assert_array_equal(values, np.ones((5, 16)))
            np.testing.assert_allclose(rebuild.call_args.args[1], times)

    def test_check_only_and_missing_timestamps_never_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            with patch.object(trim, "MIN_SAMPLES", 3):
                self.assertEqual(trim.trim_datasets(root, dataset="matlab_features_new", video_dir=root / "videos", check_alignment=True), [])
                video = root / "videos/c/record_2/worm.hdf5"
                with h5py.File(video, "a") as f:
                    del f["timestamp/time"]
                with self.assertRaises(KeyError):
                    trim.trim_datasets(root, dataset="matlab_features_new", video_dir=root / "videos")
            self.assertEqual(list(root.glob("*trimmed*")), [])
            self.assertEqual(list(root.glob("*.part")), [])

    def test_timestamp_length_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            with self.assertRaisesRegex(ValueError, "Reference timestamps disagree"):
                alignment.alignment_plan(root / "openworm/c/record_2/worm_features.hdf5",
                                         root / "videos/c/record_2/worm.hdf5", 6, 12, 5)

    def test_actual_reference_clock_is_used_when_adjacent_video_exists(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            features = root / "openworm/c/record_2/worm_features.hdf5"
            video = root / "videos/c/record_2/worm.hdf5"
            # An irregular acquisition clock differs from the nominal FPS.
            times = 10 + np.arange(12) * .04
            times[3] += .003
            with h5py.File(video, "a") as f:
                f["timestamp/time"][:] = times
            import shutil
            shutil.copyfile(video, features.with_name("worm.hdf5"))
            indices, query = alignment.alignment_plan(features, video, 5, 12, 5)
            np.testing.assert_array_equal(indices, np.arange(2, 7))
            np.testing.assert_allclose(query, times[2] + np.arange(5) * .04)

    def test_conflicting_video_times_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            video = root / "videos/c/record_2/worm.hdf5"
            with h5py.File(video, "a") as f:
                f["timestamp/time"][3] = f["timestamp/time"][2]
            with self.assertRaisesRegex(ValueError, "Conflicting/backwards"):
                alignment.alignment_plan(root / "openworm/c/record_2/worm_features.hdf5", video, 5, 12, 5)

    def test_resamples_raw_observations_on_requested_times(self):
        times = 10 + np.arange(20) * .04
        values = np.repeat(np.arange(20, dtype=float)[:, None], 16, axis=1)
        values[:, 15] = np.where(np.arange(20) % 2 == 0, 89., -89.)
        observed = np.ones_like(values, dtype=bool)
        observed[4:14, 0] = False
        raw = dict(feature_names=FEATURES, feature_values=values, observed_mask=observed,
                   time_s=times, frame_number=np.arange(20))
        requested = 10.1 + np.arange(12) * .04
        grid, actual, status = preprocess_extracted(raw, sample_times=requested)
        np.testing.assert_allclose(grid, requested - 10)
        np.testing.assert_allclose(actual[:, 1], 2.5 + np.arange(12))
        self.assertTrue(np.all(np.abs(actual[:, 15]) > 88))
        self.assertEqual(actual[5, 0], 0)
        self.assertEqual(status[5, 0], "zeroed")


if __name__ == "__main__":
    unittest.main()
