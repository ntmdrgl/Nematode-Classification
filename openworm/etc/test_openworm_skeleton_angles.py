"""Regression checks for spatial/time resampling, gap handling and angles."""
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openworm_skeleton_angles as script


def curve(amplitude=1):
    y = np.linspace(0, 10, 49)
    return np.column_stack((amplitude * np.sin(y / 3), y))


class SkeletonTests(unittest.TestCase):
    def test_equal_arc_length(self):
        uneven = np.linspace(0, 1, 49)**2
        result = script.spatial_resample(np.array([np.column_stack((2*uneven, 3*uneven))]))
        distances = np.linalg.norm(np.diff(result[0], axis=0), axis=1)
        np.testing.assert_allclose(distances, np.sqrt(13)/48)

    def test_short_interpolation_and_long_zero_gap(self):
        times = np.arange(16) * .04
        points = np.array([curve(1 + t) for t in times])
        points[2:4] = np.nan      # endpoint separation .12 s: fill
        points[6:12] = np.nan    # endpoint separation .28 s: zero
        grid, angles, valid, status = script.preprocess(times, points, 1, 1, 0)
        np.testing.assert_allclose(np.diff(grid), .04)
        self.assertTrue(valid[2:4].all())
        self.assertTrue(np.all(status[2:4] == 'interpolated'))
        self.assertTrue(np.all(status[6:12] == 'zeroed'))
        self.assertEqual(status[0], 'real')
        self.assertFalse(valid[6:12].any())
        np.testing.assert_array_equal(angles[6:12], 0)
        resampled = script.spatial_resample(points)
        middle = resampled[1] * (2/3) + resampled[4] * (1/3)
        delta = np.diff(middle, axis=0)
        expected = np.arctan(delta[:, 0] / delta[:, 1])
        np.testing.assert_allclose(angles[2], expected - expected.mean(), atol=1e-12)
        np.testing.assert_allclose(angles[valid].mean(axis=1), 0, atol=1e-12)

    def test_exact_threshold_and_unbracketed_ends(self):
        _, angles, valid, _ = script.preprocess(np.array([0., .25]), np.array([curve(), curve(2)]), 1, 1, 0)
        self.assertFalse(valid[1:].any())
        np.testing.assert_array_equal(angles[1:], 0)
        times = np.arange(10)*.04
        points = np.full((10, 49, 2), np.nan)
        points[4] = curve()
        _, _, valid, _ = script.preprocess(times, points)
        self.assertEqual(np.flatnonzero(valid).tolist(), [4])

    def test_smoothing_cannot_cross_long_gaps(self):
        times = np.arange(30)*.04
        points = np.array([curve(1 + .1*np.sin(i)) for i in range(30)])
        points[10:20] = np.nan
        _, angles, valid, _ = script.preprocess(times, points)
        changed = points.copy()
        changed[20:, :, 0] *= 20
        _, changed_angles, _, _ = script.preprocess(times, changed)
        np.testing.assert_allclose(angles[:10], changed_angles[:10])
        np.testing.assert_array_equal(angles[10:20], 0)
        self.assertFalse(valid[10:20].any())
        self.assertTrue(np.isfinite(angles).all())

    def test_missing_and_degenerate_skeletons(self):
        for points in (np.zeros((5, 49, 2)), np.full((5, 49, 2), np.nan)):
            _, angles, valid, _ = script.preprocess(np.arange(5)*.04, points)
            self.assertFalse(valid.any())
            np.testing.assert_array_equal(angles, 0)

    def test_real_video_timestamps_take_precedence_over_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worm_features.hdf5"
            with h5py.File(path, "w") as f:
                f["coordinates/skeletons"] = np.array([curve()]*4)
                table = f.create_dataset("features_timeseries", data=np.array(
                    [(i, 1) for i in range(4)], dtype=[("timestamp", "f8"), ("worm_index", "f8")]))
                table.attrs["fps"] = 100
            with h5py.File(path.with_name("worm.hdf5"), "w") as f:
                f["timestamp/raw"] = [0, 1, 3]
                f["timestamp/time"] = [0., .04, .13]
            times, _ = script.read_record(path)
            np.testing.assert_allclose(times, [0., .04, .085, .13])


if __name__ == "__main__":
    unittest.main()
