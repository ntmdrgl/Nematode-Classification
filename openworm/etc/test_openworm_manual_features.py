"""Checks for feature selection, gaps, temporal alignment and orientation wrap."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openworm_manual_features as script


class FeatureTests(unittest.TestCase):
    def test_excludes_ids_timestamps_categories_and_flags(self):
        table = np.zeros(1, dtype=[("worm_index", "f4"), ("timestamp", "f4"),
            ("skeleton_id", "f4"), ("motion_modes", "f4"), ("has_skeleton", "?"),
            ("length", "f4"), ("head_speed", "f4"), ("eigen_projection_1", "f4")])
        self.assertEqual(script.feature_names(table), ["length", "head_speed", "eigen_projection_1"])

    def test_per_feature_gaps_and_zero_preservation(self):
        times = np.arange(16)*.04
        values = np.arange(16, dtype=float)
        values[2:4] = np.nan
        values[6:12] = np.nan
        grid, result, status = script.preprocess_feature(times, values, np.ones(16, bool), time_window=1, polyorder=0)
        np.testing.assert_allclose(grid, times)
        np.testing.assert_allclose(result[2:4], [2, 3])
        self.assertTrue(np.all(status[2:4] == "interpolated"))
        self.assertTrue(np.all(status[6:12] == "zeroed"))
        np.testing.assert_array_equal(result[6:12], 0)
        self.assertEqual(status[0], "real")  # Measured zero is valid.

    def test_unusable_skeleton_masks_even_finite_features(self):
        times = np.arange(20)*.04
        valid = np.ones(20, bool)
        valid[5:15] = False
        _, result, status = script.preprocess_feature(times, np.ones(20), valid)
        np.testing.assert_array_equal(result[5:15], 0)
        self.assertTrue(np.all(status[5:15] == "zeroed"))

    def test_orientation_interpolates_across_180_not_through_zero(self):
        _, result, status = script.preprocess_feature(np.array([0., .04, .08]),
            np.array([179., np.nan, -179.]), np.ones(3, bool), angular=True, time_window=1, polyorder=0)
        self.assertAlmostEqual(abs(result[1]), 180)
        self.assertEqual(status[1], "interpolated")

    def test_missing_feature_and_gap_boundary(self):
        _, result, status = script.preprocess_feature(np.arange(4)*.04, np.full(4, np.nan), np.ones(4, bool))
        np.testing.assert_array_equal(result, 0)
        self.assertTrue(np.all(status == "zeroed"))
        _, result, status = script.preprocess_feature(np.array([0., .25]), np.array([1., 2.]),
                                                       np.ones(2, bool), time_window=1, polyorder=0)
        self.assertTrue(np.all(status[1:] == "zeroed"))


if __name__ == "__main__":
    unittest.main()
