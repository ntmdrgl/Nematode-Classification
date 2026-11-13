"""Evaluation tests for feature statuses, weighted totals and stale sources."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openworm_evaluate as script


class EvaluationTests(unittest.TestCase):
    def fixtures(self, root):
        manual = [f"feature_{i}" for i in range(47)]
        eigen = [f"eigen_projection_{i}" for i in range(1, 7)]
        angles = [f"angle_{i:02d}" for i in range(1, 49)]
        for strain, frames in (("a", 2), ("b", 1)):
            for folder, names in (("skeleton_angles", angles), ("manual_features", manual + eigen)):
                path = root / folder / strain / "record_1.csv"
                path.parent.mkdir(parents=True)
                with path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    flags = ["skeleton_status"] if folder == "skeleton_angles" else [name+"_status" for name in names]
                    writer.writerow(["time_s", *names, *flags])
                    for i in range(frames):
                        if folder == "skeleton_angles":
                            states = ["real" if strain == "a" else "zeroed"]
                        else:
                            states = (["interpolated", *(["real"]*46), *(["interpolated"]*6)]
                                      if strain == "a" else ["zeroed"]*53)
                        writer.writerow([i*.04, *([0]*len(names)), *states])
        for (_, filename, _, count), names in zip(script.DATASETS, (angles, manual, eigen)):
            np.savez(root / filename, labels=np.array(["a", "b"]), record_ids=np.array(["record_1"]*2),
                     record_keys=np.array(["data_0", "data_1"]), feature_names=np.array(names),
                     data_0=np.zeros((2, count)), data_1=np.zeros((1, count)))
        names = [f"matlab_{i}" for i in range(16)]
        path = root / "matlab_features/a/record_1.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", *names, *[name+"_status" for name in names]])
            for i in range(2):
                writer.writerow([i*.04, *([0]*16), "zeroed", *(["real"]*15)])
        np.savez(root / "openworm_matlab_features_dataset.npz", labels=np.array(["a"]),
                 record_ids=np.array(["record_1"]), record_keys=np.array(["data_0"]),
                 feature_names=np.array(names), data_0=np.zeros((2,16)))

    def test_per_strain_and_weighted_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            rows = script.evaluate(root)
            self.assertEqual(len(rows), 11)
            self.assertTrue(all(row["scope"] == "total" for row in rows[:4]))
            totals = {row["dataset"]: row for row in rows if row["scope"] == "total"}
            angles = totals["skeleton_angles"]
            self.assertEqual(angles["real_count"], 2*48)
            self.assertEqual(angles["zeroed_count"], 48)
            self.assertAlmostEqual(float(angles["real_percent"]), 100*2/3, places=5)
            manual = totals["manual_features"]
            self.assertEqual(manual["real_count"], 2*46)
            self.assertEqual(manual["interpolated_count"], 2)
            self.assertEqual(manual["zeroed_count"], 47)
            self.assertEqual(totals["eigenworms"]["interpolated_count"], 12)
            self.assertEqual(totals["eigenworms"]["zeroed_count"], 6)
            self.assertEqual(totals["matlab_features"]["records"], 1)
            self.assertEqual(totals["matlab_features"]["real_count"], 30)
            self.assertEqual(totals["matlab_features"]["zeroed_count"], 2)
            for row in rows:
                self.assertAlmostEqual(sum(float(row[state+"_percent"]) for state in script.STATES), 100, places=5)

    def test_stale_csv_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            path = root / "skeleton_angles/a/record_1.csv"
            with path.open(newline="") as f:
                rows = list(csv.reader(f))
            rows[1][1] = "99"
            with path.open("w", newline="") as f:
                csv.writer(f).writerows(rows)
            with self.assertRaisesRegex(ValueError, "CSV values differ"):
                script.evaluate(root)


if __name__ == "__main__":
    unittest.main()
