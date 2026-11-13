"""Evaluation tests for feature statuses, weighted totals and stale sources."""
import csv
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openworm_evaluate as script


class EvaluationTests(unittest.TestCase):
    def test_individual_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            expected = script.evaluate(root)
            for dataset, filename, _, _ in script.DATASETS:
                with self.subTest(dataset=dataset):
                    # Unselected archives must never be opened, even if invalid.
                    saved = {}
                    for _, other, _, _ in script.DATASETS:
                        if other != filename:
                            saved[other] = (root / other).read_bytes()
                            (root / other).write_bytes(b"invalid unselected archive")
                    try:
                        selected = [row for row in expected if row["dataset"] == dataset]
                        for workers in (1, 2):
                            self.assertEqual(script.evaluate(root, workers=workers, dataset=dataset), selected)
                        self.assertEqual(script.evaluate(root, dataset=dataset, totals_only=True), selected[:1])
                    finally:
                        for other, content in saved.items():
                            (root / other).write_bytes(content)

    def test_missing_selected_dataset_does_not_evaluate_others(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            (root / script.DATASETS[0][1]).unlink()
            with self.assertRaisesRegex(ValueError, "No dataset archives found"):
                script.evaluate(root, dataset="skeleton_angles")

    def test_invalid_dataset(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            script.evaluate(Path("unused"), dataset="unknown")

    def test_totals_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            expected = [row for row in script.evaluate(root) if row["scope"] == "total"]
            for workers in (1, 2):
                self.assertEqual(script.evaluate(root, workers=workers, totals_only=True), expected)
            (root / script.DATASETS[3][1]).unlink()
            self.assertEqual(script.evaluate(root, totals_only=True), expected[:3])

    def test_parallel_matches_serial_and_skips_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            self.assertEqual(script.evaluate(root, workers=2), script.evaluate(root))
            (root / script.DATASETS[3][1]).unlink()
            self.assertEqual(script.evaluate(root, workers=2), script.evaluate(root))

    def test_parallel_rejects_stale_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            path = root / "skeleton_angles/a/record_1.csv"
            with path.open(newline="") as source:
                rows = list(csv.reader(source))
            rows[1][1] = "99"
            with path.open("w", newline="") as destination:
                csv.writer(destination).writerows(rows)
            with self.assertRaisesRegex(ValueError, "CSV values differ"):
                script.evaluate(root, workers=2)

    def test_invalid_workers(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            script.evaluate(Path("unused"), workers=0)

    def test_missing_datasets_and_total_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            for specification in script.DATASETS[1:]:
                (root / specification[1]).unlink()
            progress = MagicMock()
            with patch.object(script, "tqdm") as bar:
                bar.return_value.__enter__.return_value = progress
                rows = script.evaluate(root)
            self.assertEqual(bar.call_args.kwargs["total"], 2)
            self.assertEqual(progress.update.call_count, 2)
            self.assertEqual(bar.write.call_count, 3)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["scope"], "total")
            self.assertTrue(all(row["dataset"] == "skeleton_angles" for row in rows))

    def test_no_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No dataset archives found"):
                script.evaluate(Path(directory))

    def test_missing_source_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            (root / "skeleton_angles/a/record_1.csv").unlink()
            with self.assertRaises(FileNotFoundError):
                script.evaluate(root)

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
