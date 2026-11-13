"""Check NPZ record labels, feature separation and variable-length storage."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openworm_dataset as script


class DatasetTests(unittest.TestCase):
    def fixtures(self, root):
        for strain, length in (("cb5", 3), ("n2 male", 5)):
            for kind in ("skeleton_angles", "manual_features"):
                folder = root / kind / strain
                folder.mkdir(parents=True)
                names = script.ANGLES if kind == "skeleton_angles" else [f"feature_{j}" for j in range(47)] + script.EIGEN
                with (folder / "record_123.csv").open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["time_s", *names, "skeleton_status"])
                    for i in range(length):
                        writer.writerow([i*.04, *[i*100+j for j in range(len(names))], "real"])
        folder = root / "matlab_features/cb5"
        folder.mkdir(parents=True)
        with (folder / "record_123.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", *[f"matlab_{i}" for i in range(16)], "matlab_0_status"])
            for i in range(3):
                writer.writerow([i*.04, *range(16), "real"])

    def test_labeled_ragged_records_and_feature_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            paths = script.build_datasets(root)
            self.assertEqual(len(paths), 4)
            for path, columns in zip(paths, (48, 47, 6)):
                with np.load(path, allow_pickle=False) as dataset:
                    self.assertEqual(dataset["labels"].tolist(), ["cb5", "n2 male"])
                    self.assertEqual(dataset["record_ids"].tolist(), ["record_123"]*2)
                    self.assertEqual(len(dataset["feature_names"]), columns)
                    for i, length in enumerate((3, 5)):
                        data = dataset[dataset["record_keys"][i]]
                        self.assertEqual(data.shape, (length, columns))
                        offset = 47 if columns == 6 else 0
                        np.testing.assert_array_equal(data[0], np.arange(columns)+offset)
            with np.load(paths[3], allow_pickle=False) as matlab:
                self.assertEqual(matlab["labels"].tolist(), ["cb5"])
                self.assertEqual(matlab["data_000000"].shape, (3,16))
                np.testing.assert_array_equal(matlab["data_000000"][0], np.arange(16))

    def test_unmatched_records_do_not_replace_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            paths = script.build_datasets(root)
            before = [path.read_bytes() for path in paths]
            (root / "manual_features/cb5/record_123.csv").unlink()
            with self.assertRaisesRegex(ValueError, "same nonempty set"):
                script.build_datasets(root)
            self.assertEqual([path.read_bytes() for path in paths], before)


if __name__ == "__main__":
    unittest.main()
