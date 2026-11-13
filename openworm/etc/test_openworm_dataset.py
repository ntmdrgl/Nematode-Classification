"""Check NPZ record labels, feature separation and variable-length storage."""
import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openworm_dataset as script


class DatasetTests(unittest.TestCase):
    def test_parallel_matches_serial_in_every_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            for mode in ("all", *script.DATASET_NAMES):
                with self.subTest(mode=mode):
                    paths = script.build_datasets(root, mode)
                    expected = []
                    for path in paths:
                        with np.load(path, allow_pickle=False) as archive:
                            expected.append({key: archive[key].copy() for key in archive.files})
                    self.assertEqual(script.build_datasets(root, mode, workers=2), paths)
                    for path, arrays in zip(paths, expected):
                        with np.load(path, allow_pickle=False) as archive:
                            self.assertEqual(archive.files, list(arrays))
                            for key, values in arrays.items():
                                np.testing.assert_array_equal(archive[key], values)

    def test_parallel_validation_failure_preserves_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            paths = script.build_datasets(root)
            before = [path.read_bytes() for path in paths]
            source = root / "manual_features/n2 male/record_123.csv"
            source.write_text(source.read_text().replace("0.04,", "0.05,"))
            for mode in ("all", "manual_features"):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "0.04-second"):
                    script.build_datasets(root, mode, workers=2)
                self.assertEqual([path.read_bytes() for path in paths], before)
                self.assertEqual(list(root.glob("*.part")), [])

    def test_invalid_workers(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            script.build_datasets(Path("unused"), workers=0)

    def test_individual_modes_need_only_their_own_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            for dataset, columns in zip(script.DATASET_NAMES, (48, 47, 6, 16)):
                with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as output:
                    output = Path(output)
                    folder = "manual_features" if dataset == "eigenworms" else dataset
                    shutil.copytree(root/folder, output/folder)
                    selected_filename = script.OUTPUTS[script.DATASET_NAMES.index(dataset)]
                    for name in script.OUTPUTS:
                        if name != selected_filename:
                            (output/name).write_bytes(b"unchanged")
                    paths = script.build_datasets(output, dataset)
                    self.assertEqual(paths, [output/selected_filename])
                    with np.load(paths[0], allow_pickle=False) as archive:
                        self.assertEqual(len(archive["feature_names"]), columns)
                        self.assertEqual(archive["data_000000"].shape[1], columns)
                        offset = 47 if dataset == "eigenworms" else 0
                        np.testing.assert_array_equal(archive["data_000000"][0], np.arange(columns)+offset)
                    for name in script.OUTPUTS:
                        if name != selected_filename:
                            self.assertEqual((output/name).read_bytes(), b"unchanged")

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
