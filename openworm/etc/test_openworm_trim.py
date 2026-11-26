"""Check strict selection, shortest retained length and aligned record labels."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import openworm.openworm_trim as script


class TrimTests(unittest.TestCase):
    def test_individual_datasets_need_only_selected_source(self):
        for dataset, width in script.ALL_DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixtures(root)
                untouched = []
                for other, _ in script.DATASETS:
                    if other != dataset:
                        (root / f"openworm_{other}_dataset.npz").unlink()
                        path = root / f"openworm_{other}_trimmed_890_dataset.npz"
                        path.write_bytes(b"unchanged")
                        untouched.append(path)
                with patch.object(script, "MIN_SAMPLES", 3):
                    for workers in (1, 2):
                        outputs = script.trim_datasets(root, workers=workers, dataset=dataset)
                        self.assertEqual(len(outputs), 1)
                        self.assertEqual(outputs[0].name, f"openworm_{dataset}_trimmed_890_dataset.npz")
                        with np.load(outputs[0], allow_pickle=False) as archive:
                            self.assertEqual(archive["labels"].tolist(), ["c", "d"])
                            self.assertEqual(archive["data_000000"].shape, (5, width))
                        for path in untouched:
                            self.assertEqual(path.read_bytes(), b"unchanged")

    def test_invalid_dataset(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            script.trim_datasets(Path("unused"), dataset="unknown")

    def test_parallel_matches_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            with patch.object(script, "MIN_SAMPLES", 3):
                serial = script.trim_datasets(root, root / "serial")
                parallel = script.trim_datasets(root, root / "parallel", workers=2)
            for first, second in zip(serial, parallel):
                with np.load(first, allow_pickle=False) as expected, np.load(second, allow_pickle=False) as actual:
                    self.assertEqual(expected.files, actual.files)
                    for key in expected.files:
                        np.testing.assert_array_equal(expected[key], actual[key])

    def test_parallel_failure_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            with patch.object(script, "MIN_SAMPLES", 3):
                outputs = script.trim_datasets(root)
                before = [path.read_bytes() for path in outputs]
                source = root / "openworm_eigenworms_dataset.npz"
                with np.load(source, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                arrays["data_000003"] = arrays["data_000003"].astype(float)
                arrays["data_000003"][0, 0] = np.nan
                np.savez_compressed(source, **arrays)
                with self.assertRaisesRegex(ValueError, "Invalid or changed record"):
                    script.trim_datasets(root, workers=2)
                self.assertEqual([path.read_bytes() for path in outputs], before)
                self.assertEqual(list(root.glob("*.part")), [])

    def test_invalid_workers(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            script.trim_datasets(Path("unused"), workers=0)

    def fixtures(self, root, mismatch=False):
        for dataset, width in script.DATASETS:
            lengths = [2, 3, 5, 7]
            if mismatch and dataset == "eigenworms":
                lengths[-1] += 1
            arrays = {f"data_{i:06d}": np.arange(n * width).reshape(n, width)
                      for i, n in enumerate(lengths)}
            np.savez_compressed(root / f"openworm_{dataset}_dataset.npz",
                                labels=["a", "b", "c", "d"],
                                record_ids=[f"record_{i}" for i in range(4)],
                                record_keys=list(arrays),
                                feature_names=[f"f_{i}" for i in range(width)], **arrays)

    def test_strict_cutoff_and_shortest_retained_record(self):
        self.assertEqual(script.MIN_SAMPLES, 22250)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            originals = {p: p.read_bytes() for p in root.glob("*.npz")}
            with patch.object(script, "MIN_SAMPLES", 3):
                outputs = script.trim_datasets(root)
            self.assertEqual(len(outputs), len(script.ALL_DATASETS))
            for path, (dataset, width) in zip(outputs, script.ALL_DATASETS):
                self.assertEqual(path.name, f"openworm_{dataset}_trimmed_890_dataset.npz")
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(archive["labels"].tolist(), ["c", "d"])
                    self.assertEqual(archive["record_ids"].tolist(), ["record_2", "record_3"])
                    for key in archive["record_keys"]:
                        self.assertEqual(archive[key].shape, (5, width))
                        np.testing.assert_array_equal(archive[key], np.arange(5 * width).reshape(5, width))
            for path, before in originals.items():
                self.assertEqual(path.read_bytes(), before)

    def test_mismatched_sources_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root, mismatch=True)
            with self.assertRaisesRegex(ValueError, "identical"):
                script.trim_datasets(root)
            self.assertEqual(list(root.glob("*trimmed*")), [])

    def test_empty_selection_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            with self.assertRaisesRegex(ValueError, "No records"):
                script.trim_datasets(root)
            self.assertEqual(list(root.glob("*trimmed*")), [])


if __name__ == "__main__":
    unittest.main()
