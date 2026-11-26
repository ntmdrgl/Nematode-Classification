import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from preprocess_openworm_dataset import preprocess_dataset, VALIDATION_FRACTION
from utils.dataset import load_normalization, create_training_loader
from utils.openworm_io import map_array
from verify_openworm_splits import verify_splits


class PreprocessingTests(unittest.TestCase):
    def test_append_statistics_preserves_data_and_excludes_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "openworm_example.npz"
            labels = np.array([0, 1] * 20)
            values = np.arange(40 * 3 * 5, dtype=np.float32).reshape(40, 3, 5)
            fit, val = train_test_split(np.arange(40), test_size=VALIDATION_FRACTION,
                                        random_state=42, stratify=labels)
            values[val] += 100000  # Would badly skew statistics if validation leaked.
            values[:, 2, :] = 7  # Constant channel must get a positive std floor.
            np.savez(path, X_train=values, y_train=labels, X_test=values[:4] + 900000,
                     y_test=labels[:4], classes=["a", "b"])
            with ZipFile(path) as z:
                before = {i.filename: (i.CRC, i.header_offset, i.file_size) for i in z.infolist()}
            preprocess_dataset("example", root)
            stats = load_normalization(path, 42)
            expected_mean = values[fit].mean(axis=(0, 2), keepdims=True, dtype=np.float64)
            expected_std = np.maximum(values[fit].std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-8)
            np.testing.assert_allclose(stats["mean"].numpy(), expected_mean, rtol=1e-6)
            np.testing.assert_allclose(stats["std"].numpy(), expected_std, rtol=1e-6)
            np.testing.assert_array_equal(map_array(path, "X_train"), values)
            with ZipFile(path) as z:
                for name, details in before.items():
                    i = z.getinfo(name)
                    self.assertEqual((i.CRC, i.header_offset, i.file_size), details)
            size = path.stat().st_size
            preprocess_dataset("example", root)
            self.assertEqual(size, path.stat().st_size)
            with self.assertRaises(ValueError):
                load_normalization(path, 43)
            with patch.object(torch.Tensor, "mean", side_effect=AssertionError("runtime mean")), \
                    patch.object(torch.Tensor, "std", side_effect=AssertionError("runtime std")):
                loader, _ = create_training_loader(torch.from_numpy(values[fit]),
                    torch.from_numpy(labels[fit]), 4, stats=stats)
                actual = loader.dataset.tensors[0].numpy()
            np.testing.assert_allclose(actual, (values[fit] - stats["mean"].numpy()) /
                                       stats["std"].numpy(), rtol=1e-6)

    def test_verifier_detects_wrong_sample_even_when_labels_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "OpenWorm_raw").mkdir()
            labels = np.array(["a", "b"] * 20)
            keys = [f"data_{i:06d}" for i in range(40)]
            records = {key: np.full((5, 3), i, dtype=np.float32) for i, key in enumerate(keys)}
            np.savez_compressed(root / "OpenWorm_raw/openworm_example_raw.npz",
                record_keys=keys, labels=labels, record_ids=[str(i) for i in range(40)], **records)
            classes, encoded = np.unique(labels, return_inverse=True)
            train, test = train_test_split(np.arange(40), test_size=.15, random_state=42, stratify=encoded)
            x = np.stack([records[keys[i]].T for i in train])
            path = root / "openworm_example.npz"
            def save():
                np.savez(path, X_train=x, y_train=encoded[train],
                         X_test=np.stack([records[keys[i]].T for i in test]),
                         y_test=encoded[test], classes=classes)
            save()
            verify_splits(root, workers=2)
            x[0] = x[1]  # Keep all labels intact while corrupting the sample identity.
            save()
            with self.assertRaises(AssertionError):
                verify_splits(root, workers=2)


if __name__ == "__main__":
    unittest.main()
