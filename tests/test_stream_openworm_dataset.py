import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from stream_openworm_dataset import convert_dataset


class StreamingConversionTests(unittest.TestCase):
    def test_matches_existing_split_format_and_values(self):
        for workers in (1, 3):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                keys = [f"data_{i:06d}" for i in range(40)]
                labels = np.array(["strain_b", "strain_a"] * 20)
                records = {key: (np.arange(15).reshape(5, 3) + i * 100).astype(np.float32)
                           for i, key in enumerate(keys)}
                source, output = root / "raw.npz", root / "output.npz"
                np.savez_compressed(source, record_keys=keys, labels=labels, **records)
                convert_dataset(source, output, workers=workers)
                classes, encoded = np.unique(labels, return_inverse=True)
                train, test = train_test_split(np.arange(len(keys)), test_size=0.15,
                                               random_state=42, stratify=encoded)
                with np.load(output, allow_pickle=False) as actual:
                    self.assertEqual(set(actual.files),
                                     {"X_train", "y_train", "X_test", "y_test", "classes"})
                    np.testing.assert_array_equal(actual["classes"], classes)
                    for name, indices in (("train", train), ("test", test)):
                        expected = np.stack([records[keys[i]].T for i in indices])
                        np.testing.assert_array_equal(actual[f"X_{name}"], expected)
                        np.testing.assert_array_equal(actual[f"y_{name}"], encoded[indices])
                        self.assertEqual(actual[f"X_{name}"].dtype, np.float32)
                        self.assertEqual(actual[f"y_{name}"].dtype, np.int64)
                self.assertFalse(output.with_suffix(".npz.part").exists())
                with self.assertRaises(FileExistsError):
                    convert_dataset(source, output, workers=workers)

    def test_rejects_inconsistent_lengths_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "raw.npz", root / "output.npz"
            np.savez(source, record_keys=["a", "b"], labels=["x", "y"],
                     a=np.zeros((5, 3)), b=np.zeros((6, 3)))
            with self.assertRaisesRegex(ValueError, "Equal recording lengths"):
                convert_dataset(source, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".npz.part").exists())


if __name__ == "__main__":
    unittest.main()
