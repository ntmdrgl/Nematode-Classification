import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from preprocess_openworm_dataset import preprocess_dataset
from utils.batched_dataset import create_dataset_loaders
from utils.evaluation import load_evaluation
from utils.training import train_model, evaluate_model


class BatchedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "openworm_example.npz"
        self.train = np.arange(40 * 3 * 5, dtype=np.float32).reshape(40, 3, 5)
        self.test = self.train[:6] + 900
        self.labels = np.array([0, 1] * 20)
        np.savez(self.path, X_train=self.train, X_test=self.test,
                 y_train=self.labels, y_test=self.labels[:6], classes=["a", "b"])
        preprocess_dataset("example", self.root)

    def test_cached_batches_match_eager_math_without_loading_feature_members(self):
        original = np.lib.npyio.NpzFile.__getitem__
        def metadata_only(archive, key):
            if key in ("X_train", "X_test"):
                raise AssertionError("Eager feature loading attempted")
            return original(archive, key)
        with patch.object(np.lib.npyio.NpzFile, "__getitem__", metadata_only), \
                patch.object(torch.Tensor, "mean", side_effect=AssertionError("runtime mean")), \
                patch.object(torch.Tensor, "std", side_effect=AssertionError("runtime std")):
            train, val, test, stats = create_dataset_loaders(self.path, 7)
            for loader in (train, val, test):
                self.assertIsNone(loader.dataset._values)
            for loader, source, labels in ((val, self.train, self.labels),
                                           (test, self.test, self.labels[:6])):
                batches = list(loader)
                actual = torch.cat([x for x, _ in batches]).numpy()
                indices = loader.dataset.indices
                expected = (source[indices] - stats["mean"].numpy()) / stats["std"].numpy()
                np.testing.assert_allclose(actual, expected, rtol=1e-6)
                np.testing.assert_array_equal(torch.cat([y for _, y in batches]), labels[indices])
            self.assertEqual(len(set(train.dataset.indices) & set(val.dataset.indices)), 0)
            self.assertEqual(sorted([*train.dataset.indices, *val.dataset.indices]), list(range(40)))
            # Seeded shuffling covers the subset exactly once, including the last partial batch.
            sampled = list(train.sampler)
            self.assertEqual(sorted(i for batch in sampled for i in batch), list(range(len(train.dataset))))
        with np.load(self.path) as data:
            np.testing.assert_array_equal(data["X_train"], self.train)
            np.testing.assert_array_equal(data["X_test"], self.test)

    def test_worker_spawn_reopens_mapping_and_keeps_pickle_small(self):
        _, _, test, _ = create_dataset_loaders(self.path, 4, num_workers=1)
        test.dataset[0]  # Open a mapping in the parent before serialization.
        serialized = pickle.dumps(test.dataset)
        self.assertLess(len(serialized), 20000)
        self.assertIsNone(pickle.loads(serialized)._values)
        batches = list(test)
        self.assertEqual([len(y) for _, y in batches], [4, 2])
        np.testing.assert_array_equal(torch.cat([y for _, y in batches]), self.labels[:6])

    def test_training_and_evaluation_accept_the_lazy_batches(self):
        train, val, _, _ = create_dataset_loaders(self.path, 7)
        model = torch.nn.Sequential(torch.nn.Conv1d(3, 2, 1),
                                    torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten())
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        loss = train_model(model, train, criterion, optimizer, "cpu")
        self.assertTrue(np.isfinite(loss))
        loss, accuracy, _ = evaluate_model(model, val, criterion, "cpu", 2)
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0)

    def test_checkpoint_evaluation_also_avoids_eager_loading(self):
        _, _, _, stats = create_dataset_loaders(self.path, 4)
        cfg = OmegaConf.create(dict(dataset="openworm_example", model="example",
                                    data_path=str(self.path), batch_size=4))
        model = torch.nn.Linear(3, 2)
        checkpoint = dict(config=dict(model="example", model_params=dict(in_channels=3, num_classes=2)),
                          normalization=stats, model_state_dict=model.state_dict())
        original = np.lib.npyio.NpzFile.__getitem__
        def metadata_only(archive, key):
            if key in ("X_train", "X_test"):
                raise AssertionError("Eager feature loading attempted")
            return original(archive, key)
        with patch("utils.evaluation.torch.load", return_value=checkpoint), \
                patch("utils.evaluation.create_model", return_value=model), \
                patch.object(np.lib.npyio.NpzFile, "__getitem__", metadata_only):
            _, loader, classes, _, _ = load_evaluation(cfg)
            self.assertIsNone(loader.dataset._values)
            x, y = next(iter(loader))
            self.assertEqual(tuple(x.shape), (4, 3, 5))
            self.assertEqual(classes, ["a", "b"])

    def test_wrong_seed_rejected_and_legacy_compressed_features_streamed(self):
        with self.assertRaises(ValueError):
            create_dataset_loaders(self.path, 4, seed=43)
        with np.load(self.path) as data:
            entries = {k: data[k] for k in data.files}
        compressed = self.root / "compressed.npz"
        np.savez_compressed(compressed, **entries)
        _, _, test, stats = create_dataset_loaders(compressed, 4)
        original = np.lib.npyio.NpzFile.__getitem__
        def metadata_only(archive, key):
            if key in ("X_train", "X_test"):
                raise AssertionError("Eager compressed-feature loading attempted")
            return original(archive, key)
        with patch.object(np.lib.npyio.NpzFile, "__getitem__", metadata_only):
            actual = torch.cat([x for x, _ in test]).numpy()
        np.testing.assert_allclose(actual, (self.test - stats["mean"].numpy()) /
                                   stats["std"].numpy(), rtol=1e-6)

    def test_original_eigenworms_without_cached_statistics_still_works(self):
        legacy = self.root / "eigenworms.npz"
        np.savez_compressed(legacy, X_train=self.train, X_test=self.test,
                            y_train=self.labels, y_test=self.labels[:6], classes=["a", "b"])
        train, _, test, stats = create_dataset_loaders(legacy, 4)
        expected_mean = self.train[train.dataset.indices].mean(axis=(0, 2), keepdims=True)
        np.testing.assert_allclose(stats["mean"].numpy(), expected_mean, rtol=1e-6)
        self.assertEqual(tuple(next(iter(test))[0].shape), (4, 3, 5))


if __name__ == "__main__":
    unittest.main()
