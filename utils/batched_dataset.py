"""Lazy batches from uncompressed NPZ members, normalized using cached statistics."""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler

from utils.dataset import load_normalization
from utils.openworm_io import feature_array


class MappedDataset(Dataset):
    """Keep only labels/indices in RAM; reopen the mapping in each worker."""

    def __init__(self, path, key, labels, stats, indices=None):
        self.path, self.key = str(Path(path).resolve()), key
        values = feature_array(self.path, key)
        if values.ndim != 3 or any(size == 0 for size in values.shape):
            raise ValueError("Expected nonempty (samples, channels, timesteps) features")
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu":
            raise ValueError("Expected one integer label per recording")
        self.indices = (np.arange(len(values)) if indices is None
                        else np.asarray(indices, dtype=np.int64))
        if (self.indices.ndim != 1 or not len(self.indices)
                or self.indices.min() < 0 or self.indices.max() >= len(values)):
            raise ValueError("Invalid recording indices")
        self.shape = (len(self.indices), *values.shape[1:])
        self.labels = torch.as_tensor(labels[self.indices].astype(np.int64, copy=True))
        self.stats = {key: torch.as_tensor(stats[key], dtype=torch.float32).detach().cpu().clone()
                      for key in ("mean", "std")}
        if (any(tuple(v.shape) != (1, self.shape[1], 1) or not torch.isfinite(v).all()
                for v in self.stats.values()) or (self.stats["std"] <= 0).any()):
            raise ValueError("Invalid normalization statistics for these channels")
        del values
        self._values = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        # Pickling an np.memmap can serialize the entire array on Windows.
        state = self.__dict__.copy()
        state["_values"] = None
        return state

    def __getitem__(self, index):
        if self._values is None:
            self._values = feature_array(self.path, self.key)
        scalar = np.isscalar(index)
        positions = np.atleast_1d(index)
        batch = np.empty((len(positions), *self.shape[1:]), dtype=np.float32)
        # Copy directly into one writable batch; avoid a second collate/stack copy.
        for row, position in enumerate(positions):
            np.copyto(batch[row], self._values[self.indices[position]], casting="unsafe")
        features = torch.from_numpy(batch)
        features.sub_(self.stats["mean"]).div_(self.stats["std"])
        labels = self.labels[torch.as_tensor(positions, dtype=torch.long)]
        return (features[0], labels[0]) if scalar else (features, labels)


def mapped_loader(dataset, batch_size, shuffle=False, num_workers=0):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    batches = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    options = {"prefetch_factor": 1} if num_workers else {}
    # The sampler supplies index lists; the dataset returns a complete batch.
    return DataLoader(dataset, batch_size=None, sampler=batches, num_workers=num_workers, **options)


def create_dataset_loaders(data_path, batch_size, seed=42, num_workers=0):
    """Split indices, keeping feature arrays on disk throughout setup and training."""
    with np.load(data_path, allow_pickle=False) as data:
        labels, test_labels = data["y_train"], data["y_test"]
    fit, validation = train_test_split(np.arange(len(labels)), test_size=.15 / (1 - .15),
                                       random_state=seed, stratify=labels)
    stats = load_normalization(data_path, seed)
    if stats is None:
        if Path(data_path).stem.startswith("openworm_"):
            raise ValueError("Saved train_mean/train_std missing; run preprocess_openworm_dataset.py first")
        # The original small EigenWorms dataset has no cached statistics.
        from preprocess_openworm_dataset import training_statistics
        values = feature_array(data_path, "X_train")
        mean, std = training_statistics(values, fit)
        del values
        stats = {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std)}
    train = MappedDataset(data_path, "X_train", labels, stats, fit)
    val = MappedDataset(data_path, "X_train", labels, stats, validation)
    test = MappedDataset(data_path, "X_test", test_labels, stats)
    return (mapped_loader(train, batch_size, True, num_workers),
            mapped_loader(val, batch_size, num_workers=num_workers),
            mapped_loader(test, batch_size, num_workers=num_workers), stats)
