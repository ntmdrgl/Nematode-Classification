
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

def load_normalization(data_path, seed):
    """Load cached statistics, ensuring they match the training/validation split."""
    with np.load(data_path, allow_pickle=False) as data:
        if "train_mean" not in data and "train_std" not in data:
            return None
        if (int(data["normalization_version"]) != 1
                or int(data["normalization_seed"]) != seed
                or float(data["normalization_fraction"]) != .15 / (1 - .15)):
            raise ValueError("Cached normalization does not match this training split")
        fit, validation = train_test_split(np.arange(len(data["y_train"])),
            test_size=.15 / (1 - .15), random_state=seed, stratify=data["y_train"])
        if (not np.array_equal(fit, data["normalization_train_indices"])
                or not np.array_equal(validation, data["normalization_val_indices"])):
            raise ValueError("Cached normalization indices do not match this training split")
        stats = {key: torch.as_tensor(data[f"train_{key}"], dtype=torch.float32)
                 for key in ("mean", "std")}
    if (stats["mean"].ndim != 3 or stats["mean"].shape[0] != 1
            or stats["mean"].shape[2] != 1 or stats["std"].shape != stats["mean"].shape
            or not all(torch.isfinite(value).all() for value in stats.values())
            or (stats["std"] <= 0).any()):
        raise ValueError("Invalid cached mean/std")
    return stats

def load_dataset(data_path):
    with np.load(data_path, allow_pickle=False) as data:
        X_train = torch.as_tensor(data["X_train"], dtype=torch.float32)
        y_train = torch.as_tensor(data["y_train"], dtype=torch.long)
        X_test = torch.as_tensor(data["X_test"], dtype=torch.float32)
        y_test = torch.as_tensor(data["y_test"], dtype=torch.long)

    return X_train, y_train, X_test, y_test

def load_unsplit_dataset(data_path):
    """Load X/y arrays, encoding labels in sorted np.unique order."""
    with np.load(data_path, allow_pickle=False) as data:
        X = torch.as_tensor(data["X"], dtype=torch.float32)
        _, labels = np.unique(data["y"], return_inverse=True)
        y = torch.as_tensor(labels, dtype=torch.long)

    return X, y

def create_split(
    X_train,
    y_train,
    fraction,
    seed
):
    train_indices, val_indices = train_test_split(
        np.arange(len(y_train)),
        test_size=fraction,
        random_state=seed,
        stratify=y_train.numpy(),
    )

    train_indices = torch.as_tensor(train_indices, dtype=torch.long)
    val_indices = torch.as_tensor(val_indices, dtype=torch.long)

    return (
        X_train[train_indices],
        y_train[train_indices],
        X_train[val_indices],
        y_train[val_indices],
    )

def create_training_loader(
    X_train,
    y_train,
    batch_size: 16,
    stats=None,
):
    if stats is None:
        stats = {
            "mean": X_train.mean(dim=(0, 2), keepdim=True),
            "std": X_train.std(dim=(0, 2), correction=0, keepdim=True).clamp_min(1e-8),
        }
    if any(tuple(value.shape) != (1, X_train.shape[1], 1) for value in stats.values()):
        raise ValueError("Normalization channel count does not match the training data")

    X_train = (X_train - stats["mean"]) / stats["std"]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, stats

def create_testing_loader(
    X_test,
    y_test,
    stats,
    batch_size: 16,
):
    X_test = (X_test - stats["mean"]) / stats["std"]

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
