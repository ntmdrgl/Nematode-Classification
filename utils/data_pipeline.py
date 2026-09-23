import numpy as np
import torch
import pandas as pd
from pathlib import Path

from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

def load_dataset(cfg, label_map):
    data_dir = Path(cfg.data_dir)

    data_path = data_dir / cfg.data_file
    label_path = data_dir / cfg.label_file

    data = loadmat(data_path)["dataset_all_time"]

    data = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)

    label_encoder = LabelEncoder()

    labels = pd.read_csv(label_path)
    labels = labels.iloc[:, 0].map(label_map)

    if labels.isna().any():
        unknown_labels = pd.read_csv(label_path).iloc[:, 0][labels.isna()].unique()

        raise ValueError(f"Labels missing from label map: {unknown_labels}")

    labels = label_encoder.fit_transform(labels)
    labels = torch.tensor(labels, dtype=torch.int64)

    class_names = list(label_encoder.classes_)

    return data, labels, class_names

def _normalize_fold(fold):
    train_data = fold["train_data"]

    train_mean = train_data.mean(dim=(0, 1), keepdim=True)
    train_std = train_data.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)

    fold["train_data"] = (fold["train_data"] - train_mean) / train_std
    if fold["val_data"] is not None:
        fold["val_data"] = (fold["val_data"] - train_mean) / train_std
    fold["test_data"] = (fold["test_data"] - train_mean) / train_std

    return fold

def _create_loader(
    data, 
    labels, 
    batch_size, 
    shuffle
):
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def _create_folds(
    data, 
    labels, 
    num_folds, 
    train_size, 
    val_size, 
    test_size, 
    seed, 
    shuffle=True
):
    total = train_size + val_size + test_size

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_size + val_size + test_size must equal 1. Got {total}"
        )

    expected_test_size = 1.0 / num_folds

    if not np.isclose(test_size, expected_test_size):
        raise ValueError(
            f"For k={num_folds}, test_size should be "
            f"{expected_test_size:.3f}. Got test_size={test_size}."
        )

    y = labels.detach().cpu().numpy()
    dummy_x = np.zeros(len(y))

    n_total = len(labels)
    n_val = round(val_size * n_total)

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=shuffle,
        random_state=seed if shuffle else None,
    )

    splits = []

    for fold, (trainval_idx_np, test_idx_np) in enumerate(skf.split(dummy_x, y)):
        if n_val == 0:
            train_idx_np = trainval_idx_np
            val_idx = torch.empty(0, dtype=torch.long)
        else:
            y_trainval = y[trainval_idx_np]

            val_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=n_val,
                random_state=seed + fold,
            )

            train_rel_idx, val_rel_idx = next(
                val_splitter.split(
                    np.zeros(len(y_trainval)),
                    y_trainval,
                )
            )

            train_idx_np = trainval_idx_np[train_rel_idx]
            val_idx_np = trainval_idx_np[val_rel_idx]
            val_idx = torch.from_numpy(val_idx_np).long()

        train_idx = torch.from_numpy(train_idx_np).long()
        test_idx = torch.from_numpy(test_idx_np).long()

        splits.append({
            "fold": fold,
            "train_data": data[train_idx],
            "train_labels": labels[train_idx],
            "val_data": None if n_val == 0 else data[val_idx],
            "val_labels": None if n_val == 0 else labels[val_idx],
            "test_data": data[test_idx],
            "test_labels": labels[test_idx],
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        })

    return splits

def create_fold_loaders(data, labels, cfg):
    folds = _create_folds(
        data=data,
        labels=labels,
        num_folds=cfg.num_folds,
        train_size=(1 - 1 / cfg.num_folds),
        val_size=0.0,
        test_size=(1 / cfg.num_folds),
        seed=cfg.seed,
        shuffle=True,
    )

    fold_loaders = []

    for idx, fold in enumerate(folds):
        fold = _normalize_fold(fold)
        
        print(f"Fold {idx}:")
        print("\tTrain data:", fold['train_data'].shape, fold['train_data'].dtype)
        print("\tTrain label:", fold['train_labels'].shape, fold['train_labels'].dtype)
        
        if fold['val_data'] is not None:
            print("\tVal data:", fold['val_data'.shape], fold['val_data'].dtype)
            print("\tVal label:", fold['val_labels'.shape], fold['val_labels'].dtype)
        
        print("\tTest data:", fold['test_data'].shape, fold['test_data'].dtype)
        print("\tTest label:", fold['test_labels'].shape, fold['test_labels'].dtype)

        loaders = {
            "train": _create_loader(
                fold["train_data"],
                fold["train_labels"],
                cfg.batch_size,
                shuffle=True,
            ),
            "test": _create_loader(
                fold["test_data"],
                fold["test_labels"],
                cfg.batch_size,
                shuffle=False,
            ),
        }

        if fold["val_data"] is not None:
            loaders["val"] = _create_loader(
                fold["val_data"],
                fold["val_labels"],
                cfg.batch_size,
                shuffle=False,
            )
        else:
            loaders["val"] = None

        fold_loaders.append(loaders)

    return fold_loaders