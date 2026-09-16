from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import numpy as np
import torch


def create_folds(
    data,
    labels,
    num_folds,
    train_size,
    val_size,
    test_size,
    seed,
    shuffle=True,
):
    """
    Creates stratified K-fold splits.

    Example with 300 samples, num_folds=10:
        train = 240
        val   = 30
        test  = 30
    """
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