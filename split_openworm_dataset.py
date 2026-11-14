from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.dataset import create_split


def load_raw_dataset(source, workers=16):
    with np.load(source, allow_pickle=False) as archive:
        keys = archive["record_keys"].tolist()
        classes, labels = np.unique(archive["labels"], return_inverse=True)
        if not keys or len(keys) != len(labels):
            raise ValueError("Expected nonempty recordings with one label per recording.")
        shape = archive[keys[0]].shape
    if len(shape) != 2 or not all(shape):
        raise ValueError("Recordings must have shape (timesteps, features).")

    X = np.empty((len(keys), shape[1], shape[0]), dtype=np.float32)
    with tqdm(total=len(keys), desc="Loading recordings", unit="record") as progress:
        def load_chunk(indices):
            with np.load(source, allow_pickle=False) as archive:
                for i in indices:
                    values = archive[keys[i]]
                    if values.shape != shape:
                        raise ValueError(f"{keys[i]} has shape {values.shape}; expected {shape}. "
                                         "Equal recording lengths are required; no trimming is applied.")
                    X[i] = values.T
                    del values
                    progress.update(1)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(load_chunk, range(i, len(keys), workers))
                       for i in range(min(workers, len(keys)))]
            for future in as_completed(futures):
                future.result()

    return torch.from_numpy(X), torch.as_tensor(labels, dtype=torch.long), classes


def split_dataset(name, workers=16):
    data_dir = Path(__file__).resolve().parent / "data"
    source = data_dir / "OpenWorm_raw" / f"openworm_{name}_raw.npz"
    X, y, classes = load_raw_dataset(source, workers)
    X_train, y_train, X_test, y_test = create_split(X, y, fraction=0.15, seed=42)
    del X, y

    output = data_dir / f"openworm_{name}.npz"
    print(f"Saving {output.name}...", flush=True)
    np.savez(
        output,
        X_train=X_train.numpy(),
        y_train=y_train.numpy(),
        X_test=X_test.numpy(),
        y_test=y_test.numpy(),
        classes=classes,
    )
    print(f"Saved {output.name}: train={tuple(X_train.shape)}, test={tuple(X_test.shape)}")


if __name__ == "__main__":
    for name in ("eigenworms", "manual_features", "skeleton_angles"):
    # for name in tqdm(("eigenworms",), desc="Datasets"):
        split_dataset(name)
