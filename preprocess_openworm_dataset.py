"""Append training-only mean/std to data/openworm_*.npz without copying features.

python preprocess_openworm_dataset.py --dataset all --seed 42
Statistics exclude the validation subset used by train.py (15% of all samples).
The existing feature arrays, labels, and train/test ordering remain unchanged.
"""
import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.openworm_io import map_array

VALIDATION_FRACTION = .15 / (1 - .15)
VERSION = 1


def training_statistics(values, indices):
    count = 0
    mean = np.zeros(values.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for index in tqdm(indices, desc="Training mean/std", unit="record", mininterval=10):
        block = np.array(values[index], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError(f"Nonfinite training values in recording {index}")
        n = block.shape[1]
        current = block.mean(axis=1)
        block -= current[:, None]
        local_m2 = np.einsum("ij,ij->i", block, block)
        delta = current - mean
        total = count + n
        m2 += local_m2 + delta**2 * (count * n / total)
        mean += delta * (n / total)
        count = total
    if not count:
        raise ValueError("No training samples")
    return (mean.astype(np.float32)[None, :, None],
            np.maximum(np.sqrt(m2 / count), 1e-8).astype(np.float32)[None, :, None])


def preprocess_dataset(name, data_dir=None, seed=42):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data"
    name = name.removeprefix("openworm_").removesuffix("_split")
    path = root / f"openworm_{name}.npz"
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["y_train"]
        if "normalization_version" in archive:
            if (int(archive["normalization_version"]) == VERSION
                    and int(archive["normalization_seed"]) == seed
                    and float(archive["normalization_fraction"]) == VALIDATION_FRACTION):
                print(f"Already preprocessed: {path}", flush=True)
                return path
            raise ValueError(f"{path}: existing normalization uses different settings")
        if "train_mean" in archive or "train_std" in archive:
            raise ValueError(f"{path}: unrecognized existing normalization metadata")
    fit, validation = train_test_split(np.arange(len(labels)), test_size=VALIDATION_FRACTION,
                                       random_state=seed, stratify=labels)
    values = map_array(path, "X_train")
    if values.ndim != 3 or len(values) != len(labels):
        raise ValueError("Expected (samples, channels, timesteps) with aligned labels")
    mean, std = training_statistics(values, fit)
    del values  # Release Windows file mapping before appending ZIP members.
    metadata = dict(train_mean=mean, train_std=std, normalization_seed=seed,
                    normalization_train_indices=fit, normalization_val_indices=validation,
                    normalization_fraction=VALIDATION_FRACTION, normalization_version=VERSION)
    # Back up only the small central directory; restore it if an append raises.
    with ZipFile(path) as archive:
        directory_offset = archive.start_dir
    original_size = path.stat().st_size
    with path.open("rb") as source:
        source.seek(directory_offset)
        directory = source.read()
    try:
        with ZipFile(path, "a") as archive:
            for key, value in metadata.items():
                with archive.open(key + ".npy", "w", force_zip64=True) as destination:
                    np.lib.format.write_array(destination, np.asarray(value), allow_pickle=False)
    except BaseException:
        with path.open("r+b") as destination:
            destination.seek(directory_offset)
            destination.write(directory)
            destination.truncate(original_size)
        raise
    print(f"Saved mean/std in {path}; fit={len(fit)}, validation={len(validation)}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_files = sorted(args.data_dir.glob("openworm_*.npz"))
    names = ([p.stem.removesuffix("_split") for p in split_files]
        if args.dataset == "all" else [args.dataset])
    if not names:
        parser.error("No split datasets found")
    for name in names:
        preprocess_dataset(name, args.data_dir, args.seed)


if __name__ == "__main__":
    main()
