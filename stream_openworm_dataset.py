"""Convert a raw OpenWorm NPZ to train/test NPZ without a dataset-sized allocation.

Example: python stream_openworm_dataset.py --dataset resnet_features --workers 4
Writes data/openworm_resnet_features.npz, using the same stratified split as
split_openworm_dataset.py. Requires space for one uncompressed output archive.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from shutil import disk_usage
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def recording_header(archive, key):
    with archive.open(f"{key}.npy") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(source)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(source)
        raise ValueError(f"Unsupported NPY version {version}: {key}")


def read_record(archive, key, shape):
    values = archive[key]
    if values.shape != shape:
        raise ValueError(f"Unexpected shape for {key}: {values.shape}, expected {shape}")
    return np.ascontiguousarray(values.T, dtype=np.float32)


def write_features(output, name, indices, readers, keys, shape, pool, progress):
    # Each reader has at most one active job; the queue bounds memory use.
    jobs = iter(indices)
    pending = deque()

    def submit(reader):
        index = next(jobs, None)
        if index is not None:
            pending.append((pool.submit(read_record, reader, keys[index], shape), reader))

    for reader in readers:
        submit(reader)
    with output.open(f"{name}.npy", "w", force_zip64=True) as destination:
        np.lib.format.write_array_header_2_0(destination, {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False,
            "shape": (len(indices), shape[1], shape[0]),
        })
        while pending:
            future, reader = pending.popleft()
            values = future.result()
            destination.write(memoryview(values).cast("B"))
            del values, future
            progress.update(1)
            submit(reader)


def convert_dataset(source, destination, workers=4):
    source, destination = Path(source), Path(destination)
    if workers < 1:
        raise ValueError("workers must be positive")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    with np.load(source, allow_pickle=False) as archive:
        keys = archive["record_keys"].tolist()
        raw_labels = archive["labels"]
    if not keys or raw_labels.ndim != 1 or len(keys) != len(raw_labels):
        raise ValueError("Expected nonempty recordings with one label per record")
    with ZipFile(source) as archive:
        shape = None
        for key in tqdm(keys, desc="Checking headers", unit="record"):
            current, _, dtype = recording_header(archive, key)
            if len(current) != 2 or not all(current) or dtype.kind not in "biuf":
                raise ValueError(f"Expected a real numeric (time, features) array: {key}")
            if shape is not None and current != shape:
                raise ValueError("Equal recording lengths are required; no trimming is applied")
            shape = current
    classes, labels = np.unique(raw_labels, return_inverse=True)
    labels = labels.astype(np.int64)
    train, test = train_test_split(
        np.arange(len(keys)), test_size=0.15, random_state=42, stratify=labels,
    )
    expected_bytes = len(keys) * shape[0] * shape[1] * 4
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = disk_usage(destination.parent).free
    reserve = 5 * 1024**3
    if free < expected_bytes + reserve:
        raise OSError(f"Need {expected_bytes / 1e9:.2f} GB plus 5 GiB free; "
                      f"only {free / 1e9:.2f} GB available")
    print(f"Output: {destination}\n"
          f"X_train: {(len(train), shape[1], shape[0])}, float32\n"
          f"X_test: {(len(test), shape[1], shape[0])}, float32\n"
          f"Classes: {len(classes)}; output: {expected_bytes / 1e9:.2f} GB; "
          f"readers: {min(workers, len(keys))}", flush=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    # Exclusive creation protects any earlier partial conversion from replacement.
    with temporary.open("xb") as target:
        try:
            with ExitStack() as stack:
                readers = [stack.enter_context(np.load(source, allow_pickle=False))
                           for _ in range(min(workers, len(keys)))]
                output = stack.enter_context(ZipFile(target, "w", compression=ZIP_STORED))
                pool = stack.enter_context(ThreadPoolExecutor(max_workers=len(readers)))
                progress = stack.enter_context(tqdm(total=len(keys), desc="Writing recordings",
                                                   unit="record", mininterval=10))
                write_features(output, "X_train", train, readers, keys, shape, pool, progress)
                write_features(output, "X_test", test, readers, keys, shape, pool, progress)
                for name, values in (("y_train", labels[train]), ("y_test", labels[test]),
                                     ("classes", classes)):
                    with output.open(f"{name}.npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, values, allow_pickle=False)
        except BaseException:
            # Preserve a partial output for inspection; never touch the raw source.
            print(f"Conversion failed; incomplete output: {temporary}", flush=True)
            raise
    temporary.rename(destination)
    print(f"Saved {destination} ({destination.stat().st_size / 1e9:.2f} GB)", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="resnet_features")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "data"
    name = args.dataset.removeprefix("openworm_")
    convert_dataset(root / "OpenWorm" / f"openworm_{name}_raw.npz",
                    root / f"openworm_{name}.npz", args.workers)


if __name__ == "__main__":
    main()
