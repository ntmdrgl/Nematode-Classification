"""Trim OpenWorm recordings to the shortest sample using parallel NPZ readers."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from tqdm import tqdm


def extract_dataset(path, output, workers=16):
    if workers < 16:
        raise ValueError("Use at least 16 workers")

    with np.load(path, allow_pickle=False) as archive:
        keys = archive["record_keys"].tolist()
        y = archive["labels"].copy()
    if not keys or len(keys) != len(y):
        raise ValueError("Expected a nonempty record list with one label per record")

    # Inspect headers without loading all full-length recordings into memory.
    shapes, dtypes = [], []
    with ZipFile(path) as archive:
        for key in tqdm(keys, desc="Reading recording lengths", unit="record"):
            with archive.open(f"{key}.npy") as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError(f"Unsupported NPY version {version}: {key}")
            if (len(shape) != 2 or not all(shape)
                    or not np.issubdtype(dtype, np.number)):
                raise ValueError(f"Expected a nonempty numeric (time, features) array: {key}")
            if shapes and shape[1] != shapes[0][1]:
                raise ValueError(f"Inconsistent feature count: {key}")
            shapes.append(shape)
            dtypes.append(dtype)

    min_timesteps = min(shape[0] for shape in shapes)
    print("Samples:", len(keys))
    print("Shortest recording:", min_timesteps)
    print("Longest recording:", max(shape[0] for shape in shapes))

    # Allocate only the trimmed output, keeping records aligned with their labels.
    X = np.empty((len(keys), shapes[0][1], min_timesteps),
                 dtype=np.result_type(*dtypes))
    with tqdm(total=len(keys), desc="Loading and trimming", unit="record") as progress:
        def load_chunk(indices):
            # Separate archive handles allow workers to read/decompress concurrently.
            with np.load(path, allow_pickle=False) as archive:
                for i in indices:
                    values = archive[keys[i]]
                    X[i] = values[:min_timesteps].T
                    del values
                    progress.update(1)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(load_chunk, range(i, len(keys), workers))
                       for i in range(min(workers, len(keys)))]
            for future in as_completed(futures):
                future.result()

    print("Features:", X.shape)  # (samples, features, min_timesteps)
    print("Labels:", y.shape)

    # print(f"Saving compressed dataset to {output} (this may take a while)...", flush=True)
    # np.savez_compressed(output, X=X, y=y)
    
    print(f"Saving uncompressed dataset to {output} (this may take a while)...", flush=True)
    np.savez(output, X=X, y=y)
    
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default="openworm_skeleton_angles")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    input = Path(f"data/OpenWorm/{args.dataset}_raw.npz")
    output = Path(f"data/{args.dataset}_trimmed.npz")

    extract_dataset(input, output, args.workers)

if __name__ == "__main__":
    main()
