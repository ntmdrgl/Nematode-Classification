"""Verify every split recording against raw data and the seed-42 split.

Checks actual feature values, not just class labels. Writes a small provenance
manifest only after every recording in every dataset has matched.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.openworm_io import map_array


def verify_splits(data_dir, workers=8, dataset=None):
    data_dir = Path(data_dir)
    reference = None
    reports = []
    files = sorted(data_dir.glob("openworm_*.npz"))
    if dataset:
        files = [p for p in files if p.stem.removesuffix("_split").removeprefix("openworm_")
                 == dataset.removeprefix("openworm_")]
    if not files:
        raise ValueError("No split datasets found")
    for path in files:
        name = path.stem.removesuffix("_split")
        source = data_dir / "OpenWorm_raw" / f"{name}_raw.npz"
        with np.load(source, allow_pickle=False) as raw, np.load(path, allow_pickle=False) as split:
            ids, labels, keys = raw["record_ids"], raw["labels"], raw["record_keys"]
            # (strain, record ID) identifies a recording even if IDs repeat by strain.
            identity = np.stack((labels.astype(str), ids.astype(str)), axis=1)
            if len(np.unique(identity, axis=0)) != len(identity):
                raise ValueError(f"Nonunique recording identities: {source}")
            if reference is None:
                reference = identity
            np.testing.assert_array_equal(identity, reference, err_msg=f"Raw order differs: {name}")
            classes, encoded = np.unique(labels, return_inverse=True)
            train, test = train_test_split(np.arange(len(keys)), test_size=.15,
                                           random_state=42, stratify=encoded)
            np.testing.assert_array_equal(split["classes"], classes)
            for part, indices in (("train", train), ("test", test)):
                np.testing.assert_array_equal(split[f"y_{part}"], encoded[indices])
                values = map_array(path, f"X_{part}")
                if len(values) != len(indices):
                    raise ValueError(f"Sample count differs: {name}/{part}")
                def check_worker(worker):
                    with np.load(source, allow_pickle=False) as reader:
                        for position in range(worker, len(indices), workers):
                            original = indices[position]
                            expected = reader[str(keys[original])].T.astype(values.dtype, copy=False)
                            np.testing.assert_array_equal(values[position], expected,
                                                           err_msg=f"{name}/{part}/{position}")
                            progress.update(1)
                with tqdm(total=len(indices), desc=f"{name} {part}", unit="record",
                          mininterval=10) as progress, ThreadPoolExecutor(max_workers=workers) as pool:
                    for result in pool.map(check_worker, range(workers)):
                        pass
                del values
            reports.append({"dataset": name, "source_size": source.stat().st_size,
                            "split_size": path.stat().st_size,
                            "split_mtime_ns": path.stat().st_mtime_ns,
                            "train_count": len(train), "test_count": len(test),
                            "train_record_ids": ids[train].tolist(),
                            "test_record_ids": ids[test].tolist()})
        print(f"VERIFIED every recording: {name}", flush=True)
    output = files[0].parent / (f"{dataset}_split_verification.json" if dataset
                                else "split_verification.json")
    output.write_text(json.dumps({"seed": 42, "test_fraction": .15,
                                  "verification": "all feature values compared with raw",
                                  "datasets": reports}, indent=2), encoding="utf-8")
    print(f"Verified all {len(files)} datasets; saved {output}", flush=True)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    verify_splits(args.data_dir, args.workers, args.dataset)
