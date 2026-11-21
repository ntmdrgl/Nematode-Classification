"""Keep records longer than 890 s and trim to the shortest retained record.

Run from the repository root:
    python openworm/openworm_trim.py
    python openworm/openworm_trim.py --data-dir D:/data
    python openworm/openworm_trim.py --workers 4
    python openworm/openworm_trim.py --dataset eigenworms --workers 4
    python openworm/openworm_trim.py --output-dir D:/data/trimmed

Reads the skeleton_angles, manual_features and eigenworms dataset NPZs from
D:/data by default. Writes openworm_<name>_trimmed_890_dataset.npz alongside
the inputs (or in --output-dir). Original datasets are not modified.

Selection is strictly >890 seconds: more than 22,250 samples at 25 Hz.
All retained records are trimmed to the shortest retained record's sample count.
Keeps samples from the start, including any interpolated/zeroed values;
does not resample, smooth, or otherwise change feature values. Labels, record
IDs and feature names are preserved; record_keys are renumbered consecutively.
The three source datasets must have matching strain/record pairs and sample
counts. All three outputs therefore contain the same records in the same order.
--dataset defaults to all. Select skeleton_angles, manual_features or eigenworms
to read and replace only that dataset, without requiring the other inputs.
In individual mode, the shortest retained record in that dataset determines
the trim length; cross-dataset alignment is checked only in all mode.

--workers defaults to 1. Higher values read and trim records in parallel, with
one archive handle per worker and at most one queued record per worker. Archive
compression/writing stays in the parent process, preserving record order.
Shows a total progress bar with ETA. All outputs
are written to temporary files first and replaced after successful processing.
Allow disk space for three new datasets; compression can take substantial time.
"""

import argparse
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from tqdm import tqdm

from openworm_dataset import write_array, write_metadata, PROGRESS_FORMAT, ordered_results

DATASETS = (("skeleton_angles", 48), ("manual_features", 47), ("eigenworms", 6))
MIN_SAMPLES = 22_250
SAMPLE_SECONDS = 0.04


def inspect_dataset(path, feature_count):
    with np.load(path, allow_pickle=False) as archive:
        labels, ids, keys, names = (archive[key] for key in
                                    ("labels", "record_ids", "record_keys", "feature_names"))
    if (any(array.ndim != 1 for array in (labels, ids, keys, names))
            or not len(keys) or not len(labels) == len(ids) == len(keys)
            or len(names) != feature_count or len(set(names)) != feature_count
            or len(set(keys)) != len(keys)):
        raise ValueError(f"Invalid archive metadata: {path}")
    records = list(zip(map(str, labels), map(str, ids)))
    if len(set(records)) != len(records):
        raise ValueError(f"Duplicate strain/record pairs: {path}")
    sizes = {}
    with ZipFile(path) as archive:
        for record, key in tqdm(zip(records, keys), total=len(keys),
                                desc=f"Checking {path.stem}", unit="record"):
            with archive.open(str(key) + ".npy") as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError(f"Unsupported NPY version: {path}/{key}")
            if (len(shape) != 2 or shape[0] < 1 or shape[1] != feature_count
                    or dtype.hasobject or dtype.kind not in "fiu"):
                raise ValueError(f"Invalid record array: {path}/{key}")
            sizes[record] = shape[0]
    return names, dict(zip(records, map(str, keys))), sizes


_worker_archive = None
_worker_path = None


def read_trimmed_record(source, key, expected_samples, target_samples):
    """Keep each worker's input archive open across records."""
    global _worker_archive, _worker_path
    if _worker_path != source:
        if _worker_archive is not None:
            _worker_archive.close()
        _worker_archive = np.load(source, allow_pickle=False)
        _worker_path = source
    return trim_record(_worker_archive, source, key, expected_samples, target_samples)


def trim_record(archive, source, key, expected_samples, target_samples):
    values = archive[key]
    if len(values) != expected_samples or not np.isfinite(values).all():
        raise ValueError(f"Invalid or changed record: {source}/{key}")
    return values[:target_samples].copy()


def trim_datasets(data_dir, output_dir=None, workers=1, dataset="all"):
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if dataset not in ("all", *(name for name, _ in DATASETS)):
        raise ValueError(f"Unknown dataset: {dataset}")
    selected = [item for item in DATASETS if dataset in ("all", item[0])]
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else data_dir
    sources = [data_dir / f"openworm_{name}_dataset.npz" for name, _ in selected]
    metadata = [inspect_dataset(path, count) for path, (_, count) in zip(sources, selected)]
    sizes = metadata[0][2]
    if any(current[2] != sizes for current in metadata[1:]):
        raise ValueError("Source datasets must have identical strain/record pairs and sample counts")
    records = sorted(record for record, frames in sizes.items() if frames > MIN_SAMPLES)
    if not records:
        raise ValueError("No records are longer than 890 seconds; outputs were not changed")
    target_samples = min(sizes[record] for record in records)
    destinations = [output_dir / f"openworm_{name}_trimmed_890_dataset.npz" for name, _ in selected]
    temporary = [path.with_suffix(".npz.part") for path in destinations]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Keeping {len(records):,}/{len(sizes):,} records; trimming each to "
          f"{target_samples:,} samples ({target_samples * SAMPLE_SECONDS:.2f} seconds).", flush=True)
    try:
        with tqdm(total=len(records) * len(selected), desc="Trimming datasets", unit="record",
                  smoothing=0, bar_format=PROGRESS_FORMAT) as progress:
            for source, temp, (names, keys, _) in zip(sources, temporary, metadata):
                with np.load(source, allow_pickle=False) as archive, \
                        ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=6) as output:
                    write_metadata(output, names, records)
                    if workers == 1:
                        completed = (trim_record(archive, source, keys[record], sizes[record], target_samples)
                                     for record in records)
                    else:
                        jobs = ((source, keys[record], sizes[record], target_samples) for record in records)
                        completed = ordered_results(read_trimmed_record, jobs, workers)
                    for i, values in enumerate(completed):
                        record = records[i]
                        write_array(output, f"data_{i:06d}", values)
                        progress.set_postfix_str(f"{source.stem}: {record[0]}/{record[1]}", refresh=False)
                        progress.update(1)
        for temp, destination in zip(temporary, destinations):
            os.replace(temp, destination)
    finally:
        for temp in temporary:
            temp.unlink(missing_ok=True)
    for destination in destinations:
        print(f"Saved {destination}")
    return destinations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("D:/data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel record reading/trimming processes (default: 1)")
    parser.add_argument("--dataset", choices=("all", *(name for name, _ in DATASETS)),
                        default="all", help="Trim all datasets or just one (default: all)")
    args = parser.parse_args()
    try:
        trim_datasets(args.data_dir, args.output_dir, args.workers, args.dataset)
    except (OSError, ValueError, KeyError, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
