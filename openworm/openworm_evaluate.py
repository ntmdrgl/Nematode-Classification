"""Evaluate real/interpolated/zeroed coverage by strain and total dataset.

Run from the repository root:
    python openworm/openworm_evaluate.py
    python openworm/openworm_evaluate.py --data-dir C:/path/to/data
    python openworm/openworm_evaluate.py --data-dir D:/data --workers 4
    python openworm/openworm_evaluate.py --data-dir D:/data --workers 4 --totals-only
    python openworm/openworm_evaluate.py --data-dir D:/data --dataset skeleton_angles --workers 4
    python openworm/openworm_evaluate.py --output data/openworm_evaluation.csv

Uses each NPZ archive's record list and feature names, and reads provenance from
the matching skeleton_angles/manual_features/matlab_features CSVs (NPZs have no status flags).
Checks source feature values against the archive to reject stale/mismatched CSVs.

Percentages count feature values, not averages of record percentages. Every
time point contributes one value per feature: 48 for angles, 47 for manual
features, 6 for eigenworms, and 16 for MATLAB features. All 48 angles share skeleton_status; manual and
eigenworm features use their individual <feature>_status columns. Measured zeros
remain real; numeric zero is never used to infer status. Longer records receive
proportionally more weight. Each dataset has a row per strain plus a total row.
Total rows come first. Each dataset uses its own available record set.
Missing NPZ datasets are skipped. A total progress bar and ETA cover all records
in the available datasets, including source CSV checks. Missing source CSVs or
invalid existing archives still raise an error. If no datasets exist, no output
is written.
Percentage columns are grouped together before
the count columns. Output defaults to data/openworm_evaluation.csv.
The NPZs/source CSVs are not modified.
--workers defaults to 1. Higher values evaluate records in separate processes;
each process opens its own archive, and only the parent updates progress and
combines counts. At most one record per worker is queued.
--totals-only saves only the dataset total rows, omitting per-strain rows.
All records are still evaluated to calculate the totals.
--dataset defaults to all; select skeleton_angles, manual_features, eigenworms,
or matlab_features to evaluate only that dataset. Each run replaces the output
CSV with the selected results.
"""

import argparse
import csv
import multiprocessing
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("skeleton_angles", "openworm_skeleton_angles_dataset.npz", "skeleton_angles", 48),
    ("manual_features", "openworm_manual_features_dataset.npz", "manual_features", 47),
    ("eigenworms", "openworm_eigenworms_dataset.npz", "manual_features", 6),
    ("matlab_features", "openworm_matlab_features_dataset.npz", "matlab_features", 16),
)
STATES = ("real", "interpolated", "zeroed")


def record_status_counts(path, feature_names, values, skeleton=False):
    counts = Counter()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        columns = next(reader)
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate CSV columns: {path}")
        indices = [columns.index(name) for name in feature_names]
        status_indices = ([columns.index("skeleton_status")] * len(feature_names) if skeleton
                          else [columns.index(name + "_status") for name in feature_names])
        rows = 0
        for row in reader:
            if len(row) != len(columns) or rows >= len(values):
                raise ValueError(f"CSV row count/width differs from archive: {path}")
            if not np.array_equal(np.array([row[i] for i in indices], dtype=float), values[rows]):
                raise ValueError(f"CSV values differ from NPZ at row {rows + 1}: {path}; rebuild the datasets")
            flags = [row[i] for i in status_indices]
            if any(flag not in STATES for flag in flags):
                raise ValueError(f"Unknown status at row {rows + 1}: {path}")
            counts.update(flags)
            rows += 1
        if rows != len(values):
            raise ValueError(f"CSV row count differs from archive: {path}")
    return counts


def summary(dataset, scope, strain, records, frames, features, counts):
    total = sum(counts.values())
    if total != frames * features or not total:
        raise ValueError("Status counts do not match the number of feature values")
    row = dict(dataset=dataset, scope=scope, strain=strain)
    for state in STATES:
        row[f"{state}_percent"] = f"{100 * counts[state] / total:.6f}"
    for state in STATES:
        row[f"{state}_count"] = counts[state]
    row.update(records=records, frames=frames, features=features, total_values=total)
    return row


def evaluate_record(archive, key, path, names, feature_count, skeleton):
    values = archive[key]
    if values.ndim != 2 or values.shape[1] != feature_count or not np.isfinite(values).all():
        raise ValueError(f"Invalid record data: {archive.zip.filename}/{key}")
    return len(values), record_status_counts(path, names, values, skeleton=skeleton)


def initialize_worker(archive_path):
    global worker_archive
    worker_archive = np.load(archive_path, allow_pickle=False)


def evaluate_worker(key, path, names, feature_count, skeleton):
    return evaluate_record(worker_archive, key, path, names, feature_count, skeleton)


def parallel_results(pool, jobs, workers):
    jobs = iter(jobs)
    pending = deque()
    try:
        for _ in range(workers):
            job = next(jobs, None)
            if job is None:
                break
            pending.append(pool.submit(evaluate_worker, *job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(evaluate_worker, *job))
    finally:
        for future in pending:
            future.cancel()


def record_jobs(data_dir, folder, labels, ids, keys, names, feature_count, skeleton):
    seen = set()
    for strain, record_id, key in zip(labels, ids, keys):
        strain, record_id = str(strain), str(record_id)
        if (strain, record_id) in seen:
            raise ValueError(f"Duplicate strain/record: {strain}/{record_id}")
        seen.add((strain, record_id))
        path = (data_dir / folder / strain / f"{record_id}.csv").resolve()
        if not path.is_relative_to((data_dir / folder).resolve()):
            raise ValueError("Source CSV path is outside its data folder")
        yield key, path, names, feature_count, skeleton


def evaluate(data_dir, workers=1, totals_only=False, dataset="all"):
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if dataset not in ("all", *(specification[0] for specification in DATASETS)):
        raise ValueError(f"Unknown dataset: {dataset}")
    available = []
    total = 0
    for specification in DATASETS:
        if dataset != "all" and specification[0] != dataset:
            continue
        path = data_dir / specification[1]
        if not path.exists():
            tqdm.write(f"Skipping missing dataset: {path.name}")
            continue
        with np.load(path, allow_pickle=False) as archive:
            total += len(archive["record_keys"])
        available.append(specification)
    if not available:
        raise ValueError(f"No dataset archives found in {data_dir}; evaluation output was not changed")
    with tqdm(total=total, desc="Evaluation records", unit="record", smoothing=0,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, TOTAL ETA {remaining}, {rate_fmt}{postfix}]") as progress:
        rows = evaluate_available(data_dir, available, progress, workers)
        return [row for row in rows if row["scope"] == "total"] if totals_only else rows


def evaluate_available(data_dir, available, progress, workers=1):
    results = []
    for dataset, filename, folder, feature_count in available:
        by_strain = {}
        with ExitStack() as stack:
            archive = stack.enter_context(np.load(data_dir / filename, allow_pickle=False))
            labels, ids, keys, names = (archive[key] for key in ("labels", "record_ids", "record_keys", "feature_names"))
            if (any(array.ndim != 1 for array in (labels, ids, keys, names))
                    or not len(labels) or not len(labels) == len(ids) == len(keys)
                    or len(names) != feature_count or len(set(names)) != feature_count or len(set(keys)) != len(keys)):
                raise ValueError(f"Invalid archive metadata: {filename}")
            jobs = record_jobs(data_dir, folder, labels, ids, keys, names, feature_count,
                               dataset == "skeleton_angles")
            if workers == 1:
                completed = (evaluate_record(archive, *job) for job in jobs)
            else:
                pool = stack.enter_context(ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_worker, initargs=(data_dir / filename,)))
                completed = parallel_results(pool, jobs, workers)
            for strain, record_id, (frames, counts) in zip(labels, ids, completed):
                strain, record_id = str(strain), str(record_id)
                group = by_strain.setdefault(strain, dict(records=0, frames=0, counts=Counter()))
                group["records"] += 1
                group["frames"] += frames
                group["counts"].update(counts)
                progress.set_postfix_str(f"{dataset}: {strain}/{record_id}", refresh=False)
                progress.update(1)
        total_counts = Counter()
        for strain, group in sorted(by_strain.items()):
            results.append(summary(dataset, "strain", strain, group["records"], group["frames"], feature_count, group["counts"]))
            total_counts.update(group["counts"])
        results.append(summary(dataset, "total", "", sum(group["records"] for group in by_strain.values()),
                               sum(group["frames"] for group in by_strain.values()), feature_count, total_counts))
    return sorted(results, key=lambda row: row["scope"] != "total")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel record evaluation processes (default: 1)")
    parser.add_argument("--totals-only", action="store_true",
                        help="Save only dataset total rows; omit per-strain rows")
    parser.add_argument("--dataset", choices=("all", *(item[0] for item in DATASETS)),
                        default="all", help="Evaluate all datasets or just one (default: all)")
    args = parser.parse_args()
    try:
        rows = evaluate(args.data_dir.resolve(), args.workers, args.totals_only, args.dataset)
        output = args.output or args.data_dir / "openworm_evaluation.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {output.resolve()}")
        for row in rows:
            if row["scope"] == "total":
                print(f"{row['dataset']}: real {row['real_percent']}%, "
                      f"interpolated {row['interpolated_percent']}%, zeroed {row['zeroed_percent']}%")
    except (OSError, ValueError, KeyError, StopIteration, IndexError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
