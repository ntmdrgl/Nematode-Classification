"""Evaluate real/interpolated/zeroed coverage by strain and total dataset.

Run from the repository root:
    python openworm/openworm_evaluate.py
    python openworm/openworm_evaluate.py --data-dir C:/path/to/data
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
The four total rows come first. Each dataset uses its own available record set.
Percentage columns are grouped together before
the count columns. Output defaults to data/openworm_evaluation.csv.
The NPZs/source CSVs are not modified.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

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


def evaluate(data_dir):
    results = []
    for dataset, filename, folder, feature_count in DATASETS:
        by_strain = {}
        seen = set()
        with np.load(data_dir / filename, allow_pickle=False) as archive:
            labels, ids, keys, names = (archive[key] for key in ("labels", "record_ids", "record_keys", "feature_names"))
            if (any(array.ndim != 1 for array in (labels, ids, keys, names))
                    or not len(labels) or not len(labels) == len(ids) == len(keys)
                    or len(names) != feature_count or len(set(names)) != feature_count or len(set(keys)) != len(keys)):
                raise ValueError(f"Invalid archive metadata: {filename}")
            for strain, record_id, key in zip(labels, ids, keys):
                strain, record_id = str(strain), str(record_id)
                if (strain, record_id) in seen:
                    raise ValueError(f"Duplicate strain/record in {filename}: {strain}/{record_id}")
                seen.add((strain, record_id))
                path = (data_dir / folder / strain / f"{record_id}.csv").resolve()
                if not path.is_relative_to((data_dir / folder).resolve()):
                    raise ValueError("Source CSV path is outside its data folder")
                values = archive[key]
                if values.ndim != 2 or values.shape[1] != feature_count or not np.isfinite(values).all():
                    raise ValueError(f"Invalid record data: {filename}/{key}")
                counts = record_status_counts(path, names, values, skeleton=dataset == "skeleton_angles")
                group = by_strain.setdefault(strain, dict(records=0, frames=0, counts=Counter()))
                group["records"] += 1
                group["frames"] += len(values)
                group["counts"].update(counts)
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
    args = parser.parse_args()
    try:
        rows = evaluate(args.data_dir.resolve())
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
