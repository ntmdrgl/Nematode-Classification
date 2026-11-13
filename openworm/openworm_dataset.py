"""Combine per-record CSVs into four labeled NPZ datasets (no pickle needed).

Run from the repository root:
    python openworm/openworm_dataset.py
    python openworm/openworm_dataset.py --data-dir C:/path/to/data

Reads skeleton_angles/<strain>/record_*.csv and manual_features/<strain>/record_*.csv.
Writes openworm_skeleton_angles_dataset.npz (48 angles),
openworm_manual_features_dataset.npz (47 features), and
openworm_eigenworms_dataset.npz (6 eigen projections), and
openworm_matlab_features_dataset.npz (16 MATLAB features) in the data directory.
MATLAB records are read from matlab_features/<strain>/record_*.csv and may be
a smaller set, since records without videos were skipped during extraction.
49 skeleton points define 48 segment angles; no artificial 49th angle is added.

Each archive contains feature_names, labels, record_ids, and record_keys.
For record i, archive[record_keys[i]] is a (time, features) float64 array;
labels[i] is its strain name. Variable-length records are stored separately,
without padding, time/status columns, or object arrays. Read with:
    with numpy.load('data/openworm_manual_features_dataset.npz') as dataset:
        values = dataset[dataset['record_keys'][0]]
        strain = dataset['labels'][0]

Record sets and time grids must match between the two CSV directories. Records
are compressed one at a time to limit memory use. Reruns replace the archives
only after all records pass validation.
"""

import argparse
import csv
import os
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANGLES = [f"angle_{i:02d}" for i in range(1, 49)]
EIGEN = [f"eigen_projection_{i}" for i in range(1, 7)]
OUTPUTS = ("openworm_skeleton_angles_dataset.npz", "openworm_manual_features_dataset.npz",
           "openworm_eigenworms_dataset.npz", "openworm_matlab_features_dataset.npz")


def record_files(directory):
    return {(path.parent.name, path.stem): path for path in directory.glob("*/record_*.csv")}


def header(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        names = next(csv.reader(source))
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate column names: {path}")
    return names


def read_values(path, names):
    columns = header(path)
    values = np.loadtxt(path, delimiter=",", skiprows=1,
                        usecols=[columns.index(name) for name in ["time_s", *names]], ndmin=2)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Empty or nonfinite data: {path}")
    if np.any(np.diff(values[:, 0]) <= 0) or not np.allclose(np.diff(values[:, 0]), .04, rtol=0, atol=1e-8):
        raise ValueError(f"Expected strictly increasing 0.04-second samples: {path}")
    return values[:, 0], values[:, 1:]


def write_array(archive, key, values):
    # A ZIP of named NPY arrays is the standard NPZ format understood by np.load.
    with archive.open(key + ".npy", "w", force_zip64=True) as destination:
        np.lib.format.write_array(destination, np.asarray(values), allow_pickle=False)


def write_metadata(archive, names, records):
    write_array(archive, "feature_names", names)
    write_array(archive, "labels", [strain for strain, _ in records])
    write_array(archive, "record_ids", [record for _, record in records])
    write_array(archive, "record_keys", [f"data_{i:06d}" for i in range(len(records))])


def build_datasets(data_dir):
    skeletons = record_files(data_dir / "skeleton_angles")
    features = record_files(data_dir / "manual_features")
    if not skeletons or skeletons.keys() != features.keys():
        raise ValueError("Both CSV folders must contain the same nonempty set of strain/record pairs. "
                         f"Missing manual: {sorted(skeletons.keys()-features.keys())}; "
                         f"missing skeleton: {sorted(features.keys()-skeletons.keys())}")
    records = sorted(skeletons)
    first_header = header(features[records[0]])
    all_features = [name for name in first_header if name != "time_s" and not name.endswith("_status")]
    manual_names = [name for name in all_features if name not in EIGEN]
    if len(manual_names) != 47 or len(all_features) != 53 or not set(EIGEN).issubset(all_features):
        raise ValueError("Expected 47 manual features and eigen_projection_1 through eigen_projection_6")
    matlab_files = record_files(data_dir / "matlab_features")
    if not matlab_files:
        raise ValueError("No MATLAB feature CSVs found in matlab_features/<strain>/record_*.csv")
    matlab_records = sorted(matlab_files)
    matlab_names = [name for name in header(matlab_files[matlab_records[0]])
                    if name != "time_s" and not name.endswith("_status")]
    if len(matlab_names) != 16:
        raise ValueError("Expected 16 MATLAB feature columns")
    paths = [data_dir / name for name in OUTPUTS]
    temporary = [path.with_suffix(".npz.part") for path in paths]
    try:
        with ExitStack() as stack:
            archives = [stack.enter_context(ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=6))
                        for path in temporary]
            for archive, names in zip(archives, (ANGLES, manual_names, EIGEN)):
                write_metadata(archive, names, records)
            write_metadata(archives[3], matlab_names, matlab_records)
            for i, record in enumerate(records):
                angle_names = [name for name in header(skeletons[record]) if name.startswith("angle_")]
                current_names = [name for name in header(features[record]) if name != "time_s" and not name.endswith("_status")]
                if set(angle_names) != set(ANGLES) or set(current_names) != set(all_features):
                    raise ValueError(f"Inconsistent feature columns in {record}")
                angle_time, angle_data = read_values(skeletons[record], ANGLES)
                feature_time, feature_data = read_values(features[record], all_features)
                if angle_time.shape != feature_time.shape or not np.allclose(angle_time, feature_time, rtol=0, atol=1e-8):
                    raise ValueError(f"Skeleton/feature time grids differ in {record}")
                arrays = (angle_data, feature_data[:, [all_features.index(name) for name in manual_names]],
                          feature_data[:, [all_features.index(name) for name in EIGEN]])
                for archive, values in zip(archives, arrays):
                    write_array(archive, f"data_{i:06d}", values)
                print(f"[{i+1}/{len(records)}] {record[0]}/{record[1]}: {len(angle_time):,} frames", flush=True)
            for i, record in enumerate(matlab_records):
                names = [name for name in header(matlab_files[record]) if name != "time_s" and not name.endswith("_status")]
                if set(names) != set(matlab_names):
                    raise ValueError(f"Inconsistent MATLAB feature columns in {record}")
                times, values = read_values(matlab_files[record], matlab_names)
                write_array(archives[3], f"data_{i:06d}", values)
                print(f"[MATLAB {i+1}/{len(matlab_records)}] {record[0]}/{record[1]}: {len(times):,} frames", flush=True)
        for source, destination in zip(temporary, paths):
            os.replace(source, destination)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    for path in paths:
        print(f"Saved {path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    try:
        build_datasets(args.data_dir.resolve())
    except (OSError, ValueError, StopIteration) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
