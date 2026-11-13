"""Resample stored Tierpsy features without smoothing, one CSV per strain/record.

Run from the repository root:
    python openworm/openworm_manual_features.py
    python openworm/openworm_manual_features.py --workers 4
    python openworm/openworm_manual_features.py --input-dir data/openworm/test
    python openworm/openworm_manual_features.py --input-dir E:/openworm

Outputs: data/manual_features/<strain>/record_*.csv. Reruns replace these CSVs.
Columns: time_s, numeric feature columns, then <feature>_status columns.
Exclude timestamps, tracking IDs, flags, and categorical motion_modes.
Keep all remaining numeric scalar features, including eigen projections.

Reuse the skeleton script's actual timestamps, 0.04s grid, strict <0.25s gap
interpolation and long-gap zeroing. No Savitzky-Golay smoothing is applied.
Process missing values separately per feature; do not interpolate across long
gaps. Features on unusable source skeletons are treated as missing.
Status is real, interpolated, or zeroed.

Spatial resampling/smoothing operates on coordinates, not feature columns with
different units. These stored features were already computed by Tierpsy; they
are not recomputed from newly spatially smoothed skeletons. Feature units are
preserved, with no mean centering. Orientation angles are unwrapped in degrees
within short-gap-connected runs, then wrapped back to [-180, 180) after interpolation.
--workers processes independent records in separate CPU processes. The parent
shows one total ETA; per-record stage bars are disabled during parallel runs.
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

try:
    from .openworm_skeleton_angles import ROOT, PROGRESS_FORMAT, read_record, spatial_resample, temporal_resample, run_records
except ImportError:
    from openworm_skeleton_angles import ROOT, PROGRESS_FORMAT, read_record, spatial_resample, temporal_resample, run_records

NON_FEATURES = {
    "timestamp", "timestamp_raw", "timestamp_time", "frame_number", "skeleton_id",
    "worm_index", "worm_index_joined", "worm_index_blob", "plate_worm_id", "int_map_id",
    "motion_modes", "has_skeleton", "is_good_skel", "was_skeletonized", "skel_outliers_flag",
    "old_trajectory_data_index",
}
ORIENTATIONS = {"head_orientation", "tail_orientation", "tail_to_head_orientation"}


def feature_names(table):
    return [name for name in table.dtype.names or ()
            if name not in NON_FEATURES and not name.endswith("_id")
            and not name.startswith("worm_index") and not table.dtype[name].shape
            and table.dtype[name].kind in "fiu"]


def preprocess_feature(times, values, source_valid, angular=False, angle_period=360):
    values = values.astype(float).copy()
    values[~source_valid] = np.nan
    if angular:
        good = np.flatnonzero(np.isfinite(values))
        # Unwrap only within runs we are allowed to connect by interpolation.
        breaks = np.flatnonzero(np.diff(times[good]) >= 0.25 - 1e-10) + 1
        for indices in np.split(good, breaks):
            values[indices] = np.unwrap(values[indices] * 2*np.pi / angle_period) * angle_period / (2*np.pi)
    grid, result, valid, status = temporal_resample(times, values)
    if angular:
        result[valid] = (result[valid] + angle_period/2) % angle_period - angle_period/2
    return grid, result, status


def process_record(path, output_dir, show_progress=True):
    if not path.parent.name.startswith("record_") or len(list(path.parent.glob("*_features.hdf5"))) != 1:
        raise ValueError("Expected one features file inside each <strain>/record_* folder")
    times, skeletons = read_record(path)
    # Same finite/nondegenerate source-skeleton check and spatial normalization.
    # Do not filter across unrelated feature columns or change their units.
    source_valid = np.isfinite(spatial_resample(skeletons)).all(axis=(1, 2))
    with h5py.File(path, "r") as source:
        table = source["features_timeseries"][:]
    names = feature_names(table)
    if not names:
        raise ValueError("No numeric feature columns found")
    features, statuses = [], []
    for name in tqdm(names, desc="Processing features", unit="feature", position=1, leave=False,
                     disable=not show_progress, bar_format=PROGRESS_FORMAT):
        grid, result, status = preprocess_feature(times, table[name], source_valid,
                                                  name in ORIENTATIONS)
        features.append(result)
        statuses.append(status)
    features, statuses = np.column_stack(features), np.column_stack(statuses)
    output = output_dir / path.parent.parent.name / f"{path.parent.name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["time_s", *names, *[f"{name}_status" for name in names]])
        for time, values, flags in tqdm(zip(grid, features, statuses), total=len(grid),
                                        desc="Writing features", unit="frame", position=1, leave=False,
                                        disable=not show_progress, bar_format=PROGRESS_FORMAT):
            writer.writerow([f"{time:.8f}", *[f"{value:.8f}" for value in values], *flags])
    return f"{output}: {len(grid):,} frames, {len(names)} features"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/openworm")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/manual_features")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent record processes (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    files = sorted(args.input_dir.rglob("*_features.hdf5"))
    if not files:
        parser.error("No *_features.hdf5 files found")
    return run_records(files, process_record, (args.output_dir,), args.workers, "All manual records")


if __name__ == "__main__":
    raise SystemExit(main())
