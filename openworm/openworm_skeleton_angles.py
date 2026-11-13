"""Preprocess skeletons and save 48 mean-centered segment angles in radians.

From the repository root:
    python openworm/openworm_skeleton_angles.py
    python openworm/openworm_skeleton_angles.py --workers 4
    python openworm/openworm_skeleton_angles.py --input-dir data/openworm/test
    python openworm/openworm_skeleton_angles.py --input-dir E:/openworm

Input: <input-dir>/<strain>/record_*/*_features.hdf5 (or a single strain folder).
Output: data/skeleton_angles/<strain>/record_*.csv, one row every 0.04 seconds.
Columns: time_s, skeleton_status, angle_01 ... angle_48.
Status: real = observed skeleton or routine resampling between consecutive valid
source frames; interpolated = filling a missing-skeleton gap; zeroed = unusable.

Each finite, nondegenerate skeleton is resampled to 49 equal arc-length points.
Temporal interpolation uses the closest valid frames only when their separation
is strictly less than 0.25 s. Unbracketed ends and larger gaps stay zero.
Savitzky-Golay smoothing is applied to coordinates in space and then time, within
each valid run, so zero gaps cannot contaminate neighboring skeletons.
Defaults: 5-point windows, polynomial order 3 (not specified in the quotation).
Short runs use the largest fitting odd window and a fitting polynomial order.

Angles follow the supplied formula literally: arctan(dx/dy), then subtract the
mean of the 48 angles in each frame. The result is not an atan2/unwrapped angle.
Zero-length segments after smoothing invalidate that frame. Reruns replace CSVs.
--workers processes independent records in separate CPU processes. The parent
shows one total ETA; per-record stage bars are disabled during parallel runs.
"""

import argparse
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
PROGRESS_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, ETA {remaining}, {rate_fmt}]{postfix}"


def read_record(path):
    with h5py.File(path, "r") as source:
        skeletons = source["coordinates/skeletons"][:].astype(float)
        table = source["features_timeseries"]
        frames = table["timestamp"].astype(float)
        fps = float(table.attrs.get("fps", np.nan))
        worms = np.unique(table["worm_index"])
        if len(worms[np.isfinite(worms) & (worms > 0)]) > 1:
            raise ValueError("Multiple worms in one record; expected a single-worm recording")
    if skeletons.ndim != 3 or skeletons.shape[1:] != (49, 2) or len(frames) != len(skeletons):
        raise ValueError("Expected aligned timestamps and an (n, 49, 2) skeleton array")
    video = path.with_name(path.name.replace("_features.hdf5", ".hdf5"))
    times = None
    if video.exists():
        with h5py.File(video, "r") as source:
            if "timestamp/raw" in source and "timestamp/time" in source:
                raw, seconds = source["timestamp/raw"][:], source["timestamp/time"][:]
                if (len(raw) == len(seconds) and len(raw) >= 2 and np.isfinite(raw).all()
                        and np.isfinite(seconds).all() and np.all(np.diff(raw) > 0)
                        and np.all(np.diff(seconds) > 0)):
                    if frames.min() < raw[0] or frames.max() > raw[-1]:
                        raise ValueError("Skeleton frame indices extend outside the video timestamps")
                    times = np.interp(frames, raw, seconds)
    if times is None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("No usable video timestamps or frame-rate metadata")
        times = frames / fps
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Skeleton times must be finite and strictly increasing")
    return times - times[0], skeletons


def spatial_resample(skeletons):
    result = np.full_like(skeletons, np.nan, dtype=float)
    for i in np.flatnonzero(np.isfinite(skeletons).all(axis=(1, 2))):
        points = skeletons[i]
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        keep = np.r_[True, np.diff(distance) > 0]
        if distance[-1] <= 0:
            continue
        grid = np.linspace(0, distance[-1], 49)
        for coordinate in (0, 1):
            result[i, :, coordinate] = np.interp(grid, distance[keep], points[keep, coordinate])
    return result


def temporal_resample(times, values):
    """Shared 25 Hz resampling and strict <0.25-second missing-data rule."""
    grid = np.arange(int(np.floor(times[-1] / 0.04 + 1e-9)) + 1) * 0.04
    good = np.isfinite(values.reshape(len(values), -1)).all(axis=1)
    coordinates = np.zeros((len(grid), *values.shape[1:]))
    valid = np.zeros(len(grid), dtype=bool)
    status = np.full(len(grid), "zeroed", dtype="U12")
    if good.any():
        t = times[good]
        right = np.searchsorted(t, grid, side="left")
        nearest = np.minimum(right, len(t) - 1)
        exact = np.isclose(grid, t[nearest], rtol=0, atol=1e-10)
        bracketed = (right > 0) & (right < len(t))
        left = np.maximum(right - 1, 0)
        # The gap is the elapsed time between its closest valid endpoints.
        valid = exact | (bracketed & (t[nearest] - t[left] < 0.25 - 1e-10))
        source_indices = np.flatnonzero(good)
        filled_gap = valid & ~exact & (source_indices[nearest] - source_indices[left] > 1)
        status[valid] = "real"
        status[filled_gap] = "interpolated"
        if len(t) == 1:
            coordinates[exact] = values[good][0]
        else:
            interpolate = interp1d(t, values[good], axis=0, bounds_error=False, fill_value=0)
            coordinates[valid] = interpolate(grid[valid])
    return grid, coordinates, valid, status


def smooth_time(values, valid, time_window=5, polyorder=3):
    """Smooth valid runs independently, preserving zero gaps."""
    result = values.copy()
    edges = np.diff(np.r_[False, valid, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        window = min(time_window, (end-start) if (end-start) % 2 else (end-start)-1)
        result[start:end] = savgol_filter(values[start:end], window, min(polyorder, window-1), axis=0)
    result[~valid] = 0
    return result


def preprocess(times, skeletons, space_window=5, time_window=5, polyorder=3):
    skeletons = spatial_resample(skeletons)
    grid, coordinates, valid, status = temporal_resample(times, skeletons)

    edges = np.diff(np.r_[False, valid, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        coordinates[start:end] = savgol_filter(coordinates[start:end], space_window, polyorder, axis=1)
    coordinates = smooth_time(coordinates, valid, time_window, polyorder)

    differences = np.diff(coordinates, axis=1)
    valid &= np.all(np.linalg.norm(differences, axis=2) > 1e-12, axis=1)
    angles = np.zeros((len(grid), 48))
    with np.errstate(divide="ignore", invalid="ignore"):
        angles[valid] = np.arctan(differences[valid, :, 0] / differences[valid, :, 1])
    angles[valid] -= angles[valid].mean(axis=1, keepdims=True)
    status[~valid] = "zeroed"
    return grid, angles, valid, status


def process_record(path, output_dir, space_window=5, time_window=5, polyorder=3, show_progress=True):
    if not path.parent.name.startswith("record_") or len(list(path.parent.glob("*_features.hdf5"))) != 1:
        raise ValueError("Expected one features file inside each <strain>/record_* folder")
    times, skeletons = read_record(path)
    times, angles, valid, status = preprocess(times, skeletons, space_window, time_window, polyorder)
    output = output_dir / path.parent.parent.name / f"{path.parent.name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["time_s", "skeleton_status", *[f"angle_{i:02d}" for i in range(1, 49)]])
        for t, state, row in tqdm(zip(times, status, angles), total=len(times),
                                  desc="Writing angles", unit="frame", position=1, leave=False,
                                  disable=not show_progress, bar_format=PROGRESS_FORMAT):
            writer.writerow([f"{t:.8f}", state, *[f"{angle:.8f}" for angle in row]])
    return f"{output}: {len(times):,} frames; {valid.sum():,} valid; {(~valid).sum():,} zero gap frames"


def run_records(files, worker, arguments, workers, description):
    """Bounded process pool; only the parent writes overall progress/ETA."""
    failures = 0
    with tqdm(total=len(files), desc=description, unit="record", smoothing=0,
              bar_format=PROGRESS_FORMAT.replace("ETA {remaining}", "TOTAL ETA {remaining}")) as progress:
        def completed(path, operation):
            nonlocal failures
            try:
                tqdm.write(operation())
            except Exception as error:
                failures += 1
                tqdm.write(f"FAILED {path}: {error}")
            progress.set_postfix_str(f"{path.parent.parent.name}/{path.parent.name}")
            progress.update(1)

        if workers == 1:
            for path in files:
                completed(path, lambda: worker(path, *arguments, show_progress=True))
        else:
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            pending = {}
            paths = iter(files)

            def submit_next():
                path = next(paths, None)
                if path is not None:
                    pending[pool.submit(worker, path, *arguments, show_progress=False)] = path
            try:
                for _ in range(min(workers, len(files))):
                    submit_next()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        path = pending.pop(future)
                        completed(path, future.result)
                        submit_next()
            finally:
                for future in pending:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/openworm")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/skeleton_angles")
    parser.add_argument("--space-window", type=int, default=5)
    parser.add_argument("--time-window", type=int, default=5)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent record processes (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if (args.space_window < 1 or args.space_window > 49 or args.space_window % 2 == 0
            or args.time_window < 1 or args.time_window % 2 == 0 or args.polyorder < 0
            or args.polyorder >= min(args.space_window, args.time_window)):
        parser.error("Windows must be positive and odd (space <=49); order must be smaller than both")
    files = sorted(args.input_dir.rglob("*_features.hdf5"))
    if not files:
        parser.error("No *_features.hdf5 files found")
    return run_records(files, process_record, (args.output_dir, args.space_window, args.time_window, args.polyorder), args.workers, "All skeleton records")


if __name__ == "__main__":
    raise SystemExit(main())
