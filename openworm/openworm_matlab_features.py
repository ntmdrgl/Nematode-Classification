"""Extract the original 16 MATLAB features, then preprocess them in Python.

Run from the repository root (requires MATLAB + Image Processing Toolbox):
    python openworm/openworm_matlab_features.py
    python openworm/openworm_matlab_features.py --input-dir E:/openworm
    python openworm/openworm_matlab_features.py --input-dir data/openworm/test
    python openworm/openworm_matlab_features.py --max-frames 100 --output-dir data/matlab_features_preview

Default output: data/matlab_features/<strain>/record_*.csv. Outputs include time_s,
16 features selected in old/openworm_dataset_preprocessing.m, and per-feature
real/interpolated/zeroed statuses. FPS and absolute coordinates are excluded.
Records without a video .hdf5 are skipped. A matching _features.hdf5 is also
required for the original algorithm's skeleton/contour coordinates (not its
features_timeseries). Raw image calculations and their original indexing and
units are preserved; no spatial smoothing is added to MATLAB's image pipeline.

MATLAB retains the legacy formulas, including FPS-scaled total displacements
and nonnegative clipping. Original missing-value masks are passed separately
so the old unlimited fillmissing cannot hide gaps. Python restores those gaps,
inserts dropped video frames, resamples every 0.04s, fills gaps strictly shorter
than 0.25s, and zeroes longer/unbracketed gaps. No Savitzky-Golay smoothing is
applied to these features. Orientation has a 180-degree period.
The old frame trimming/downsampling/statistical summaries are not applied.
Reruns replace CSVs. --max-frames is for previews, not full dataset generation.
"""

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    from .openworm_manual_features import ROOT, preprocess_feature
except ImportError:
    from openworm_manual_features import ROOT, preprocess_feature

FEATURES = ["SkeletonLength", "EndpointSeparation", "CentroidDisplacementFS0",
            "Endpoint1DisplacementFS0", "Endpoint2DisplacementFS0", "CentroidTotalDisplacement",
            "Endpoint1TotalDisplacement", "Endpoint2TotalDisplacement", "Tortuosity",
            "NumberOfBends", "BendLengthStd", "WidthMean", "AspectRatio", "Area",
            "Perimeter", "OrientationAngle"]


def matlab_executable(explicit=None):
    if explicit:
        return str(explicit)
    executable = shutil.which("matlab")
    if executable:
        return executable
    installed = sorted(Path("C:/Program Files/MATLAB").glob("*/bin/matlab.exe"))
    if installed:
        return str(installed[-1])
    raise ValueError("MATLAB not found; pass --matlab PATH_TO_MATLAB")


def matlab_string(path):
    return "'" + Path(path).resolve().as_posix().replace("'", "''") + "'"


def preprocess_extracted(raw):
    names = [str(name) for name in np.asarray(raw["feature_names"]).reshape(-1)]
    values = np.asarray(raw["feature_values"], dtype=float).reshape(-1, 16)
    observed = np.asarray(raw["observed_mask"], dtype=bool).reshape(values.shape)
    times = np.asarray(raw["time_s"], dtype=float).reshape(-1)
    frames = np.asarray(raw["frame_number"], dtype=float).reshape(-1)
    if names != FEATURES or len(times) != len(values) or len(frames) != len(times) or not len(times):
        raise ValueError("Unexpected MATLAB feature schema or lengths")
    if (not np.isfinite(times).all() or not np.isfinite(frames).all()
            or np.any(np.diff(times) <= 0) or np.any(np.diff(frames) <= 0)
            or not np.all(frames == np.floor(frames))):
        raise ValueError("MATLAB timestamps/frame indices must be finite and strictly increasing")
    # Video rows omit dropped frames. Insert NaNs so shared gap/status logic
    # distinguishes missing frames from routine temporal resampling.
    full_frames = np.arange(int(frames[0]), int(frames[-1])+1)
    full_times = np.interp(full_frames, frames, times) - times[0]
    available = np.where(observed, values, np.nan)
    expanded = np.full((len(full_frames), 16), np.nan)
    expanded[frames.astype(int)-int(frames[0])] = available
    features, statuses = [], []
    for i, name in enumerate(FEATURES):
        grid, result, status = preprocess_feature(full_times, expanded[:,i], np.ones(len(full_times), bool),
                                                  name == "OrientationAngle", angle_period=180)
        if name != "OrientationAngle":
            result = np.maximum(result, 0)  # Preserve legacy nonnegative feature values.
        features.append(result)
        statuses.append(status)
    return grid, np.column_stack(features), np.column_stack(statuses)


def process_record(video, output_dir, matlab, max_frames=None):
    with tempfile.TemporaryDirectory(prefix="openworm-matlab-") as directory:
        mat_file = Path(directory) / "features.mat"
        command = (f"addpath({matlab_string(Path(__file__).parent)}); "
                   f"openworm_matlab_features({matlab_string(video)}, {matlab_string(mat_file)}, "
                   f"{max_frames if max_frames else 'inf'});")
        subprocess.run([matlab, "-wait", "-noFigureWindows", "-batch", command], check=True)
        raw = loadmat(mat_file, simplify_cells=True)
    grid, values, statuses = preprocess_extracted(raw)
    output = output_dir / video.parent.parent.name / f"{video.parent.name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["time_s", *FEATURES, *[name+"_status" for name in FEATURES]])
        for t, row, flags in zip(grid, values, statuses):
            writer.writerow([f"{t:.8f}", *[f"{value:.8f}" for value in row], *flags])
    print(f"{output}: {len(grid):,} frames, 16 features", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/openworm")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/matlab_features")
    parser.add_argument("--matlab", type=Path)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    records = ([args.input_dir] if args.input_dir.name.startswith("record_") else
               sorted(path for path in args.input_dir.rglob("record_*") if path.is_dir()))
    failures = skipped = processed = 0
    executable = None
    for record in records:
        videos = [path for path in record.glob("*.hdf5") if not path.name.endswith("_features.hdf5")]
        if not videos:
            print(f"SKIP {record}: no video .hdf5")
            skipped += 1
            continue
        try:
            if len(videos) != 1:
                raise ValueError("Expected exactly one video .hdf5 per record")
            if not videos[0].with_name(videos[0].stem+"_features.hdf5").exists():
                raise ValueError("Original algorithm requires companion skeleton/contour coordinates")
            executable = executable or matlab_executable(args.matlab)
            process_record(videos[0].resolve(), args.output_dir, executable, args.max_frames)
            processed += 1
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            failures += 1
            print(f"FAILED {record}: {error}")
    print(f"Processed {processed}; skipped {skipped}; failed {failures}.")
    return 1 if failures or not records else 0


if __name__ == "__main__":
    raise SystemExit(main())
