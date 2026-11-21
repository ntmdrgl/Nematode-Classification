"""Extract the original 16 MATLAB features, then preprocess them in Python.

Run from the repository root (requires MATLAB + Image Processing Toolbox):
    python openworm/openworm_matlab_features.py
    python openworm/openworm_matlab_features.py --input-dir E:/openworm
    python openworm/openworm_matlab_features.py --input-dir D:/data/openworm --workers 2
    python openworm/openworm_matlab_features.py --variant new --input-dir D:/data/openworm --workers 2
    python openworm/openworm_matlab_features.py --variant old
    python openworm/openworm_matlab_features.py --variant new --input-dir "E:/nematode classification/openworm" --output-dir D:/data/matlab_features_new --workers 2 --frame-workers 3
    python openworm/openworm_matlab_features.py --input-dir data/openworm/test
    python openworm/openworm_matlab_features.py --max-frames 100 --output-dir data/matlab_features_preview

Default output: data/matlab_features/<strain>/record_*.csv. Outputs include time_s,
16 features selected in old/openworm_dataset_preprocessing.m, and per-feature
real/interpolated/zeroed statuses. FPS and absolute coordinates are excluded.
Records without a video .hdf5 are skipped. A matching _features.hdf5 is also
required for the original algorithm's skeleton/contour coordinates (not its
features_timeseries). Raw image calculations and their original indexing and
units are preserved; no spatial smoothing is added to MATLAB's image pipeline.
MATLAB batches frame reads in a bounded ~32 MiB cache and reuses its disk
structuring element; all frame calculations still run in the original order.

MATLAB retains the legacy formulas, including FPS-scaled total displacements
and nonnegative clipping. Original missing-value masks are passed separately
so the old unlimited fillmissing cannot hide gaps. Python restores those gaps,
inserts dropped video frames, resamples every 0.04s, fills gaps strictly shorter
than 0.25s, and zeroes longer/unbracketed gaps. No Savitzky-Golay smoothing is
applied to these features. Orientation has a 180-degree period.
The old frame trimming/downsampling/statistical summaries are not applied.
Existing CSVs and records without videos are skipped silently; the final summary
reports the skipped count. CSVs are published only after MATLAB finishes
all requested frames and Python preprocessing/writing succeeds, using an atomic
rename of a temporary file. Unfinished records restart from their first frame.
Existing CSVs are trusted, not validated; use a separate output folder for
different variants/settings. --max-frames is for previews: always use a separate
preview output folder so truncated previews are not skipped as full records.
--workers defaults to 1. Each worker launches a separate MATLAB session for a
record; start with 2 workers because MATLAB sessions use substantial memory and
may themselves use multiple CPU threads. MATLAB/toolbox licenses must support
the concurrent sessions. --frame-workers enables bounded parfor image processing
inside each new-mode MATLAB session and requires Parallel Computing Toolbox.
Endpoint tracking, motion calculations, and feature assembly stay sequential.
Frame workers default to 1; --pool-type threads shares memory (recommended).
--pool-type processes is available as a higher-memory fallback. Total frame
workers = --workers * --frame-workers; avoid creating too many nested workers.
The total progress bar includes all eligible records and reports TOTAL ETA.
Each active record has a frame-count bar and ETA. Updates follow MATLAB's
existing progress messages (normally every 1,000 frames); the bar can pause
during startup or when skipped frames bypass MATLAB's progress reporting.
Completion is confirmed after extraction, Python preprocessing and CSV writing.
MATLAB output is captured per record; failures include the end of that output.

--variant old (default) uses openworm_matlab_features.m unchanged.
--variant new uses openworm_matlab_features_new.m: raw frame numbers align video
and coordinates; physical lengths are micrometers, areas micrometers squared,
speeds micrometers/second using timestamp differences, angles degrees, and
counts/ratios unitless. Net displacement from the start has no FPS multiplier.
New-mode skeleton length follows the mask skeleton's endpoint-to-endpoint
quasi-Euclidean path (Freeman axial/diagonal link weights), calibrated to um.
New-mode CSVs default to data/matlab_features_new, separating them from old-mode
outputs. Override --output-dir explicitly to choose a different destination.
Identical repeated acquisition frame/time pairs are omitted, and trailing
timestamps without mask images are ignored. Other invalid timestamps still fail.
"""

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from collections import deque
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm
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


FRAME_PROGRESS = re.compile(r":\s*[\d.]+%\s*\((\d+)/(\d+)\)")
BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, ETA {remaining}, {rate_fmt}]{postfix}"


def process_record(video, output_dir, matlab, max_frames=None, variant="old", show_progress=True,
                   progress_callback=None, frame_workers=1, pool_type="threads"):
    output = output_dir / video.parent.parent.name / f"{video.parent.name}.csv"
    if output.is_file():
        return f"SKIP {output}: CSV already exists"
    if variant not in ("old", "new"):
        raise ValueError(f"Unknown MATLAB variant: {variant}")
    function = "openworm_matlab_features_new" if variant == "new" else "openworm_matlab_features"
    if frame_workers < 1 or (variant == "old" and frame_workers != 1):
        raise ValueError("Frame parallelism requires --variant new and positive --frame-workers")
    if pool_type not in ('threads', 'processes'):
        raise ValueError("Unknown pool type")
    pool = 'Threads' if pool_type == 'threads' else 'Processes'
    with tempfile.TemporaryDirectory(prefix="openworm-matlab-") as directory:
        mat_file = Path(directory) / "features.mat"
        command = (f"addpath({matlab_string(Path(__file__).parent)}); "
                   f"{function}({matlab_string(video)}, {matlab_string(mat_file)}, "
                   f"{max_frames if max_frames else 'inf'}"
                   f"{', 32, ' + str(frame_workers) + ', ' + repr(pool) if variant == 'new' else ''});")
        arguments = [matlab, "-wait", "-noFigureWindows", "-batch", command]
        if progress_callback is not None:
            tail = deque(maxlen=30)
            with subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    tail.append(line)
                    match = FRAME_PROGRESS.search(line)
                    if match:
                        progress_callback(int(match[1]), int(match[2]))
                returncode = process.wait()
            if returncode:
                raise RuntimeError(f"MATLAB exited with code {returncode}:\n{''.join(tail)[-4000:]}")
        elif show_progress:
            subprocess.run(arguments, check=True)
        else:
            log_path = Path(directory) / "matlab.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                with log_path.open("rb") as log:
                    log.seek(max(0, log_path.stat().st_size - 4000))
                    tail = log.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"MATLAB exited with code {result.returncode}:\n{tail}")
        raw = loadmat(mat_file, simplify_cells=True)
    grid, values, statuses = preprocess_extracted(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
                                         dir=output.parent, prefix=output.name + ".", suffix=".part") as destination:
            temporary = Path(destination.name)
            writer = csv.writer(destination)
            writer.writerow(["time_s", *FEATURES, *[name+"_status" for name in FEATURES]])
            for t, row, flags in zip(grid, values, statuses):
                writer.writerow([f"{t:.8f}", *[f"{value:.8f}" for value in row], *flags])
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return f"{output}: {len(grid):,} frames, 16 features"


def run_records(files, worker, arguments, workers, description):
    """One MATLAB subprocess per thread; stable bars owned by the Python parent."""
    pending = Queue()
    for video in files:
        pending.put(video)
    stop = threading.Event()
    failures = 0
    lock = threading.Lock()
    with tqdm(total=len(files), desc=description, unit="record", position=0, smoothing=0,
              bar_format=BAR_FORMAT.replace("ETA {remaining}", "TOTAL ETA {remaining}")) as overall:
        def consume(slot):
            nonlocal failures
            while not stop.is_set():
                try:
                    video = pending.get_nowait()
                except Empty:
                    return
                try:
                    with h5py.File(video, "r") as source:
                        frames = source["mask"].shape[0]
                    if arguments[2] is not None:
                        frames = min(frames, arguments[2])
                    with tqdm(total=frames, desc=f"{video.parent.parent.name}/{video.parent.name}",
                              unit="frame", position=slot + 1, leave=False, smoothing=0,
                              bar_format=BAR_FORMAT) as current:
                        current.set_postfix_str("Starting MATLAB")
                        def advance(done, total):
                            if total != frames or not 0 <= done <= total:
                                raise ValueError(f"Unexpected MATLAB frame progress: {done}/{total}")
                            current.set_postfix_str("Calculating features", refresh=False)
                            current.update(max(0, done - current.n))
                        options = dict(show_progress=False, progress_callback=advance)
                        if len(arguments) >= 5:
                            options['frame_workers'] = arguments[4]
                        if len(arguments) >= 6:
                            options['pool_type'] = arguments[5]
                        message = worker(video, *arguments[:4], **options)
                        current.set_postfix_str("Complete", refresh=False)
                        current.update(frames - current.n)
                    tqdm.write(message)
                except Exception as error:
                    with lock:
                        failures += 1
                    tqdm.write(f"FAILED {video}: {error}")
                overall.update(1)
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            for future in as_completed([pool.submit(consume, slot) for slot in range(min(workers, len(files)))]):
                future.result()
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/openworm")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: data/matlab_features (old) or data/matlab_features_new (new)")
    parser.add_argument("--variant", choices=("old", "new"), default="old")
    parser.add_argument("--matlab", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent MATLAB record sessions (default: 1; start with 2)")
    parser.add_argument("--frame-workers", type=int, default=1,
                        help="MATLAB frame workers per record, new variant only (default: 1)")
    parser.add_argument("--pool-type", choices=('threads', 'processes'), default='threads',
                        help="Frame pool implementation; threads share memory (default: threads)")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.frame_workers < 1 or (args.variant == 'old' and args.frame_workers != 1):
        parser.error("--frame-workers must be positive and requires --variant new when greater than 1")
    output_dir = args.output_dir or ROOT / "data" / ("matlab_features_new" if args.variant == "new" else "matlab_features")
    records = ([args.input_dir] if args.input_dir.name.startswith("record_") else
               sorted(path for path in args.input_dir.rglob("record_*") if path.is_dir()))
    failures = skipped = 0
    eligible = []
    executable = None
    for record in records:
        output = output_dir / record.parent.name / f"{record.name}.csv"
        if output.is_file():
            skipped += 1
            continue
        videos = [path for path in record.glob("*.hdf5") if not path.name.endswith("_features.hdf5")]
        if not videos:
            skipped += 1
            continue
        try:
            if len(videos) != 1:
                raise ValueError("Expected exactly one video .hdf5 per record")
            if not videos[0].with_name(videos[0].stem+"_features.hdf5").exists():
                raise ValueError("Original algorithm requires companion skeleton/contour coordinates")
            executable = executable or matlab_executable(args.matlab)
            eligible.append(videos[0].resolve())
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            failures += 1
            print(f"FAILED {record}: {error}")
    status = run_records(eligible, process_record,
                         (output_dir.resolve(), executable, args.max_frames, args.variant, args.frame_workers, args.pool_type),
                         args.workers, "MATLAB records") if eligible else 0
    print(f"Eligible {len(eligible)}; skipped {skipped}; preflight failures {failures}.")
    return 1 if status or failures or not records else 0


if __name__ == "__main__":
    raise SystemExit(main())
