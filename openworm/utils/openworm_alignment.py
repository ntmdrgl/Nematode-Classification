"""Map the skeleton pipeline's samples to the original video acquisition clock."""
from collections import deque
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

STEP = 0.04


def sample_count(times):
    return int(np.floor((times[-1] - times[0]) / STEP + 1e-9)) + 1


def video_timestamps(path):
    with h5py.File(path, "r") as source:
        raw = np.asarray(source["timestamp/raw"][:], dtype=float).reshape(-1)
        times = np.asarray(source["timestamp/time"][:], dtype=float).reshape(-1)
        count = source["mask"].shape[0]
    if len(raw) != len(times) or len(raw) < count or count < 1:
        raise ValueError(f"Video timestamps do not cover mask images: {path}")
    raw, times = raw[:count], times[:count]
    if (not np.isfinite(raw).all() or not np.isfinite(times).all()
            or np.any(raw != np.floor(raw))):
        raise ValueError(f"Invalid video timestamps: {path}")
    keep = np.r_[True, (np.diff(raw) != 0) | (np.diff(times) != 0)]
    raw, times = raw[keep], times[keep]
    if np.any(np.diff(raw) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError(f"Conflicting/backwards video timestamps: {path}")
    return raw, times


def reference_timestamps(features):
    # Mirror openworm_skeleton_angles.read_record, including its FPS fallback
    # when the adjacent video is absent or its original timestamps are invalid.
    with h5py.File(features, "r") as source:
        table = source["features_timeseries"]
        frames = np.asarray(table["timestamp"][:], dtype=float).reshape(-1)
        fps = float(table.attrs.get("fps", np.nan))
    if (not len(frames) or not np.isfinite(frames).all()
            or np.any(np.diff(frames) <= 0)):
        raise ValueError(f"Invalid reference frame numbers: {features}")
    times = None
    clock_frames = clock_times = None
    video = features.with_name(features.name.replace("_features.hdf5", ".hdf5"))
    if video.exists():
        with h5py.File(video, "r") as source:
            if "timestamp/raw" in source and "timestamp/time" in source:
                raw = source["timestamp/raw"][:].reshape(-1)
                seconds = source["timestamp/time"][:].reshape(-1)
                if (len(raw) == len(seconds) and len(raw) >= 2
                        and np.isfinite(raw).all() and np.isfinite(seconds).all()
                        and np.all(np.diff(raw) > 0) and np.all(np.diff(seconds) > 0)):
                    if frames[0] < raw[0] or frames[-1] > raw[-1]:
                        raise ValueError(f"Reference frames outside video timestamps: {features}")
                    times = np.interp(frames, raw, seconds)
                    clock_frames, clock_times = raw, seconds
    if times is None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Missing reference FPS: {features}")
        times = frames / fps
        clock_frames, clock_times = frames, times
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Invalid reference times: {features}")
    return frames, times, clock_frames, clock_times


def alignment_plan(features, video, reference_samples, matlab_samples, target_samples):
    frames, reference_times, clock_frames, clock_times = reference_timestamps(features)
    if sample_count(reference_times) != reference_samples:
        raise ValueError(f"Reference timestamps disagree with archive length: {features}")
    raw, video_times = video_timestamps(video)
    if sample_count(video_times) != matlab_samples:
        raise ValueError(f"Video timestamps disagree with MATLAB archive length: {video}")
    # Convert the reference clock to acquisition frames, then to the actual
    # video clock. This also handles the reference pipeline's FPS fallback.
    if not 0 < target_samples <= reference_samples:
        raise ValueError("Target length must be within the reference record")
    grid = np.arange(target_samples) * STEP
    query_frames = np.interp(grid + reference_times[0], clock_times, clock_frames)
    if query_frames[0] < raw[0] or query_frames[-1] > raw[-1]:
        raise ValueError(f"Video does not cover target acquisition frames: {video}")
    query_times = np.interp(query_frames, raw, video_times)
    positions = (query_times - video_times[0]) / STEP
    indices = np.rint(positions).astype(np.int64)
    if (np.allclose(positions, indices, rtol=0, atol=1e-7)
            and np.all(indices >= 0) and np.all(indices < matlab_samples)):
        return indices, query_times
    return None, query_times


def rebuild_record(video, query_times, matlab=None, frame_workers=1, pool_type="threads",
                   progress_callback=None, temporary_dir=None):
    # Resample original observations once. Interpolating already-resampled NPZ
    # rows would change gap boundaries and orientation wrapping.
    if frame_workers < 1:
        raise ValueError("frame_workers must be at least 1")
    if pool_type not in ("threads", "processes"):
        raise ValueError("Unknown pool type")
    from scipy.io import loadmat
    from openworm.openworm_matlab_features import (FRAME_PROGRESS, matlab_executable, matlab_string,
                                          preprocess_extracted)
    pool = "Threads" if pool_type == "threads" else "Processes"
    with h5py.File(video, "r") as source:
        total = source["mask"].shape[0]
    def report(stage, done=0):
        if progress_callback is not None:
            progress_callback(stage, done, total)
    report("Starting MATLAB")
    scripts = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="openworm-align-", dir=temporary_dir) as directory:
        output = Path(directory) / "features.mat"
        command = (f"addpath({matlab_string(scripts)}); "
                   f"openworm_matlab_features_new({matlab_string(video)}, "
                   f"{matlab_string(output)}, inf, 32, {frame_workers}, {pool!r});")
        tail = deque(maxlen=30)
        with subprocess.Popen([matlab_executable(matlab), "-wait", "-noFigureWindows",
                               "-batch", command], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    tail.append(line[-4000:])
                    match = FRAME_PROGRESS.search(line)
                    if match:
                        done, reported_total = int(match[1]), int(match[2])
                        if reported_total != total or not 0 <= done <= total:
                            raise ValueError(f"Unexpected MATLAB frame progress: {done}/{reported_total}")
                        report("Calculating features", done)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if returncode:
            raise ValueError(f"MATLAB alignment failed for {video}: {''.join(tail)[-4000:]}")
        report("Resampling to reference grid", total)
        raw = loadmat(output, simplify_cells=True)
    _, values, _ = preprocess_extracted(raw, sample_times=query_times)
    if values.shape != (len(query_times), 16) or not np.isfinite(values).all():
        raise ValueError(f"Invalid rebuilt MATLAB record: {video}")
    report("complete", total)
    return values
