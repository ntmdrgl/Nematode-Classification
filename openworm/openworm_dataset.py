"""Combine per-record CSVs into labeled NPZ datasets (no pickle needed).

Run from the repository root:
    python openworm/openworm_dataset.py
    python openworm/openworm_dataset.py --dataset skeleton_angles
    python openworm/openworm_dataset.py --dataset skeleton_angles --workers 4
    python openworm/openworm_dataset.py --workers 4
    python openworm/openworm_dataset.py --dataset manual_features
    python openworm/openworm_dataset.py --dataset eigenworms
    python openworm/openworm_dataset.py --dataset matlab_features
    python openworm/openworm_dataset.py --dataset matlab_features_new
    python openworm/openworm_dataset.py --dataset video --data-dir D:/data --workers 4
    python openworm/openworm_dataset.py --data-dir C:/path/to/data

Reads skeleton_angles/<strain>/record_*.csv and manual_features/<strain>/record_*.csv.
Writes openworm_skeleton_angles_dataset.npz (48 angles),
openworm_manual_features_dataset.npz (47 features), and
openworm_eigenworms_dataset.npz (6 eigen projections), and
openworm_matlab_features_dataset.npz (16 MATLAB features) in the data directory.
MATLAB records are read from matlab_features/<strain>/record_*.csv and may be
a smaller set, since records without videos were skipped during extraction.
The new MATLAB variant is read separately from matlab_features_new/<strain>/record_*.csv
and saved as openworm_matlab_features_new_dataset.npz (16 features). In all mode,
this additional archive is included only when new-variant CSVs are present.
Video mode reads video/<strain>/record_*.npz and writes openworm_video_dataset.npz.
Every data_<index> entry remains a 3D (frames, height, width) array with unchanged
source dtype and pixels. Labels, IDs and sorted record keys follow the same
strain/record scheme as the feature datasets; video records may be a subset.
It retains each record's selected lossless ZIP codec (bzip2, LZMA or Deflate),
copying compressed pixel payloads unchanged instead of re-encoding them, and stores frame_number_<index>, time_s_<index>
and source_file_<index> when supplied by openworm_video. No flattening or float
conversion occurs. --workers runs independent record processes that validate
pixels/CRC and prepare compressed records. The parent merges compressed ZIP
entries without another decompression/compression pass. Workers use bounded
buffers and at most one queued compressed record per worker plus the one being
merged; temporary compressed parts are cleaned up after success or failure. Video is included in all mode only if per-record mask NPZs exist.
--staging-dir copies video inputs one at a time onto the selected SSD while
workers run, keeps compressed worker parts there, and removes owned copies when
their jobs finish. --output-dir controls the video archive destination.
--dataset video --trim-video writes only the trimmed archive without assembling
an untrimmed one. Use --video-source hdf5 --video-dir PATH to read raw /mask
images directly, or records for existing per-record NPZs. It uses exactly the
same reference selection/alignment as openworm_trim; reference HDF5s stay at
--input-dir so their original clock-selection behavior does not change.
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
--dataset defaults to all. Individual modes read only their source folder and
replace only the selected NPZ, without requiring the other datasets' CSVs.
--workers defaults to 1. Higher values read and validate records in parallel;
archive writing stays in the parent process, with at most one record per worker
queued in memory. Record order and labels are unchanged.
A total progress bar reports records written, elapsed time, and TOTAL ETA,
including CSV reading, validation, and compression, in serial and parallel runs.
"""

# Allow both `python openworm/<script>.py` and package imports/module execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
ANGLES = [f"angle_{i:02d}" for i in range(1, 49)]
EIGEN = [f"eigen_projection_{i}" for i in range(1, 7)]
OUTPUTS = ("openworm_skeleton_angles_dataset.npz", "openworm_manual_features_dataset.npz",
           "openworm_eigenworms_dataset.npz", "openworm_matlab_features_dataset.npz",
           "openworm_matlab_features_new_dataset.npz", "openworm_video_dataset.npz")
DATASET_NAMES = ("skeleton_angles", "manual_features", "eigenworms", "matlab_features", "matlab_features_new", "video")
PROGRESS_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, TOTAL ETA {remaining}, {rate_fmt}{postfix}]"


def total_progress(total):
    return tqdm(total=total, desc="Dataset records", unit="record", smoothing=0,
                bar_format=PROGRESS_FORMAT)


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


def ordered_results(function, jobs, workers):
    """Read ahead a bounded number of records, yielding in submission order."""
    if workers == 1:
        for job in jobs:
            yield function(*job)
        return
    jobs = iter(jobs)
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = deque()
        try:
            for _ in range(workers):
                job = next(jobs, None)
                if job is None:
                    break
                pending.append(pool.submit(function, *job))
            while pending:
                yield pending.popleft().result()
                job = next(jobs, None)
                if job is not None:
                    pending.append(pool.submit(function, *job))
        finally:
            for future in pending:
                future.cancel()


def selected_names(path, dataset):
    columns = [name for name in header(path) if name != "time_s" and not name.endswith("_status")]
    if dataset == "skeleton_angles":
        if set(columns) != set(ANGLES):
            raise ValueError(f"Expected 48 angle columns: {path}")
        return ANGLES
    if dataset == "eigenworms":
        if not set(EIGEN).issubset(columns):
            raise ValueError(f"Expected eigen_projection_1 through eigen_projection_6: {path}")
        return EIGEN
    names = [name for name in columns if name not in EIGEN] if dataset == "manual_features" else columns
    if len(names) != (47 if dataset == "manual_features" else 16):
        raise ValueError(f"Unexpected feature count for {dataset}: {path}")
    return names


def read_record(path, dataset, names):
    if set(selected_names(path, dataset)) != set(names):
        raise ValueError(f"Inconsistent {dataset} feature columns: {path}")
    return read_values(path, names)[1]


def read_paired_record(skeleton_path, feature_path, all_features, manual_names):
    angle_names = [name for name in header(skeleton_path) if name.startswith("angle_")]
    current_names = [name for name in header(feature_path)
                     if name != "time_s" and not name.endswith("_status")]
    if set(angle_names) != set(ANGLES) or set(current_names) != set(all_features):
        raise ValueError(f"Inconsistent feature columns: {skeleton_path}, {feature_path}")
    angle_time, angle_data = read_values(skeleton_path, ANGLES)
    feature_time, feature_data = read_values(feature_path, all_features)
    if angle_time.shape != feature_time.shape or not np.allclose(angle_time, feature_time, rtol=0, atol=1e-8):
        raise ValueError(f"Skeleton/feature time grids differ: {skeleton_path}")
    return (angle_data, feature_data[:, [all_features.index(name) for name in manual_names]],
            feature_data[:, [all_features.index(name) for name in EIGEN]])


def build_single_dataset(data_dir, dataset, workers=1, staging_dir=None, output_dir=None):
    if dataset == "video":
        from openworm.utils.openworm_video_dataset import build_video_dataset
        return build_video_dataset(data_dir, workers, staging_dir, output_dir)
    folder = "manual_features" if dataset == "eigenworms" else dataset
    files = record_files(data_dir / folder)
    if not files:
        raise ValueError(f"No CSVs found in {folder}/<strain>/record_*.csv")
    records = sorted(files)

    names = selected_names(files[records[0]], dataset)
    destination = data_dir / OUTPUTS[DATASET_NAMES.index(dataset)]
    temporary = destination.with_suffix(".npz.part")
    try:
        with total_progress(len(records)) as progress, \
                ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            write_metadata(archive, names, records)
            jobs = ((files[record], dataset, names) for record in records)
            for i, values in enumerate(ordered_results(read_record, jobs, workers)):
                record = records[i]
                write_array(archive, f"data_{i:06d}", values)
                progress.set_postfix_str(f"{dataset}: {record[0]}/{record[1]}", refresh=False)
                progress.update(1)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {destination}")
    return destination


def build_datasets(data_dir, dataset="all", workers=1, staging_dir=None, output_dir=None,
                   trim_video=False, input_dir=None, video_dir=None, video_source="auto"):
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if trim_video:
        if dataset != "video":
            raise ValueError("--trim-video requires --dataset video")
        from openworm.utils.openworm_video_trim import trim_video as write_trimmed
        return write_trimmed(data_dir, output_dir, workers, input_dir, video_dir,
                             staging_dir=staging_dir, video_source=video_source)
    if output_dir is not None and dataset != "video":
        raise ValueError("--output-dir is supported for video mode only")
    if dataset != "all":
        if dataset not in DATASET_NAMES:
            raise ValueError(f"Unknown dataset: {dataset}")
        return [build_single_dataset(data_dir, dataset, workers, staging_dir, output_dir)]
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
    new_files = record_files(data_dir / "matlab_features_new")
    new_records = sorted(new_files)
    new_names = selected_names(new_files[new_records[0]], "matlab_features_new") if new_records else None
    from openworm.utils.openworm_video_dataset import video_files, write_video_records
    videos = video_files(data_dir)
    selected_outputs = (*OUTPUTS[:4], *((OUTPUTS[4],) if new_records else ()),
                        *((OUTPUTS[5],) if videos else ()))
    paths = [data_dir / name for name in selected_outputs]
    temporary = [path.with_suffix(".npz.part") for path in paths]
    try:
        with ExitStack() as stack:
            progress = stack.enter_context(total_progress(len(records) + len(matlab_records) + len(new_records) + len(videos)))
            archives = [stack.enter_context(ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=6))
                        for path in temporary]
            for archive, names in zip(archives, (ANGLES, manual_names, EIGEN)):
                write_metadata(archive, names, records)
            write_metadata(archives[3], matlab_names, matlab_records)
            if new_records:
                write_metadata(archives[4], new_names, new_records)
            jobs = ((skeletons[record], features[record], all_features, manual_names) for record in records)
            for i, arrays in enumerate(ordered_results(read_paired_record, jobs, workers)):
                record = records[i]
                for archive, values in zip(archives, arrays):
                    write_array(archive, f"data_{i:06d}", values)
                progress.set_postfix_str(f"angles/manual/eigen: {record[0]}/{record[1]}", refresh=False)
                progress.update(1)
            jobs = ((matlab_files[record], "matlab_features", matlab_names) for record in matlab_records)
            for i, values in enumerate(ordered_results(read_record, jobs, workers)):
                record = matlab_records[i]
                write_array(archives[3], f"data_{i:06d}", values)
                progress.set_postfix_str(f"matlab_features: {record[0]}/{record[1]}", refresh=False)
                progress.update(1)
            if new_records:
                jobs = ((new_files[record], "matlab_features_new", new_names) for record in new_records)
                for i, values in enumerate(ordered_results(read_record, jobs, workers)):
                    record = new_records[i]
                    write_array(archives[4], f"data_{i:06d}", values)
                    progress.set_postfix_str(f"matlab_features_new: {record[0]}/{record[1]}", refresh=False)
                    progress.update(1)
            if videos:
                write_video_records(archives[-1], videos, workers, progress, staging_dir)
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
    parser.add_argument("--dataset", choices=("all", *DATASET_NAMES), default="all",
                        help="Build all archives or just the selected dataset (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel CSV processing/video-record processes (default: 1)")
    parser.add_argument("--staging-dir", type=Path, help="SSD folder for bounded video input copies and compressed parts")
    parser.add_argument("--output-dir", type=Path, help="Video output folder (default: --data-dir)")
    parser.add_argument("--trim-video", action="store_true", help="Save only the skeleton-aligned trimmed video archive; no combined untrimmed archive")
    parser.add_argument("--input-dir", type=Path, help="Original skeleton HDF5 root for --trim-video")
    parser.add_argument("--video-dir", type=Path, help="Original video HDF5 root for --trim-video")
    parser.add_argument("--video-source", choices=("auto", "records", "archive", "hdf5"), default="auto",
                        help="Video trim source: per-record NPZs, combined archive, or raw mask HDF5s")
    args = parser.parse_args()
    try:
        build_datasets(args.data_dir.resolve(), args.dataset, args.workers, staging_dir=args.staging_dir,
                       output_dir=args.output_dir, trim_video=args.trim_video, input_dir=args.input_dir,
                       video_dir=args.video_dir, video_source=args.video_source)
    except (OSError, ValueError, KeyError, EOFError, StopIteration) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
