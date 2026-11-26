"""Select real mask images on the skeleton reference grid without altering pixels."""
from contextlib import ExitStack
from functools import partial
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZipFile

import h5py
import numpy as np
from tqdm import tqdm

from openworm.utils.openworm_alignment import reference_timestamps, sample_count
from openworm.openworm_skeleton_angles import temporal_resample
from openworm.openworm_dataset import ordered_results, write_array
from openworm.utils.openworm_video_workers import packed_records
from openworm.utils.openworm_zip import append_archive, append_entry
from openworm.utils.openworm_video_dataset import (FIELDS, METHOD_NAMES, image_header, load_array, inspect_record, video_files,
                                    copy_images, record_codec, validate_images, video_metadata, write_record_metadata)

_worker_archive = None
_worker_path = None


def close_worker_archive():
    global _worker_archive, _worker_path
    if _worker_archive is not None:
        _worker_archive.close()
    _worker_archive = _worker_path = None


def inspect_archive(path):
    with ZipFile(path) as archive:
        if load_array(archive, "source_dataset").item() != "mask":
            raise ValueError("Video dataset must contain mask images")
        labels, ids, keys = (load_array(archive, key) for key in ("labels", "record_ids", "record_keys"))
        if (any(value.ndim != 1 for value in (labels, ids, keys)) or not len(keys)
                or not len(labels) == len(ids) == len(keys) or len(set(map(str, keys))) != len(keys)):
            raise ValueError("Invalid video archive labels/record IDs/keys")
        result = {}
        for index, (strain, record_id, key) in enumerate(zip(labels, ids, keys)):
            pair, key = (str(strain), str(record_id)), str(key)
            if pair in result:
                raise ValueError("Duplicate video strain/record pair")
            with archive.open(key + ".npy") as source:
                shape, fortran, dtype = image_header(source)
            if fortran:
                raise ValueError("Video trimming requires C-order mask exports from openworm_video")
            method = archive.getinfo(key + ".npy").compress_type
            if method not in METHOD_NAMES:
                raise ValueError("Video records must use lossless Deflate, bzip2 or LZMA compression")
            result[pair] = (key, f"{index:06d}", shape[0], method)
    return result


def frame_plan(archive, key, suffix, count, features, reference_samples, target_samples,
               video_dir, record, alignment, max_error, metadata=None):
    if metadata is None:
        available = set(archive.namelist())
        metadata = {name: load_array(archive, f"{name}_{suffix}") for name in FIELDS
                    if f"{name}_{suffix}.npy" in available}
    if "frame_number" in metadata and "time_s" in metadata:
        raw, times = metadata["frame_number"], metadata["time_s"]
    else:
        if video_dir is not None:
            video = Path(video_dir) / record[0] / record[1] / features.name.replace("_features.hdf5", ".hdf5")
        elif "source_file" in metadata:
            video = Path(str(metadata["source_file"].item()))
        else:
            raise ValueError(f"Missing video timestamps for {record}; re-export videos or set --video-dir")
        with h5py.File(video, "r") as source:
            if source["mask"].shape[0] != count:
                raise ValueError(f"Source video length disagrees with video archive: {video}")
            raw = source["timestamp/raw"][:].reshape(-1)[:count]
            times = source["timestamp/time"][:].reshape(-1)[:count]
    raw, times = np.asarray(raw), np.asarray(times)
    if (raw.ndim != 1 or times.ndim != 1 or len(raw) != count or len(times) != count
            or not np.isfinite(raw).all() or not np.isfinite(times).all()
            or np.any(raw != np.floor(raw))):
        raise ValueError(f"Invalid video timestamps for {record}")
    keep = np.r_[True, (np.diff(raw) != 0) | (np.diff(times) != 0)]
    rows = np.flatnonzero(keep)
    unique_raw = np.asarray(raw[keep], dtype=float)
    unique_times = np.asarray(times[keep], dtype=float)
    if np.any(np.diff(unique_raw) <= 0) or np.any(np.diff(unique_times) <= 0):
        raise ValueError(f"Conflicting/backwards video timestamps for {record}")
    _, reference_times, clock_frames, clock_times = reference_timestamps(features)
    if sample_count(reference_times) != reference_samples or not 0 < target_samples <= reference_samples:
        raise ValueError(f"Reference timestamps disagree with skeleton archive length: {features}")
    grid = np.arange(target_samples) * .04
    query_frames = np.interp(reference_times[0] + grid, clock_times, clock_frames)
    query_times = np.interp(query_frames, unique_raw, unique_times)
    # Preserve out-of-coverage targets so gap handling blacks them instead of
    # clamping them to the first/last available image.
    for outside, a, b in ((query_frames < unique_raw[0], 0, 1),
                          (query_frames > unique_raw[-1], -2, -1)):
        if outside.any():
            if len(unique_raw) == 1:
                query_times[outside] = unique_times[0] + (query_frames[outside] - unique_raw[0]) * .04
            else:
                query_times[outside] = unique_times[a] + (query_frames[outside] - unique_raw[a]) * (
                    (unique_times[b] - unique_times[a]) / (unique_raw[b] - unique_raw[a]))
    # Mirror MATLAB preprocessing: expand dropped acquisition frames as NaNs,
    # then use the same strict <0.25 s bracket rule on the reference grid.
    full_frames = np.arange(int(unique_raw[0]), int(unique_raw[-1]) + 1)
    full_times = np.interp(full_frames, unique_raw, unique_times) - unique_times[0]
    available = np.full(len(full_frames), np.nan)
    available[unique_raw.astype(np.int64) - int(unique_raw[0])] = 1.
    _, _, valid, status = temporal_resample(full_times, available, grid=query_times - unique_times[0])
    right = np.minimum(np.searchsorted(unique_times, query_times), len(unique_times) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(abs(unique_times[right] - query_times) < abs(unique_times[left] - query_times), right, left)
    indices = rows[nearest]
    errors = np.where(valid, times[indices] - query_times, np.nan)
    tolerance = 1e-8 if alignment == "exact" else (np.inf if max_error is None else max_error)
    if np.any(abs(errors) > tolerance + 1e-12):
        raise ValueError(f"Video alignment for {record} exceeds {tolerance:.8f}s tolerance "
                         f"(maximum {np.nanmax(abs(errors)):.8f}s); pixels were not interpolated")
    result = {name: value for name, value in metadata.items() if name not in ("frame_number", "time_s")}
    result.update(frame_number=np.where(valid, raw[indices], -1),
                  time_s=np.where(valid, times[indices], np.nan),
                  source_index=np.where(valid, indices, -1), frame_status=status,
                  reference_frame_number=query_frames, target_time_s=query_times, alignment_error_s=errors)
    return result["source_index"], result


def maximum_timing_error(metadata):
    errors = metadata["alignment_error_s"]
    return float(np.max(abs(errors[np.isfinite(errors)]), initial=0.))


def prepare_video_record(path, key, suffix, count, features, reference_samples, target_samples,
                         video_dir, record, alignment, max_error):
    global _worker_archive, _worker_path
    if _worker_path != path:
        close_worker_archive()
        _worker_archive = ZipFile(path)
        _worker_path = path
    return frame_plan(_worker_archive, key, suffix, count, features, reference_samples,
                      target_samples, video_dir, record, alignment, max_error)


def pack_trim_record(index, arguments, directory, callback):
    # Plan and image selection/compression run in the same worker. Only the
    # compressed part's path and maximum timing error are returned to the parent.
    method = arguments[-1]
    indices, metadata = prepare_video_record(*arguments[:-1])
    key = arguments[1]
    part = Path(directory) / f"{index:06d}.npz.part"
    try:
        with ZipFile(part, "w", compression=ZIP_DEFLATED, compresslevel=9) as output, record_codec(output, method):
            callback(0, len(indices))
            copy_images(_worker_archive, key, output, f"data_{index:06d}", indices, callback)
            write_record_metadata(output, index, metadata, method)
        return index, part, maximum_timing_error(metadata)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def export_plan(arguments):
    path, record, features, reference_samples, target, video_dir, alignment, max_error = arguments
    info = inspect_record(path, record)
    if info["fortran"]:
        raise ValueError("Video trimming requires C-order mask exports")
    with ZipFile(path) as source:
        indices, metadata = frame_plan(source, "mask", None, info["shape"][0], features,
                                       reference_samples, target, video_dir, record, alignment,
                                       max_error, metadata=info["metadata"])
    return indices, metadata, info


def prepare_export_record(*arguments):
    indices, metadata, _ = export_plan(arguments)
    return indices, metadata


def pack_trim_export_record(index, arguments, directory, callback):
    indices, metadata, info = export_plan(arguments)
    part = Path(directory) / f"{index:06d}.npz.part"
    try:
        with ZipFile(arguments[0]) as source, ZipFile(part, "w") as output, record_codec(output, info["method"]):
            callback(0, len(indices))
            unchanged = (len(indices) == info["shape"][0]
                         and np.array_equal(indices, np.arange(info["shape"][0])))
            if unchanged and source.getinfo("mask.npy").compress_type in METHOD_NAMES:
                # Keep an already-aligned stack's compressed bytes, after CRC validation.
                validate_images(source, callback)
                append_entry(output, source, "mask.npy", f"data_{index:06d}.npy")
            else:
                copy_images(source, "mask", output, f"data_{index:06d}", indices, callback)
            write_record_metadata(output, index, metadata, info["method"])
        return index, part, maximum_timing_error(metadata)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def hdf5_plan(arguments):
    path, record, features, reference_samples, target, alignment, max_error, original = arguments
    with h5py.File(path, "r") as source:
        images = source["mask"]
        if images.ndim != 3 or any(n < 1 for n in images.shape) or images.dtype.kind not in "buif":
            raise ValueError(f"Expected nonempty numeric 3D /mask: {original}")
        count, shape, dtype = images.shape[0], images.shape, images.dtype
        metadata = {"source_file": np.asarray(str(original))}
        for name in ("fps", "expected_fps"):
            if name in images.attrs:
                metadata[name] = np.asarray(images.attrs[name])
        for key, name in (("timestamp/raw", "frame_number"), ("timestamp/time", "time_s")):
            if key not in source:
                raise ValueError(f"Missing /{key}; cannot align mask images: {original}")
            metadata[name] = np.asarray(source[key][:]).reshape(-1)[:count]
        indices, metadata = frame_plan(None, None, None, count, features, reference_samples,
                                       target, None, record, alignment, max_error, metadata=metadata)
    return indices, metadata, shape, dtype


def prepare_hdf5_record(*arguments):
    indices, metadata, _, _ = hdf5_plan(arguments)
    return indices, metadata


def pack_trim_hdf5_record(index, arguments, directory, callback):
    indices, metadata, shape, dtype = hdf5_plan(arguments)
    part = Path(directory) / f"{index:06d}.npz.part"
    try:
        with h5py.File(arguments[0], "r") as source, ZipFile(part, "w") as output, record_codec(output, ZIP_BZIP2):
            with output.open(f"data_{index:06d}.npy", "w", force_zip64=True) as destination:
                np.lib.format.write_array_header_2_0(destination, {
                    "descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False,
                    "shape": (len(indices), *shape[1:])})
                callback(0, len(indices))
                previous, image = None, None
                black = np.zeros(shape[1:], dtype=dtype)
                images = source["mask"]
                for done, row in enumerate(indices, 1):
                    row = int(row)
                    if row < 0:
                        selected = black
                    else:
                        if row != previous:
                            image = np.ascontiguousarray(images[row], dtype=dtype)
                            previous = row
                        selected = image
                    destination.write(memoryview(selected).cast("B"))
                    callback(done, len(indices))
            write_record_metadata(output, index, metadata, ZIP_BZIP2)
        return index, part, maximum_timing_error(metadata)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def stage_archive_record(record, arguments, directory, stop, callback, archive):
    from openworm.utils.openworm_video_staging import StagingCancelled
    path, key, suffix, count, features, reference_samples, target, video_dir, pair, alignment, max_error, method = arguments
    destination = Path(directory) / "input.npz"
    def report(done, total):
        if stop.is_set():
            raise StagingCancelled()
        callback(done, total)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as output:
        write_array(output, "source_dataset", "mask")
        write_array(output, "strain", record[0])
        write_array(output, "record_id", record[1])
        append_entry(output, archive, key + ".npy", "mask.npy", report)
        available = set(archive.namelist())
        for name in FIELDS:
            if f"{name}_{suffix}.npy" in available:
                write_array(output, name, load_array(archive, f"{name}_{suffix}"))
    return destination, record, features, reference_samples, target, video_dir, alignment, max_error


def trim_video(data_dir, output_dir=None, workers=1, input_dir=None, video_dir=None,
               check_alignment=False, alignment="nearest", max_error=None,
               staging_dir=None, video_source="auto"):
    from openworm.openworm_trim import ALL_DATASETS, MIN_SAMPLES, inspect_dataset
    if (alignment not in ("nearest", "exact")
            or (max_error is not None and (not np.isfinite(max_error) or max_error < 0))):
        raise ValueError("Invalid video alignment method or maximum frame error")
    if video_source not in ("auto", "records", "archive", "hdf5"):
        raise ValueError("Invalid video source")
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else data_dir
    input_dir = Path(input_dir).resolve() if input_dir is not None else data_dir / "openworm"
    references = [inspect_dataset(data_dir / f"openworm_{name}_dataset.npz", width) for name, width in ALL_DATASETS]
    sizes = references[0][2]
    if any(item[2] != sizes for item in references[1:]):
        raise ValueError("Reference datasets must have identical strain/record pairs and sample counts")
    records = sorted(record for record, length in sizes.items() if length > MIN_SAMPLES)
    if not records:
        raise ValueError("No reference records are longer than 890 seconds")
    target = min(sizes[record] for record in records)
    path = data_dir / "openworm_video_dataset.npz"
    files = video_files(data_dir) if video_source in ("auto", "records") else {}
    if video_source == "auto":
        video_source = "records" if files else ("archive" if path.is_file() else "hdf5")
    features = {}
    for record in records:
        paths = list((input_dir / record[0] / record[1]).glob("*_features.hdf5"))
        if len(paths) != 1:
            raise ValueError(f"Expected one reference features HDF5 for {record}; set --input-dir")
        features[record] = paths[0]
    if video_source == "archive":
        mapping = inspect_archive(path)
        if set(records) - mapping.keys():
            raise ValueError("Video dataset is missing retained skeleton records")
        arguments = [(path, *mapping[r][:3], features[r], sizes[r], target, video_dir, r, alignment, max_error) for r in records]
        prepare = prepare_video_record
        pack = pack_trim_record
        jobs = [(r, (*a, mapping[r][3])) for r, a in zip(records, arguments)]
    elif video_source == "records":
        if set(records) - files.keys():
            raise ValueError("Per-record video NPZs are missing retained skeleton records")
        arguments = [(files[r], r, features[r], sizes[r], target, video_dir, alignment, max_error) for r in records]
        prepare, pack = prepare_export_record, pack_trim_export_record
        jobs = list(zip(records, arguments))
    else:
        raw_root = Path(video_dir).resolve() if video_dir is not None else input_dir
        arguments = []
        for r in records:
            original = raw_root / r[0] / r[1] / features[r].name.replace("_features.hdf5", ".hdf5")
            if not original.is_file():
                raise ValueError(f"Missing video HDF5: {original}; set --video-dir")
            arguments.append((original, r, features[r], sizes[r], target, alignment, max_error, original))
        prepare, pack = prepare_hdf5_record, pack_trim_hdf5_record
        jobs = list(zip(records, arguments))
    destination = output_dir / "openworm_video_trimmed_890_dataset.npz"
    temporary = destination.with_suffix(".npz.part")
    maximum_error, completed = 0., None
    over_20ms = 0
    print(f"Keeping {len(records):,}/{len(sizes):,} reference records; selecting {target:,} real mask images per record ({alignment}; {video_source}).", flush=True)
    try:
        if check_alignment:
            completed = ordered_results(prepare, arguments, workers)
            with tqdm(total=len(records), desc="Checking video frame alignment", unit="record") as progress:
                for indices, metadata in completed:
                    error = maximum_timing_error(metadata)
                    maximum_error = max(maximum_error, error)
                    over_20ms += error > .02 + 1e-9
                    progress.update(1)
            print(f"Checked {len(records):,} records, {target:,} images each; maximum timing offset {maximum_error:.8f}s; {over_20ms:,} records exceed 20 ms.")
            return []
        output_dir.mkdir(parents=True, exist_ok=True)
        scratch = Path(staging_dir).resolve() if staging_dir is not None else output_dir
        scratch.mkdir(parents=True, exist_ok=True)
        with ExitStack() as resources:
            parts = resources.enter_context(tempfile.TemporaryDirectory(prefix=".openworm-video-trim-", dir=scratch))
            output = resources.enter_context(ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9))
            progress = resources.enter_context(tqdm(total=len(records), desc="Trimming video records", unit="record"))
            video_metadata(output, records)
            write_array(output, "time_s", np.arange(target) * .04)
            write_array(output, "alignment_reference", "skeleton_angles")
            write_array(output, "alignment_method", alignment)
            write_array(output, "gap_policy", "nearest_short_gap_black_long_gap")
            write_array(output, "gap_threshold_s", .25)
            write_array(output, "alignment_tolerance_s",
                        1e-8 if alignment == "exact" else (np.inf if max_error is None else max_error))
            release = poll = None
            if staging_dir is not None:
                from openworm.utils.openworm_archive_staging import StagedJobs, stage_file
                stager = stage_file
                if video_source == "archive":
                    original = resources.enter_context(ZipFile(path))
                    stager = partial(stage_archive_record, archive=original)
                    pack = pack_trim_export_record
                feed = resources.enter_context(StagedJobs(jobs, staging_dir, workers, progress, stager))
                jobs, release, poll = feed, feed.release, feed.poll
            completed = packed_records(pack, jobs, workers, parts, progress, release, poll)
            try:
                for index, part, error in completed:
                    try:
                        append_archive(output, part)
                    finally:
                        part.unlink(missing_ok=True)
                    maximum_error = max(maximum_error, error)
                    over_20ms += error > .02 + 1e-9
                    progress.set_postfix_str(f"max offset {maximum_error * 1000:.3f} ms")
                    progress.update(1)
            finally:
                completed.close()
                completed = None
        os.replace(temporary, destination)
    finally:
        if completed is not None:
            completed.close()
        close_worker_archive()
        temporary.unlink(missing_ok=True)
    print(f"Saved {destination}; maximum timing offset {maximum_error:.8f}s; {over_20ms:,} records exceed 20 ms.")
    return [destination]
