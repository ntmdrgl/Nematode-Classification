"""Stream compressed mask NPZ records without expanding a video in memory."""
from contextlib import ExitStack, contextmanager
from math import prod
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZIP_STORED, ZipFile

import numpy as np
from tqdm import tqdm

from openworm.openworm_dataset import total_progress, write_array, write_metadata
from openworm.utils.openworm_zip import append_archive, append_entry
from openworm.utils.openworm_video_workers import packed_records

BUFFER_BYTES = 32 * 1024 ** 2
FIELDS = ("frame_number", "time_s", "source_file", "fps", "expected_fps")
METHOD_NAMES = {ZIP_DEFLATED: "deflate", ZIP_BZIP2: "bzip2", ZIP_LZMA: "lzma"}


def video_files(data_dir):
    return {(p.parent.name, p.stem): p for p in (Path(data_dir) / "video").glob("*/record_*.npz")}


def load_array(archive, key):
    with archive.open(key + ".npy") as source:
        return np.lib.format.read_array(source, allow_pickle=False)


def image_header(source):
    version = np.lib.format.read_magic(source)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
    else:
        raise ValueError(f"Unsupported image NPY version: {version}")
    if len(shape) != 3 or any(n < 1 for n in shape) or dtype.hasobject or dtype.kind not in "buif":
        raise ValueError(f"Expected a nonempty numeric 3D mask stack; got {shape}, {dtype}")
    return shape, fortran, dtype


def inspect_record(path, record):
    with ZipFile(path) as archive:
        if "mask.npy" not in archive.namelist():
            raise ValueError(f"Missing mask array (full_data is not supported): {path}")
        if load_array(archive, "source_dataset").item() != "mask":
            raise ValueError(f"Expected source_dataset=mask: {path}")
        for key, expected in zip(("strain", "record_id"), record):
            actual = load_array(archive, key)
            if actual.shape != () or str(actual.item()) != expected:
                raise ValueError(f"{key} disagrees with strain/record path: {path}")
        with archive.open("mask.npy") as source:
            shape, fortran, dtype = image_header(source)
        metadata = {}
        for key in FIELDS:
            if key + ".npy" in archive.namelist():
                value = load_array(archive, key)
                if key in ("frame_number", "time_s"):
                    if value.ndim != 1 or len(value) != shape[0] or value.dtype.kind not in "fiu":
                        raise ValueError(f"Invalid {key} metadata: {path}")
                elif value.ndim != 0:
                    raise ValueError(f"Expected scalar {key}: {path}")
                metadata[key] = value
        method = archive.getinfo("mask.npy").compress_type
        if method != ZIP_STORED and method not in METHOD_NAMES:
            raise ValueError(f"Unsupported lossless ZIP method {method}: {path}")
        return dict(shape=shape, fortran=fortran, dtype=dtype, metadata=metadata,
                    method=ZIP_BZIP2 if method == ZIP_STORED else method)


@contextmanager
def record_codec(archive, method):
    """Use the source's lossless codec for this record, including its metadata."""
    previous = archive.compression, archive.compresslevel
    archive.compression = method
    archive.compresslevel = 9 if method in (ZIP_BZIP2, ZIP_DEFLATED) else None
    try:
        yield
    finally:
        archive.compression, archive.compresslevel = previous


def copy_bytes(source, destination, count):
    while count:
        block = source.read(min(count, BUFFER_BYTES))
        if not block:
            raise ValueError("Truncated mask image payload")
        if destination is not None:
            destination.write(block)
        count -= len(block)


def copy_images(source_archive, source_key, output, key, indices=None, callback=None):
    """Stream raw NPY pixels; optionally select ordered source frames unchanged."""
    with source_archive.open(source_key + ".npy") as source:
        shape, fortran, dtype = image_header(source)
        frame_bytes = prod(shape[1:]) * dtype.itemsize
        if indices is not None:
            indices = np.asarray(indices)
            if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
                raise ValueError("Frame selection requires ordered integer indices")
            selected = indices[indices >= 0]
            if (fortran or np.any(indices < -1) or np.any(selected >= shape[0])
                    or np.any(selected[1:] < selected[:-1])):
                raise ValueError("Frame selection requires ordered, in-range indices or -1 for black frames, and C-order images")
            out_shape = (len(indices), *shape[1:])
        else:
            out_shape = shape
        with output.open(key + ".npy", "w", force_zip64=True) as destination:
            np.lib.format.write_array_header_2_0(destination, {
                "descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": fortran,
                "shape": out_shape})
            if indices is None:
                remaining = prod(shape) * dtype.itemsize
                copied = 0
                while remaining:
                    size = min(remaining, BUFFER_BYTES)
                    copy_bytes(source, destination, size)
                    copied += size
                    remaining -= size
                    if callback is not None:
                        callback(min(shape[0], copied // frame_bytes), shape[0])
            else:
                cursor, previous_frame = 0, None
                black = bytes(frame_bytes) if np.any(indices < 0) else None
                for done, index in enumerate(indices, 1):
                    index = int(index)
                    if index < 0:
                        destination.write(black)
                        if callback is not None:
                            callback(done, len(indices))
                        continue
                    if index != cursor - 1:
                        copy_bytes(source, None, (index - cursor) * frame_bytes)
                        previous_frame = source.read(frame_bytes)
                        if len(previous_frame) != frame_bytes:
                            raise ValueError("Truncated mask image payload")
                        cursor = index + 1
                    destination.write(previous_frame)
                    if callback is not None:
                        callback(done, len(indices))
                copy_bytes(source, None, (shape[0] - cursor) * frame_bytes)
            if source.read(1):
                raise ValueError("Unexpected trailing bytes in mask array")


def video_metadata(output, records):
    write_metadata(output, ["mask"], records)
    write_array(output, "source_dataset", "mask")
    write_array(output, "array_axes", ["frame", "height", "width"])


def write_record_metadata(output, index, metadata, method):
    for key, value in metadata.items():
        write_array(output, f"{key}_{index:06d}", value)
    write_array(output, f"compression_{index:06d}", METHOD_NAMES[method])


def validate_images(archive, callback):
    """Validate the complete NPY payload/ZIP CRC without retaining image arrays."""
    with archive.open("mask.npy") as source:
        shape, _, dtype = image_header(source)
        total = prod(shape) * dtype.itemsize
        frame_bytes = prod(shape[1:]) * dtype.itemsize
        copied = 0
        callback(0, shape[0])
        while copied < total:
            size = min(BUFFER_BYTES, total - copied)
            copy_bytes(source, None, size)
            copied += size
            callback(min(shape[0], copied // frame_bytes), shape[0])
        if source.read(1):
            raise ValueError("Unexpected trailing bytes in mask array")


def pack_video_record(index, arguments, directory, callback):
    path, record = arguments
    signature = Path(path).stat()
    info = inspect_record(path, record)
    part = Path(directory) / f"{index:06d}.npz.part"
    try:
        with ZipFile(path) as source, ZipFile(part, "w", compression=ZIP_DEFLATED, compresslevel=9) as output:
            opened = os.fstat(source.fp.fileno())
            if (opened.st_size, opened.st_mtime_ns) != (signature.st_size, signature.st_mtime_ns):
                raise ValueError(f"Input video export changed during processing: {path}")
            with record_codec(output, info["method"]):
                if source.getinfo("mask.npy").compress_type in METHOD_NAMES:
                    validate_images(source, callback)
                    append_entry(output, source, "mask.npy", f"data_{index:06d}.npy")
                else:
                    # Legacy uncompressed NPZs still get lossless bzip2 encoding.
                    callback(0, info["shape"][0])
                    copy_images(source, "mask", output, f"data_{index:06d}", callback=callback)
                write_record_metadata(output, index, info["metadata"], info["method"])
        return index, part, 0.
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def write_video_records(output, files, workers, progress, staging_dir=None):
    records = sorted(files)
    video_metadata(output, records)
    directory = Path(staging_dir).resolve() if staging_dir is not None else (Path(output.filename).parent if output.filename is not None else None)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        parts = resources.enter_context(tempfile.TemporaryDirectory(prefix=".openworm-video-build-", dir=directory))
        jobs = ((record, (files[record], record)) for record in records)
        release = poll = None
        if staging_dir is not None:
            from openworm.utils.openworm_archive_staging import StagedJobs
            feed = resources.enter_context(StagedJobs(jobs, staging_dir, workers, progress))
            jobs, release, poll = feed, feed.release, feed.poll
        completed = packed_records(pack_video_record, jobs, workers, parts, progress, release, poll)
        try:
            for index, part, _ in completed:
                record = records[index]
                try:
                    append_archive(output, part)
                finally:
                    part.unlink(missing_ok=True)
                progress.set_postfix_str(f"video: {record[0]}/{record[1]}", refresh=False)
                progress.update(1)
        finally:
            completed.close()


def build_video_dataset(data_dir, workers=1, staging_dir=None, output_dir=None):
    data_dir = Path(data_dir)
    files = video_files(data_dir)
    if not files:
        raise ValueError("No mask NPZs found in video/<strain>/record_*.npz")
    output_dir = Path(output_dir) if output_dir is not None else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "openworm_video_dataset.npz"
    temporary = destination.with_suffix(".npz.part")
    try:
        with total_progress(len(files)) as progress, ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as output:
            write_video_records(output, files, workers, progress, staging_dir)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {destination}")
    return destination
