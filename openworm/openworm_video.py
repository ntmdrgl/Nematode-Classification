"""Save every /mask image from each record's video HDF5 as a raw 3D stack.

Run from the repository root:
    python openworm/openworm_video.py --input-dir "E:/nematode classification/openworm" --output-dir D:/data/video --staging-dir C:/openworm-staging --workers 24 --compression bzip2
    python openworm/openworm_video.py --input-dir "E:/nematode classification/openworm" --output-dir D:/data/video --workers 4 --compression bzip2

Default input: data/openworm. Default output: data/video/<strain>/record_*.npz.
Each archive contains mask, a 3D (frames, height, width) array with every stored
/mask image in its original order, shape and dtype (normally uint8, 480 x 640).
There is no rotation, resizing, smoothing, temporal resampling or frame removal.
Only /mask is read for images: /full_data is never used as a fallback, and
_features.hdf5 files are excluded. MATLAB is not required.

Metadata includes source_dataset="mask", strain, record_id and source_file.
Available /mask fps and expected_fps attributes are retained. When available,
frame_number and time_s store the video's original /timestamp/raw and
/timestamp/time entries, one per mask image. Duplicate acquisition entries are
preserved along with their images; trailing timestamps without images are
omitted. These timestamps are not rebased to zero or generated from nominal FPS.
Missing timestamps do not prevent image export; present arrays must cover the
images. Later time alignment must validate their clock before using them.

Load images with numpy.load(path, allow_pickle=False)["mask"]. --workers controls
concurrent record processes (CLI default 24). Each process owns its HDF5 handles
and codecs, so h5py's global thread lock cannot serialize reads across workers.
Arrays are never sent between processes. Progress events are throttled to four
per second per worker, with start/final updates always delivered. Per-record
frame bars and a total record bar
report progress and ETA. Reads are bounded to about 32 MiB per worker (or one
image if larger), and lossless ZIP compression is streamed. A full mask stack
can be much larger than the old sparse full_data export.

--staging-dir selects a folder on an internal SSD (default: system temporary
folder/openworm-video-staging). One copier reads one original video file at a
time, sequentially, while up to --workers processes export completed copies.
Staged inputs are bounded to workers+1 files including active/copying/queued
files (25 with 24 workers). Only the video HDF5 is copied, never _features.hdf5.
A byte-progress bar reports staging alongside per-record image bars. Each copy
is deleted as soon as its worker finishes, including failed records. Completed
exports are skipped before copying. Outputs retain the original source_file,
strain and record IDs; no metadata points at deleted staged paths. A unique
per-run subfolder is cleaned up on normal completion, errors and handled Ctrl+C;
the configured staging folder and original source files are never deleted.
A forced process kill can leave that run's staging subfolder behind.
--no-staging keeps direct input reads. Use an SSD path with space for the bounded
staged inputs; NPZ output paths remain controlled separately by --output-dir.

--compression bzip2 is the CLI default and uses lossless level 9. Optional
--compression best writes complete NPZ candidates using Deflate level
9, bzip2 level 9 and ZIP LZMA, then keeps the smallest completed archive, including
metadata. This finds the smallest of these tested NumPy-compatible codecs for
each full record, not a universal optimum over every possible lossless format.
All candidates use exactly the source dtype and pixel bytes. HDF5 image blocks
are read once and sent to each codec; memory remains bounded. Three candidates
require extra CPU, codec buffers and temporary disk space. For a single pass,
use --compression bzip2, deflate or lzma. ZIP LZMA uses Python's ZIP preset; its
compression level is not configurable. NumPy loads every option normally.

Existing mask exports with the requested compression strategy/version are
skipped. Switching strategy regenerates them. Earlier exports (including the
old Deflate-6 mask files), full_data exports and unmarked/invalid archives are
regenerated at the same output path;
the previous file is replaced atomically only after the mask export succeeds.
Records without video HDF5s are skipped. Missing/invalid mask or multiple videos
fail that record; other records continue and failures yield a nonzero exit code.
Each record uses a unique temporary .part file, cleaned up on failure. Interrupted
records restart; completed mask records are skipped.
"""

# Allow both `python openworm/<script>.py` and package imports/module execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
import os
import multiprocessing
from pathlib import Path
from queue import Empty
import tempfile
import time
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, BadZipFile, ZipFile

import h5py
import numpy as np
from tqdm import tqdm

from openworm.openworm_dataset import PROGRESS_FORMAT, write_array

ROOT = Path(__file__).resolve().parents[1]
READ_BYTES = 32 * 1024 ** 2
CODECS = {"deflate": (ZIP_DEFLATED, 9), "bzip2": (ZIP_BZIP2, 9), "lzma": (ZIP_LZMA, None)}
COMPRESSION_VERSION = 1


def is_mask_export(path, compression="best"):
    """Do not mistake old sparse full_data outputs for completed mask exports."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return ("mask" in archive.files and "full_data" not in archive.files
                    and "source_dataset" in archive.files
                    and archive["source_dataset"].shape == ()
                    and archive["source_dataset"].item() == "mask"
                    and "compression_strategy" in archive.files
                    and archive["compression_strategy"].item() == compression
                    and "compression_version" in archive.files
                    and archive["compression_version"].item() == COMPRESSION_VERSION)
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return False


def process_record(record, output_dir, progress_callback=None, compression="best",
                   video_file=None, source_file=None):
    """Return (status, output path, image count), publishing one complete record."""
    if compression not in ("best", *CODECS):
        raise ValueError(f"Unknown lossless compression: {compression}")
    record, output_dir = Path(record), Path(output_dir)
    output = output_dir / record.parent.name / (record.name + ".npz")
    if output.is_file() and is_mask_export(output, compression):
        return "skipped", output, 0
    if video_file is None:
        videos = sorted(path for path in record.glob("*.hdf5")
                        if path.is_file() and not path.name.lower().endswith("_features.hdf5"))
        if not videos:
            return "skipped", output, 0
        if len(videos) != 1:
            raise ValueError(f"Expected exactly one video HDF5 in {record}; found {len(videos)}")
        video = videos[0]
    else:
        # Staged workers never discover/open files on the original source drive.
        video = Path(video_file)
    temporary = {}
    try:
        with h5py.File(video, "r") as source:
            if "mask" not in source or not isinstance(source["mask"], h5py.Dataset):
                raise ValueError(f"Missing /mask image dataset: {video}")
            images = source["mask"]
            if images.ndim != 3 or any(size < 1 for size in images.shape) or images.dtype.kind not in "buif":
                raise ValueError(f"Expected nonempty numeric 3D /mask: {video}; "
                                 f"shape={images.shape}, dtype={images.dtype}")
            count = images.shape[0]
            block_frames = max(1, READ_BYTES // (int(np.prod(images.shape[1:])) * images.dtype.itemsize))
            if progress_callback is not None:
                progress_callback(0, count)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Metadata is independent of compression. Read/validate once.
            metadata = {"source_dataset": "mask", "strain": record.parent.name,
                        "record_id": record.name, "source_file": str(video.resolve()) if source_file is None else str(source_file),
                        "compression_strategy": compression,
                        "compression_version": COMPRESSION_VERSION}
            for attribute in ("fps", "expected_fps"):
                if attribute in images.attrs:
                    value = np.asarray(images.attrs[attribute])
                    if value.ndim != 0 or value.dtype.kind not in "fiu":
                        raise ValueError(f"Invalid /mask {attribute}: {video}")
                    metadata[attribute] = value
            for source_key, output_key in (("timestamp/raw", "frame_number"),
                                           ("timestamp/time", "time_s")):
                if source_key in source:
                    values = np.asarray(source[source_key][:]).reshape(-1)
                    if len(values) < count:
                        raise ValueError(f"/{source_key} does not cover every mask image: {video}")
                    metadata[output_key] = values[:count]
            codecs = tuple(CODECS) if compression == "best" else (compression,)
            with ExitStack() as archives:
                candidates = {}
                for name in codecs:
                    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + "." + name + ".",
                                                     suffix=".part", delete=False) as temp:
                        temporary[name] = Path(temp.name)
                    method, level = CODECS[name]
                    candidates[name] = archives.enter_context(ZipFile(
                        temporary[name], "w", compression=method, compresslevel=level))
                with ExitStack() as streams:
                    destinations = [streams.enter_context(archive.open("mask.npy", "w", force_zip64=True))
                                    for archive in candidates.values()]
                    for destination in destinations:
                        np.lib.format.write_array_header_2_0(destination, {
                            "descr": np.lib.format.dtype_to_descr(images.dtype),
                            "fortran_order": False, "shape": images.shape})
                    for start in range(0, count, block_frames):
                        end = min(count, start + block_frames)
                        block = np.ascontiguousarray(images[start:end])
                        # Decode each HDF5 block once; every candidate receives
                        # identical original bytes. No quantization or packing.
                        raw_bytes = memoryview(block).cast("B")
                        for destination in destinations:
                            destination.write(raw_bytes)
                        if progress_callback is not None:
                            progress_callback(end, count)
                for name, archive in candidates.items():
                    for key, value in metadata.items():
                        write_array(archive, key, value)
                    write_array(archive, "compression", name)
            sizes = {name: path.stat().st_size for name, path in temporary.items()}
            selected = min(codecs, key=sizes.get)
        os.replace(temporary[selected], output)
        return "saved", output, count
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def process_job(job_id, slot, record, output_dir, compression, progress_queue,
                video_file=None, source_file=None):
    """One process owns its HDF5 handles and compressors; return only metadata."""
    last_update = 0.
    def report(done, total):
        nonlocal last_update
        now = time.monotonic()
        if done == 0 or done == total or now - last_update >= .25:
            progress_queue.put((job_id, slot, done, total))
            last_update = now
    if video_file is None:
        return process_record(record, output_dir, report, compression)
    return process_record(record, output_dir, report, compression,
                          video_file=video_file, source_file=source_file)


def run_records(records, output_dir, workers=1, compression="best", staging_dir=None):
    """Bounded processes bypass h5py's thread-global HDF5 lock.

    Only paths/settings and small progress events cross process boundaries;
    image arrays and compression remain inside the process handling a record.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if compression not in ("best", *CODECS):
        raise ValueError(f"Unknown lossless compression: {compression}")
    if staging_dir is not None:
        from openworm.utils.openworm_video_staging import run_staged
        return run_staged(records, output_dir, workers, compression, staging_dir)
    records = list(map(Path, records))
    counts = dict(saved=0, skipped=0, failed=0)
    active = {}
    with tqdm(total=len(records), desc="Video records", unit="record", position=0,
              smoothing=0, bar_format=PROGRESS_FORMAT) as overall:
        def start(job_id, slot, record):
            bar = tqdm(total=0, desc=f"{record.parent.name}/{record.name}", unit="image",
                       position=slot + 1, leave=False, smoothing=0,
                       bar_format=PROGRESS_FORMAT, disable=overall.disable)
            active[slot] = (job_id, bar)

        def advance(job_id, slot, done, total):
            if slot in active and active[slot][0] == job_id:
                bar = active[slot][1]
                bar.total = total
                bar.update(max(0, done - bar.n))

        def finish(record, operation):
            try:
                status, output, count = operation()
                if status == "saved":
                    with np.load(output, allow_pickle=False) as saved:
                        codec = saved["compression"].item()
                    tqdm.write(f"Saved {output}: {count:,} mask images, {codec}, "
                               f"{output.stat().st_size / 1024**2:.2f} MiB")
                counts[status] += 1
            except Exception as error:
                counts["failed"] += 1
                tqdm.write(f"FAILED {record}: {error}")
            overall.update(1)

        def close_slot(slot):
            _, bar = active.pop(slot)
            bar.close()

        try:
            if workers == 1:
                for job_id, record in enumerate(records):
                    start(job_id, 0, record)
                    finish(record, lambda: process_record(
                        record, output_dir, lambda done, total: advance(job_id, 0, done, total), compression))
                    close_slot(0)
            elif records:
                context = multiprocessing.get_context("spawn")
                # Shared proxy carries throttled progress only, never image data.
                with context.Manager() as manager:
                    events = manager.Queue()
                    pool = ProcessPoolExecutor(max_workers=min(workers, len(records)), mp_context=context)
                    pending = {}
                    jobs = iter(enumerate(records))
                    def submit(slot):
                        job = next(jobs, None)
                        if job is not None:
                            job_id, record = job
                            start(job_id, slot, record)
                            future = pool.submit(process_job, job_id, slot, record, output_dir, compression, events)
                            pending[future] = (job_id, slot, record)
                    def drain_events():
                        while True:
                            try:
                                advance(*events.get_nowait())
                            except Empty:
                                break
                    try:
                        for slot in range(min(workers, len(records))):
                            submit(slot)
                        while pending:
                            done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                            drain_events()
                            for future in done:
                                job_id, slot, record = pending.pop(future)
                                finish(record, future.result)
                                close_slot(slot)
                                submit(slot)
                    finally:
                        for future in pending:
                            future.cancel()
                        pool.shutdown(wait=True, cancel_futures=True)
                        drain_events()
        finally:
            for _, bar in active.values():
                bar.close()
    print(f"Saved {counts['saved']}; skipped {counts['skipped']}; failed {counts['failed']}.")
    return 1 if counts["failed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/openworm",
                        help="Dataset root, strain folder, or one record directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/video")
    parser.add_argument("--workers", type=int, default=24, help="Concurrent record processes (default: 24)")
    parser.add_argument("--compression", choices=("best", *CODECS), default="bzip2",
                        help="best compares complete Deflate-9, bzip2-9 and LZMA NPZs and keeps the smallest; choose a codec for single-pass compression")
    staging = parser.add_mutually_exclusive_group()
    staging.add_argument("--staging-dir", type=Path,
                         default=Path(tempfile.gettempdir()) / "openworm-video-staging",
                         help="Internal SSD staging folder (default: system temporary folder/openworm-video-staging)")
    staging.add_argument("--no-staging", action="store_true", help="Read video files directly without SSD staging")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    records = ([args.input_dir] if args.input_dir.name.startswith("record_") else
               sorted(path for path in args.input_dir.rglob("record_*") if path.is_dir()))
    if not records:
        parser.error("No record_* directories found")
    return run_records(records, args.output_dir, args.workers, args.compression,
                       staging_dir=None if args.no_staging else args.staging_dir)


if __name__ == "__main__":
    raise SystemExit(main())
