"""Keep records longer than 890 s and trim to the shortest retained record.

Run from the repository root:
    python openworm/openworm_trim.py
    python openworm/openworm_trim.py --data-dir D:/data
    python openworm/openworm_trim.py --workers 4
    python openworm/openworm_trim.py --dataset eigenworms --workers 4
    python openworm/openworm_trim.py --dataset video --data-dir D:/data --input-dir D:/data/openworm --workers 4
    python openworm/openworm_trim.py --dataset matlab_features_new --workers 4 --frame-workers 3 --pool-type threads
    python openworm/openworm_trim.py --dataset matlab_features_new --input-dir D:/data/openworm --video-dir "E:/nematode classification/openworm" --check-alignment
    python openworm/openworm_trim.py --output-dir D:/data/trimmed

Reads the skeleton_angles, manual_features and eigenworms dataset NPZs from
D:/data by default. Writes openworm_<name>_trimmed_890_dataset.npz alongside
the inputs (or in --output-dir). Original datasets are not modified.

Selection is strictly >890 seconds: more than 22,250 samples at 25 Hz.
All retained records are trimmed to the shortest retained record's sample count.
For the original three datasets, keeps samples from the start, including any
interpolated/zeroed values, without resampling or changing feature values.
Labels, record IDs and feature names are preserved; record_keys are renumbered
consecutively. MATLAB alignment/rebuilding is described below.
The three source datasets must have matching strain/record pairs and sample
counts. All three outputs therefore contain the same records in the same order.
--dataset defaults to all. Select skeleton_angles, manual_features or eigenworms
to read and replace only that dataset, without requiring the other inputs.
Select matlab_features_new to trim openworm_matlab_features_new_dataset.npz
(16 features) into openworm_matlab_features_new_trimmed_890_dataset.npz.
This mode requires all three reference archives, validates their matching record
lengths, and uses their retained records and common trim length. --input-dir is
the original skeleton preprocessing input (default: <data-dir>/openworm).
--video-dir holds the videos used for MATLAB extraction (default: --input-dir).
Source HDF5 timestamps are required: NPZ/CSV relative times cannot identify the
acquisition origin. Do not point --input-dir at a different copy with different
adjacent-video availability: that changes the skeleton pipeline's clock choice.
Exact coincident MATLAB rows are selected by acquisition time. Fractionally
shifted/different-clock grids are rebuilt from the original MATLAB observations
on the reference grid, requiring MATLAB + Image Processing Toolbox. This can be
much slower than trimming. A separate pool runs up to four MATLAB sessions,
independently of --workers, and immediately starts the next queued rebuild when
a slot opens. --workers controls reading/alignment. Fast records keep processing
while MATLAB is busy. Queued work is bounded to limit memory use.
--frame-workers (default 1) enables bounded parfor image processing inside each
rebuild session; values above 1 require Parallel Computing Toolbox. --pool-type
is threads (default, shared memory) or processes, as in the extraction script.
Up to 4 * --frame-workers MATLAB frame workers can run concurrently.
Active rebuilds have frame bars showing MATLAB startup, calculation and Python
resampling. The overall bar counts records written as they finish and also shows
active/completed rebuild counts. Frame updates follow MATLAB's existing progress
messages (normally every 1,000 frames); elapsed times refresh between messages.
No interpolation of already-resampled feature rows is used. --check-alignment
validates all retained records and reports how many require rebuilding, without
writing output or launching MATLAB. --matlab can specify the executable.
All mode still processes the original three datasets. Other individual modes
retain their existing independent selection and trim behavior.

--dataset video reads openworm_video_dataset.npz, validates the same three
reference archives, and keeps their retained records/order and common length.
Each output record stays 3D (reference samples, height, width) in its original
dtype with its source lossless codec. Source videos need no MATLAB rebuild.
The reference HDF5 timestamps/clock determine acquisition targets. Since raw
video is usually ~30 Hz and skeletons are resampled at 25 Hz, nearest mode selects
one real mask image per reference sample without interpolating pixels. It saves
source_index_<index>, frame_number_<index>, actual time_s_<index>, target_time_s,
reference_frame_number and alignment_error_s per record. The common time_s array
is the 25 Hz reference grid. Nearest ties select the earlier frame; repeated
source rows can be selected if needed. Like MATLAB preprocessing, acquisition
frames are expanded onto their original clock and only gaps strictly shorter
than 0.25 s are filled. Video uses nearest original pixels for those short gaps;
long or unbracketed gaps get black frames. Per-record frame_status is real,
interpolated (short acquisition gap), or zeroed; black frames have source_index
and frame_number -1, and actual time_s/alignment_error_s NaN. Target times remain
on the reference grid. Nearest mode has no fixed error cutoff:
slower cameras and missing acquisition frames can cause larger offsets, which
are saved per sample and summarized in the output. --max-frame-error explicitly
sets a strict limit on selected real images in seconds. Conflicting clocks
remain errors; unavailable coverage produces black frames. Use
--video-alignment exact to reject non-coincident timestamps (tolerance 1e-8s).
--check-alignment also validates video without writing images. Video archives
must come from the mask exporter, not full_data. Embedded timestamps are used;
missing timing metadata can be recovered from source_file or --video-dir.
--workers runs independent video processes for alignment, decompression, frame
selection and lossless compression. The parent merges prepared compressed ZIP
entries without re-encoding pixels. Buffers and in-flight compressed records are
bounded; temporary compressed parts are cleaned up on success/failure.
--video-source records trims directly from video/<strain>/record_*.npz, without
creating a combined untrimmed dataset. Already-aligned full stacks retain their
compressed bytes; other stacks are reselected and losslessly recompressed.
archive uses an existing combined NPZ.
hdf5 reads original /mask images directly from --video-dir, needs no per-record
NPZs or untrimmed video archive, and uses lossless bzip2 level 9. auto prefers
per-record exports, then a combined archive, then HDF5 inputs. Only the final
openworm_video_trimmed_890_dataset.npz is saved under --output-dir.
--staging-dir stages one input at a time on the selected SSD while workers run.
For combined archives, only each record's compressed entries are staged, never
the whole archive. At most workers+1 staged inputs and a bounded set of worker
parts are held. Copies are deleted after worker completion; originals and the
configured staging folder are preserved. The reference HDF5 clock files remain
at --input-dir. Final .npz.part output is written directly in --output-dir and
renamed after success, with no second complete copy on the SSD.
All mode retains its existing original-three-dataset behavior.

--workers defaults to 1. Higher values read and trim records in parallel, with
one archive handle per worker and at most one queued record per worker. Archive
For the original three datasets, compression/writing stays in the parent.
Video compression runs in worker processes as described above. MATLAB records are
written as they finish under their original keys; metadata, labels and logical
record order are preserved. The other datasets retain ordered processing.
Shows a total progress bar with ETA. All outputs
are written to temporary files first and replaced after successful processing.
Allow disk space for three new datasets; compression can take substantial time.
"""

# Allow both `python openworm/<script>.py` and package imports/module execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from contextlib import ExitStack
from functools import partial
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from tqdm import tqdm

from openworm.openworm_dataset import write_array, write_metadata, PROGRESS_FORMAT, ordered_results

ALL_DATASETS = (("skeleton_angles", 48), ("manual_features", 47), ("eigenworms", 6))
DATASETS = (*ALL_DATASETS, ("matlab_features_new", 16))
MIN_SAMPLES = 22_250
SAMPLE_SECONDS = 0.04
MATLAB_WORKERS = 4


def inspect_dataset(path, feature_count):
    with np.load(path, allow_pickle=False) as archive:
        labels, ids, keys, names = (archive[key] for key in
                                    ("labels", "record_ids", "record_keys", "feature_names"))
    if (any(array.ndim != 1 for array in (labels, ids, keys, names))
            or not len(keys) or not len(labels) == len(ids) == len(keys)
            or len(names) != feature_count or len(set(names)) != feature_count
            or len(set(keys)) != len(keys)):
        raise ValueError(f"Invalid archive metadata: {path}")
    records = list(zip(map(str, labels), map(str, ids)))
    if len(set(records)) != len(records):
        raise ValueError(f"Duplicate strain/record pairs: {path}")
    sizes = {}
    with ZipFile(path) as archive:
        for record, key in tqdm(zip(records, keys), total=len(keys),
                                desc=f"Checking {path.stem}", unit="record"):
            with archive.open(str(key) + ".npy") as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError(f"Unsupported NPY version: {path}/{key}")
            if (len(shape) != 2 or shape[0] < 1 or shape[1] != feature_count
                    or dtype.hasobject or dtype.kind not in "fiu"):
                raise ValueError(f"Invalid record array: {path}/{key}")
            sizes[record] = shape[0]
    return names, dict(zip(records, map(str, keys))), sizes


_worker_archive = None
_worker_path = None


def read_trimmed_record(source, key, expected_samples, target_samples):
    """Keep each worker's input archive open across records."""
    global _worker_archive, _worker_path
    if _worker_path != source:
        if _worker_archive is not None:
            _worker_archive.close()
        _worker_archive = np.load(source, allow_pickle=False)
        _worker_path = source
    return trim_record(_worker_archive, source, key, expected_samples, target_samples)


def trim_record(archive, source, key, expected_samples, target_samples):
    values = archive[key]
    if len(values) != expected_samples or not np.isfinite(values).all():
        raise ValueError(f"Invalid or changed record: {source}/{key}")
    return values[:target_samples].copy()


def prepare_aligned_record(source, key, expected_samples, target_samples, reference_samples,
                           features, video):
    from openworm.utils.openworm_alignment import alignment_plan
    indices, times = alignment_plan(features, video, reference_samples,
                                    expected_samples, target_samples)
    if indices is None:
        return None, (video, times)
    values = read_trimmed_record(source, key, expected_samples, expected_samples)
    return values[indices].copy(), None


def run_rebuild(video, times, matlab=None, frame_workers=1, pool_type="threads",
                progress_queue=None):
    from openworm.utils.openworm_alignment import rebuild_record
    def report(stage, done, total):
        if progress_queue is not None:
            progress_queue.put((str(video), stage, done, total))
    try:
        return rebuild_record(video, times, matlab, frame_workers=frame_workers,
                              pool_type=pool_type, progress_callback=report)
    except BaseException:
        report("failed", 0, 0)
        raise


def aligned_record(source, key, expected_samples, target_samples, reference_samples,
                   features, video, matlab, frame_workers=1, pool_type="threads",
                   progress_queue=None):
    values, arguments = prepare_aligned_record(source, key, expected_samples, target_samples,
                                               reference_samples, features, video)
    if arguments is not None:
        return run_rebuild(*arguments, matlab=matlab, frame_workers=frame_workers,
                           pool_type=pool_type, progress_queue=progress_queue)
    return values


def alignment_files(input_dir, video_dir, records):
    result = {}
    for strain, record in records:
        candidates = list((input_dir / strain / record).glob("*_features.hdf5"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one reference features HDF5: {input_dir / strain / record}")
        features = candidates[0]
        video = video_dir / strain / record / features.name.replace("_features.hdf5", ".hdf5")
        if not video.is_file():
            raise ValueError(f"Missing MATLAB source video: {video}; set --video-dir")
        result[(strain, record)] = (features, video)
    return result


def trim_datasets(data_dir, output_dir=None, workers=1, dataset="all", input_dir=None,
                  video_dir=None, matlab=None, check_alignment=False, frame_workers=1,
                  pool_type="threads", video_alignment="nearest", max_frame_error=None,
                  staging_dir=None, video_source="auto"):
    global _worker_archive, _worker_path
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if dataset not in ("all", *(name for name, _ in DATASETS), "video"):
        raise ValueError(f"Unknown dataset: {dataset}")
    if frame_workers < 1:
        raise ValueError("frame_workers must be at least 1")
    if pool_type not in ("threads", "processes"):
        raise ValueError("Unknown pool type")
    if frame_workers > 1 and dataset != "matlab_features_new":
        raise ValueError("--frame-workers greater than 1 requires --dataset matlab_features_new")
    if dataset == "video":
        from openworm.utils.openworm_video_trim import trim_video
        return trim_video(data_dir, output_dir, workers, input_dir, video_dir,
                          check_alignment, video_alignment, max_frame_error, staging_dir, video_source)
    selected = list(ALL_DATASETS) if dataset == "all" else [item for item in DATASETS if item[0] == dataset]
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else data_dir
    sources = [data_dir / f"openworm_{name}_dataset.npz" for name, _ in selected]
    metadata = [inspect_dataset(path, count) for path, (_, count) in zip(sources, selected)]
    matlab_mode = dataset == "matlab_features_new"
    if check_alignment and not matlab_mode:
        raise ValueError("--check-alignment requires --dataset matlab_features_new")
    reference = ([inspect_dataset(data_dir / f"openworm_{name}_dataset.npz", count)
                  for name, count in ALL_DATASETS] if matlab_mode else metadata)
    sizes = reference[0][2]
    if any(current[2] != sizes for current in reference[1:]):
        raise ValueError("Source datasets must have identical strain/record pairs and sample counts")
    records = sorted(record for record, frames in sizes.items() if frames > MIN_SAMPLES)
    if not records:
        raise ValueError("No records are longer than 890 seconds; outputs were not changed")
    target_samples = min(sizes[record] for record in records)
    files = {}
    if matlab_mode:
        from openworm.openworm_matlab_features import FEATURES
        if list(metadata[0][0]) != FEATURES:
            raise ValueError("Unexpected MATLAB feature names/order; rebuild would be ambiguous")
        missing = set(records) - metadata[0][2].keys()
        if missing:
            raise ValueError(f"MATLAB archive is missing {len(missing)} reference records")
        input_dir = Path(input_dir).resolve() if input_dir is not None else data_dir / "openworm"
        video_dir = Path(video_dir).resolve() if video_dir is not None else input_dir
        files = alignment_files(input_dir, video_dir, records)
        if check_alignment:
            from openworm.utils.openworm_alignment import alignment_plan
            rebuild = 0
            for record in tqdm(records, desc="Checking frame alignment", unit="record"):
                indices, _ = alignment_plan(*files[record], sizes[record],
                                            metadata[0][2][record], target_samples)
                rebuild += indices is None
            print(f"Reference selection: {len(records):,} records, {target_samples:,} samples each. "
                  f"Exact row selection: {len(records) - rebuild:,}; MATLAB rebuild: {rebuild:,}.")
            return []
    destinations = [output_dir / f"openworm_{name}_trimmed_890_dataset.npz" for name, _ in selected]
    temporary = [path.with_suffix(".npz.part") for path in destinations]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Keeping {len(records):,}/{len(sizes):,} records; trimming each to "
          f"{target_samples:,} samples ({target_samples * SAMPLE_SECONDS:.2f} seconds).", flush=True)
    try:
        with tqdm(total=len(records) * len(selected), desc="Trimming datasets", unit="record",
                  smoothing=0, bar_format=PROGRESS_FORMAT) as progress, ExitStack() as processing:
            rebuilds = None
            if matlab_mode:
                from openworm.utils.openworm_progress import RebuildProgress
                rebuilds = processing.enter_context(RebuildProgress(progress, MATLAB_WORKERS))
            for source, temp, (names, keys, source_sizes) in zip(sources, temporary, metadata):
                with np.load(source, allow_pickle=False) as archive, \
                        ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=6) as output:
                    write_metadata(output, names, records)
                    if matlab_mode:
                        write_array(output, "time_s", np.arange(target_samples) * SAMPLE_SECONDS)
                        write_array(output, "alignment_reference", "skeleton_angles")
                        from openworm.utils.openworm_scheduling import staged_results
                        jobs = ((source, keys[record], source_sizes[record], target_samples,
                                 sizes[record], *files[record]) for record in records)
                        rebuild = partial(run_rebuild, matlab=matlab, frame_workers=frame_workers,
                                          pool_type=pool_type, progress_queue=rebuilds.queue)
                        completed = staged_results(prepare_aligned_record, jobs, workers,
                                                   rebuild, MATLAB_WORKERS)
                    elif workers == 1:
                        completed = (trim_record(archive, source, keys[record], sizes[record], target_samples)
                                     for record in records)
                    else:
                        jobs = ((source, keys[record], sizes[record], target_samples) for record in records)
                        completed = ordered_results(read_trimmed_record, jobs, workers)
                    processing.callback(completed.close)
                    indexed = completed if matlab_mode else enumerate(completed)
                    for i, values in indexed:
                        record = records[i]
                        write_array(output, f"data_{i:06d}", values)
                        if rebuilds is not None:
                            rebuilds.wrote_record(record)
                        else:
                            progress.set_postfix_str(f"{source.stem}: {record[0]}/{record[1]}", refresh=False)
                        progress.update(1)
        for temp, destination in zip(temporary, destinations):
            os.replace(temp, destination)
    finally:
        if _worker_archive is not None:
            _worker_archive.close()
            _worker_archive = _worker_path = None
        for temp in temporary:
            temp.unlink(missing_ok=True)
    for destination in destinations:
        print(f"Saved {destination}")
    return destinations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("D:/data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1,
                        help="Record processes, including video alignment/compression; MATLAB rebuilds use a separate four-session pool (default: 1)")
    parser.add_argument("--dataset", choices=("all", *(name for name, _ in DATASETS), "video"),
                        default="all", help="Trim the three aligned datasets or one selected dataset (default: all)")
    parser.add_argument("--input-dir", type=Path, help="Original skeleton HDF5 input (default: <data-dir>/openworm)")
    parser.add_argument("--video-dir", type=Path, help="MATLAB source video root (default: --input-dir)")
    parser.add_argument("--matlab", type=Path, help="MATLAB executable for rebuilding shifted grids")
    parser.add_argument("--frame-workers", type=int, default=1,
                        help="MATLAB frame workers per rebuild, matlab_features_new only (default: 1)")
    parser.add_argument("--pool-type", choices=("threads", "processes"), default="threads",
                        help="MATLAB frame pool implementation (default: threads)")
    parser.add_argument("--check-alignment", action="store_true", help="Check MATLAB or video frame alignment without writing/rebuilding")
    parser.add_argument("--video-alignment", choices=("nearest", "exact"), default="nearest",
                        help="Video alignment: select nearest real mask frame (default) or require exact timestamp match")
    parser.add_argument("--max-frame-error", type=float, default=None,
                        help="Optional strict nearest-frame error limit in seconds (default: use nearest available frame)")
    parser.add_argument("--staging-dir", type=Path, help="SSD folder for one-at-a-time video staging and bounded compressed worker parts")
    parser.add_argument("--video-source", choices=("auto", "records", "archive", "hdf5"), default="auto",
                        help="auto prefers per-record NPZs; hdf5 reads /mask directly and needs no untrimmed NPZ dataset")
    args = parser.parse_args()
    try:
        trim_datasets(args.data_dir, args.output_dir, args.workers, args.dataset,
                      args.input_dir, args.video_dir, args.matlab, args.check_alignment,
                      args.frame_workers, args.pool_type,
                      video_alignment=args.video_alignment, max_frame_error=args.max_frame_error,
                      staging_dir=args.staging_dir, video_source=args.video_source)
    except (OSError, ValueError, KeyError, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
