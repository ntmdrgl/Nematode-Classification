# Quick start: two metadata CSVs to five trimmed datasets

Run these PowerShell commands **from the repository root**. Start with these two
existing files; you do not need to fetch Zenodo metadata again:

- `openworm/openworm_movement_zenodo_metadata.csv`
- `openworm/openworm_parsed_all_fields.csv`

The CSVs describe the recordings; they do not contain the images or feature
measurements. Download the original HDF5s once, then let `openworm_pipeline.py`
extract, align, trim, and save the five final datasets directly.

## 1. Set up the environment

Skip this block if your virtual environment and dependencies are already ready.
Python 3.12+ and MATLAB with Image Processing Toolbox are required. The commands
below use six MATLAB frame workers per session, which also requires Parallel
Computing Toolbox; use `--frame-workers 1` if you do not have that toolbox.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy scipy h5py tqdm
```

## 2. Download the original HDF5s using the two CSVs

```powershell
python openworm/openworm_download.py --metadata-csv openworm/openworm_movement_zenodo_metadata.csv --parsed-csv openworm/openworm_parsed_all_fields.csv --output-dir "E:/nematode classification/openworm" --file-type video-features --workers 4
```

This applies the existing record-selection filters and downloads both original
mask-video `.hdf5` files and their companion `_features.hdf5` files into
`<strain>/record_*/`. Rerunning resumes `.part` downloads and skips size-verified
completed files. Skip this step if both HDF5 types are already downloaded.

## 3. Generate only the five trimmed datasets

```powershell
python openworm/openworm_pipeline.py --dataset all --hdf5-dir "E:/nematode classification/openworm" --data-dir D:/data --output-dir "E:/nematode classification" --staging-dir C:/openworm-staging --workers 24 --matlab-workers 4 --frame-workers 6 --pool-type threads
```

That is the entire processing step. The pipeline reads the original HDF5s;
it does not call `openworm_download.py` or `openworm_dataset.py`. No separate
extraction commands, generated feature CSVs, `D:/data/video` exports, or untrimmed
dataset NPZs are needed. `--dataset all` here runs exactly `skeleton_angles`,
`manual_features`, `eigenworms`, `matlab_features_new`, and `video`.

Change the folder options as needed:

| Option | Purpose |
|---|---|
| `--hdf5-dir` | Original video and feature HDF5s; kept unchanged. |
| `--data-dir` | Small manifests and ZIP recovery journals in `.openworm-pipeline`; no untrimmed datasets. |
| `--output-dir` | Final trimmed archives and their resumable `.npz.part` files. |
| `--staging-dir` | Internal SSD for bounded HDF5 copies and temporary trimmed record chunks. |

The pipeline stages inputs one record at a time while up to 24 processing
workers run. Staged HDF5 copies are deleted after each worker finishes, and
trimmed chunks are deleted after merging. MATLAB uses up to four concurrent
sessions with six frame workers each; reduce the worker counts for your machine.
Use `--matlab PATH` if MATLAB is not found automatically.

The final output folder contains:

```text
openworm_skeleton_angles_trimmed_890_dataset.npz
openworm_manual_features_trimmed_890_dataset.npz
openworm_eigenworms_trimmed_890_dataset.npz
openworm_matlab_features_new_trimmed_890_dataset.npz
openworm_video_trimmed_890_dataset.npz
```

All five datasets use the same retained records and common length on the 25 Hz
skeleton timeline. Records must exceed 890 seconds; the pipeline computes the
retained list and shortest common length from the available original clocks.
Video stays a 3D `(frames, height, width)` array with the original pixel dtype and
lossless bzip2 compression. It reads `/mask`, not `/full_data`. Gaps strictly
shorter than 0.25 s use the nearest original image; longer or unbracketed gaps
use black frames. Black placeholders do not necessarily match rows zeroed by
skeleton-tracking or individual feature-calculation failures.

## Stop and resume

Press **Ctrl+C and wait for the script to exit**, then rerun the identical
pipeline command. Resume is automatic; there is no `--resume` flag.

- Matching completed datasets and committed records are skipped.
- Finished worker chunks are reused; unfinished records restart.
- You may change `--workers` without invalidating completed work.
- Keep the same folders, dataset/reference selection, original inputs, output
  `.npz.part` files, `D:/data/.openworm-pipeline`, and pending trimmed SSD chunks.
- Do not run concurrent pipelines targeting the same folders.

Recovery journals protect committed ZIP data when an append is interrupted.
An older/damaged partial with no readable ZIP directory or pipeline resume
metadata is preserved unchanged as `.npz.part.preserved-<unique ID>` before a
fresh checkpoint starts. Such a file cannot supply trusted committed records;
compatible finished worker chunks can still be reused. Valid checkpoints with
different input/settings signatures are left untouched and reported as errors.
No second complete video archive is created by normal checkpointing/merging.

## Only regenerate or resume video against an existing skeleton dataset

When your other datasets are already complete, use this command instead of the
all-datasets command. It launches no MATLAB and writes only the trimmed video
archive, with the same records, record order, and common frame count as the
explicit skeleton reference:

```powershell
python openworm/openworm_pipeline.py --dataset video --hdf5-dir "E:/nematode classification/openworm" --reference-skeleton "E:/nematode classification/openworm_skeleton_angles_trimmed_890_dataset.npz" --data-dir D:/data --output-dir "E:/nematode classification" --staging-dir C:/openworm-staging --workers 24
```

Your existing reference has **5,418 records and 22,257 frames per record**.
Rerun this same video-only command to resume it. Without an explicit reference,
video-only mode checks the output folder, then the data folder, for the trimmed
skeleton archive; if neither has one, it derives the selection from HDF5s.
Resume support described here belongs to `openworm_pipeline.py`; the older
`openworm_trim.py` video command does not checkpoint completed records.

---

## Folder layout

Command-line entry points remain directly under `openworm/`, so the commands
above are unchanged. Internal alignment, archive, staging, progress, scheduling,
and video-processing helpers live in `openworm/utils/`, alongside the two MATLAB
extraction implementations. The Python wrappers add that MATLAB folder to the
MATLAB search path automatically. Regression tests and auxiliary checks stay in
`openworm/etc/`; metadata CSVs remain under `openworm/`.

---

# OpenWorm CSV generation, strain counts and downloads

Build just one NPZ dataset:

```powershell
python openworm/openworm_dataset.py --dataset skeleton_angles
python openworm/openworm_dataset.py --dataset manual_features
python openworm/openworm_dataset.py --dataset eigenworms
python openworm/openworm_dataset.py --dataset matlab_features
```

Each mode requires only its own source CSV folder and replaces only its selected
NPZ file. Eigenworms uses the eigen-projection columns in `data/manual_features`.
Omit `--dataset`, or use `--dataset all`, to build all four with the existing
cross-dataset record/time alignment checks.

Process skeleton-angle or manual-feature records in parallel:

```powershell
python openworm/openworm_skeleton_angles.py --workers 4
python openworm/openworm_manual_features.py --workers 4
```

The default is one worker. Each worker handles one record in a separate process
and opens its own HDF5 files. Parallel runs show one combined total ETA without
competing per-worker progress bars; records can finish in a different order.
Only a worker-sized set of records is queued. Ctrl+C cancels queued work and
waits for active records to finish. CSV output matches sequential processing.
Start with 2–4 workers and consider CPU cores, RAM usage, and disk bandwidth.
Do not run two instances targeting the same output folders simultaneously.

`openworm_skeleton_angles.py` and `openworm_manual_features.py` show overall
record progress with elapsed time and estimated time remaining. The
top bar explicitly labels this `TOTAL ETA`: time to finish all remaining records,
using average elapsed time per completed record across the entire run. The current
strain/record is displayed alongside it. Stage progress bars show feature
processing (manual features) and CSV writing with their own ETAs. Overall ETA
becomes available after the first record finishes and is an estimate based on
completed records; different record lengths can change it.

Extract the original 16 MATLAB features from video HDF5 files:

```powershell
python openworm/openworm_matlab_features.py
python openworm/openworm_matlab_features.py --input-dir E:/openworm
```

The Python wrapper calls `openworm_matlab_features.m` through MATLAB's headless
batch mode. MATLAB and Image Processing Toolbox must be installed/licensed.
MATLAB is found on PATH or in `C:/Program Files/MATLAB`; use `--matlab PATH`
to override. Outputs go to `data/matlab_features/<strain>/record_*.csv` with
`time_s`, 16 feature columns, and one status column per feature. Records without
a video HDF5 are skipped. The companion `_features.hdf5` is also required for the
original centroid/endpoint coordinate calculations, not its features_timeseries.

The collection script's image processing and indexing are preserved. The exact
16 features/order come from `old/openworm_dataset_preprocessing.m`:
SkeletonLength, EndpointSeparation, CentroidDisplacementFS0,
Endpoint1DisplacementFS0, Endpoint2DisplacementFS0, CentroidTotalDisplacement,
Endpoint1TotalDisplacement, Endpoint2TotalDisplacement, Tortuosity, NumberOfBends,
BendLengthStd, WidthMean, AspectRatio, Area, Perimeter, OrientationAngle.
Displacements retain the legacy FPS scaling (including total displacement),
endpoint separation is recalculated from coordinates, and negative values are
clipped except orientation. FPS and absolute coordinates are not output features.
The original coordinate indexing by video row is intentionally retained, even
when dropped video frames exist; this preserves the old extraction algorithm.

Python replaces the old unlimited filling/trimming with the common 0.04-second
time grid, strict <0.25s gap interpolation and long/unbracketed-gap zeroing.
No Savitzky–Golay smoothing is applied to MATLAB features. MATLAB returns original-availability masks;
values filled solely by the old fillmissing are masked again before this step.
Dropped video-frame positions are inserted as missing samples. The orientation
angle uses its 180-degree period, and non-orientation outputs remain nonnegative.
For a short preview, use
`--max-frames 100 --output-dir data/matlab_features_preview`; leave this cap out
when generating the full dataset. No parallel MATLAB toolbox is required.

Evaluate real/interpolated/zeroed coverage:

```powershell
python openworm/openworm_evaluate.py
```

This saves `data/openworm_evaluation.csv`, with one row per strain and a total
row for each of the four NPZ datasets, with all four total rows first.
Counts and percentages measure feature
values: frames × 48 angles, × 47 manual features, × 6 eigen projections, or
× 16 MATLAB features. Each dataset uses its own available records; MATLAB may
have fewer records because video files are required for extraction.
Longer records contribute proportionally; totals are not averages of record or
strain percentages. Columns include dataset, scope, strain, records, frames,
features, total_values, and each status's count and percentage.

Since NPZ archives exclude statuses, evaluation reads the matching source CSVs
using each archive's record list and feature names. All angles use their shared
`skeleton_status`; features use individual `<feature>_status` columns. Numeric
zeros do not imply missing data. CSV values are checked against NPZ arrays to
reject stale sources. Keep the preprocessed CSVs alongside the archives.
Use `--data-dir PATH` or `--output PATH` to override defaults.

Build the four labeled NPZ datasets from the preprocessed CSV folders:

```powershell
python openworm/openworm_dataset.py
```

This writes `data/openworm_skeleton_angles_dataset.npz` (48 angles),
`data/openworm_manual_features_dataset.npz` (47 features), and
`data/openworm_eigenworms_dataset.npz` (6 eigen projections), and
`data/openworm_matlab_features_dataset.npz` (16 MATLAB features). The skeleton format
uses the previously confirmed 48 segment angles from 49 points. Time/status
columns are excluded from feature arrays. Every strain/record pair must be
present in both CSV folders with matching time samples.

Each archive stores `labels`, `record_ids`, `record_keys`, and `feature_names`.
Each key in `record_keys` points to one `(time, features)` float64 array. This
preserves unequal recording lengths without padding or pickle/object arrays.
Archives are compressed one record at a time to limit memory use. Labels and
record order match across the first three archives. MATLAB uses its own sorted
strain/record list from `data/matlab_features`; missing MATLAB records do not
remove records from the other datasets. At least one MATLAB CSV is required.
To read a record:

```python
import numpy as np
with np.load("data/openworm_manual_features_dataset.npz", allow_pickle=False) as dataset:
    i = 0
    values = dataset[dataset["record_keys"][i]]
    strain = dataset["labels"][i]
    record = dataset["record_ids"][i]
```

Export stored Tierpsy manual features:

```powershell
python openworm/openworm_manual_features.py
```

Outputs are `data/manual_features/<strain>/record_*.csv`. The current records
provide 53 numeric features; timestamps, tracking IDs, flags and categorical
`motion_modes` are excluded. Numeric eigen projections are retained. CSVs contain
`time_s`, feature values, and a separate `<feature>_status` for each feature.
Features can be missing independently, so one shared row-level status would be
misleading. Labels are `real`, `interpolated`, and `zeroed`.

The script shares the skeleton-angle script's time axis and temporal processing:
0.04s sampling, strict <0.25s interpolation, larger/unbracketed gaps set to zero,
without Savitzky–Golay smoothing. Invalid source
skeletons mask feature values. Source skeletons receive the same 49-point spatial
normalization for validity checks. Stored feature values are not recomputed from
spatially smoothed skeletons, and spatial filtering is not applied across
unrelated features. Original feature units are preserved, without mean centering.
Orientation angles are handled across the ±180-degree boundary.
Use `--input-dir E:/openworm` or `--output-dir PATH`
to override defaults. Reruns overwrite the generated CSVs.

Generate skeleton-angle CSVs from locally downloaded records:

```powershell
python openworm/openworm_skeleton_angles.py
```

This reads `data/openworm/<strain>/record_*/*_features.hdf5` and writes
`data/skeleton_angles/<strain>/record_*.csv`. To process the external drive,
use `--input-dir E:/openworm`. Each CSV contains `time_s`, `skeleton_status`,
and 48 angle columns in radians. Status is `real` for observed
skeletons or routine resampling between consecutive valid source frames,
`interpolated` for filled missing-skeleton gaps, and `zeroed` for unusable rows.
These labels describe data availability before smoothing. The image's adjacent-point formula yields
48 angles from 49 points. Missing long gaps are zero and have `skeleton_status=zeroed`.

Coordinates are spatially resampled to 49 equal arc-length points, temporally
resampled to 0.04 seconds, interpolated only across valid endpoint separations
strictly shorter than 0.25 seconds, and smoothed in space and time with
Savitzky–Golay filters. Long gaps and unbracketed ends remain zero, with no
smoothing across their boundaries. Defaults are 5-point windows and order 3;
these can be changed with `--space-window`, `--time-window`, and `--polyorder`.
Video timestamps are used when available; otherwise feature frame indices are
converted using the stored FPS. Angles use `arctan(dx/dy)` minus their frame mean,
following the supplied equation literally. Dependencies: NumPy, h5py, SciPy.

`openworm_zenodo_api.py` replaces `old/openworm_zenodo_api_notebook.ipynb`.
It fetches the complete Zenodo community and generates both CSVs and the counts:

```powershell
python openworm/openworm_zenodo_api.py
```

The outputs are `openworm_movement_zenodo_metadata.csv`,
`openworm_parsed_all_fields.csv`, and `strain_counts.csv`, all beside the script.
Use `--output-dir PATH` to write them elsewhere. The script requires Python 3.10+
and only the standard library. It fetches metadata, not the attached video files.

Anonymous Zenodo access permits 25 records per page and currently limits requests
to 30 per minute. A full fetch of 15,001 records takes over 20 minutes. The script
paces requests, retries transient failures, and splits record-ID ranges to stay
below the 10,000-result paging window. It rejects incomplete or duplicate pages
and validates the new files through the counting script before replacing exports.

To rebuild the outputs from existing metadata without fetching it again:

```powershell
python openworm/openworm_zenodo_api.py --from-metadata openworm/openworm_movement_zenodo_metadata.csv
```

The HTML parser handles both `<b>` and `<strong>` labels, HTML entities, and
nested arena/lab/software sections. Nested fields have names such as `arena_style`,
`lab_name`, and `software_version`. Parsed rows include `record_id` for a reliable
join. Text values retain their source spelling and case. Descriptive durations
such as `3600 per file` remain verbatim and are excluded by numeric time filters.
Array fields in freshly fetched metadata are JSON-encoded within their CSV cells.

Each CSV row represents one Zenodo record. Some records bundle multiple worms,
strains, or videos, and some worms have repeated recordings. Counts measure
records with a `strain` label, including compound labels exactly as supplied;
records providing only plural `strains` or no strain label are not assigned to an
individual strain. The files cannot establish a count of distinct animals.

Auxiliary verification scripts and their reports live in `etc/`. Run the local
pipeline regression checks with:

```powershell
python openworm/etc/test_openworm_pipeline.py
```

`openworm_strain_filter.py` is the command-line version of the
notebook's CSV loading, strain counting, and filtering workflow.
It uses Python 3.10+ and requires no third-party packages.

From the repository root, generate the full strain list:

```powershell
python openworm/openworm_strain_filter.py
```

This reads `openworm_movement_zenodo_metadata.csv` and
`openworm_parsed_all_fields.csv` beside the script, then writes
`openworm/strain_counts.csv` with columns
`strain,count,adult_fed_count,adult_fed_700_1000s_count,adult_fed_850_950s_count,adult_fed_op50_850_950s_30m_wait_count,adult_fed_op50_850_950s_30m_wait_min90pct_skeletonized_count`.
`count` includes all selected recordings. `adult_fed_count` counts only recordings
with stage `adult` and a known, nonempty food value other than `no food`, ignoring
case and surrounding whitespace. `adult_fed_700_1000s_count` applies the same
adult/fed filter and additionally requires `total_time_s` between 700 and 1000
seconds, inclusive. `adult_fed_850_950s_count` applies the same adult/fed filter
with a duration between 850 and 950 seconds, inclusive.
`adult_fed_op50_850_950s_30m_wait_count` further restricts that selection to food
exactly `OP50` and habituation either `30m wait` or missing/blank, ignoring case
and surrounding whitespace. Explicit values such as `no waiting` are excluded.
Missing or invalid durations
are excluded from all time-range columns.
`adult_fed_op50_850_950s_30m_wait_min90pct_skeletonized_count` applies the same
adult, OP50, 850–950-second and 30-minute-or-blank habituation filters and requires
`number_of_segmented_skeletons / (total_time_s * frames_per_second) >= 0.9`.
Missing, invalid or nonpositive FPS and
missing, invalid, negative or nonfinite skeleton counts are excluded. Values
above 100% from the metadata-based formula are retained. Counts are computed for
every strain group, without requiring a group to have at least 10 videos.
Rows are always sorted by the rightmost count column from highest to lowest,
with alphabetical ordering for ties. Currently this is
`adult_fed_op50_850_950s_30m_wait_min90pct_skeletonized_count`.
Use that rightmost column for video-count thresholds and summaries unless another
column is explicitly requested.
Strain names are trimmed and lowercased; missing names and the notebook's missing
value markers are excluded. All recording durations are included by default.
N2 is split into `n2 male` and `n2 hermaphrodite` using the `sex` column. Any N2
records with missing sex are counted as `n2 unknown`. To select an N2 sex, use
`--strain n2 --sex male` or `--strain n2 --sex hermaphrodite`.

Newly generated parsed CSVs join to metadata by `record_id`. The counting script
also accepts the old parsed CSV without IDs; those rows must remain in the same
order as metadata, and their row counts must agree.

To reproduce the notebook's 700–900 second strain counts:

```powershell
python openworm/openworm_strain_filter.py --min-time 700 --max-time 900 --output openworm/strain_count_time_700_to_900.csv
```

Filter and export the notebook's example selection:

```powershell
python openworm/openworm_strain_filter.py --strain cb5 --sex hermaphrodite --stage adult --food OP50 --min-time 700 --max-time 1000 --output openworm/cb5_counts.csv --filtered-output openworm/filtered_cb5.csv
```

All supplied filters apply to both counts and the optional record export.
Categorical filters ignore case and surrounding whitespace; time bounds are
inclusive. Records with invalid or missing durations are excluded only when a
time filter is supplied.

Download using the separate `openworm_download.py` script:

```powershell
python openworm/openworm_download.py --dry-run
python openworm/openworm_download.py
```

By default, four records download simultaneously, with one active file per
record (at most four connections). Set `--workers 2` for two or `--workers 1`
for sequential downloads. Existing final and partial files are reused when
changing the worker count. Stop the previous process with Ctrl+C before starting
a new run. Increasing workers helps only when the server and connection have
available bandwidth.

This recomputes the exact rightmost 90% record filter using the same function as
the counting script, then keeps only strain groups with at least 10 qualifying
records. The current selection is 5,464 records across 178 groups. Defaults fetch
video and feature HDF5 files, excluding WCON.
The dry run makes no network requests and creates no files.

Downloads go to `E:/openworm/<strain>/record_<id>/<filename>`. N2 folders are
`n2 male` and `n2 hermaphrodite`. `--output-dir PATH` overrides the root.
Use `--file-type video` or `features` to select one format, or `video-features`
for both (the default). WCON files are excluded in every mode. Use `--strain cb5`
to restrict the group, or `--max-records 10` for a small batch. Group eligibility
is computed before applying the optional record cap.

Rerun the same command to resume. Completed files are checked against the
metadata byte size and skipped. Interrupted files retain their `.part` suffix
and resume through HTTP Range; if the server ignores Range, that file restarts
safely. Completed partials are promoted only after their final size matches.
Size checks do not provide cryptographic checksum validation. Do not modify the
downloaded or partial files manually. An OS lock prevents concurrent runs from
writing to the same root and releases automatically if the process exits.

The terminal displays aggregate byte progress with ETA for everything,
record-count progress, and a separate byte progress bar with ETA and filename
for each active record. ETAs require
some transferred data and vary with network speed. Partial bytes from earlier
runs are included in the initial progress. Transient failures retry with backoff
and rate-limit delays; failures are logged to `E:/openworm/download_log.jsonl`
and processing continues with other files. Exit status is nonzero if any record
fails. Ctrl+C stops scheduling new records, interrupts retry waits, and leaves
partials ready to resume. Active reads may take up to the 60-second timeout to
stop; wait for the process to exit before restarting. Available disk space is checked
before transfers begin. HTTP read timeouts are 60 seconds.

Run downloader regression checks (local HTTP fixtures; no Zenodo downloads):

```powershell
python openworm/etc/test_openworm_download.py
```

Use `--help` for input/output path overrides and other options. Default paths are
relative to the script, so it can also be run from another working directory.
The original notebook remains available for its exploratory stage/sex and bucket
tables.

ResNet-34 frame features:

```powershell
python openworm/openworm_resnet_features.py --workers 2
python openworm/openworm_resnet_features.py --estimate-only
```

Defaults read `E:/nematode classification/openworm_video_trimmed_890_dataset.npz`
and write `openworm_resnet_features_dataset.npz` in the same folder. Install
`torch torchvision numpy pillow tqdm` in the Python environment first. ImageNet weights
download on first use. Every grayscale uint8 frame is converted to RGB and passed
through the pretrained weights' resize/crop/normalization. No frames are sampled
or dropped. Each `data_<index>` becomes `(frames, 512)` float32; record keys,
labels, IDs and timestamps are preserved. Features are global average pool outputs.

Two independent processes stream/decompress records and perform inference in
parallel. CUDA is selected if available; both workers share that GPU. Reduce
`--workers 1 --batch-size 16` for limited GPU memory, or select `--device cpu`.
`--threads` sets CPU inference threads per worker. `--input` and `--output`
override paths. Header-only `--estimate-only` needs only standard Python.

Progress is persisted every 30 seconds per worker and at record completion in the adjacent `.work` folder. Ctrl+C retains these checkpoints; batches since the last checkpoint may repeat. Rerun to
resume; wait for the previous process to exit first. Input identity and extraction
settings are checked, and an OS lock prevents concurrent writers. A resumed
compressed record must decompress past completed frames before inference resumes.
The final NPZ is published atomically; interrupted final packing restarts using
the saved features. Keep the `.work` directory until completion; it is retained
and may be removed manually after verifying the output. Allow approximately twice
the uncompressed feature size for checkpoints plus the final archive. Final ZIP
compression savings depend on the extracted values.

ResNet extraction displays archive scanning progress, a combined frame progress
bar across workers with elapsed time, frame rate and ETA, and a byte progress
bar with ETA while packing the final NPZ. Resumed frames count toward completion;
the extraction ETA uses throughput from the current run and excludes final packing.

Performance: each worker prefetches one batch, overlapping ZIP decompression and
CPU image transforms with GPU inference. Checkpoints default to every 30 seconds
(`--checkpoint-seconds`), avoiding forced disk synchronization every batch. Try
`--workers 2 --batch-size 128 --threads 4` on a 16 GB GPU; reduce batch size if
CUDA runs out of memory. Compare sustained frame rates after several minutes;
resumed records first decompress past saved frames. The extraction ETA excludes
final packing and may fluctuate during startup.

Bounded ResNet profiling (does not write features or change checkpoints):

```powershell
python openworm/openworm_resnet_features.py --device cuda --batch-size 128 --threads 4 --profile-batches 20
```

Reports read/decompression, preprocessing, CPU-to-GPU transfer, inference and
GPU-to-CPU transfer times after model warmup, plus the original versus batched
preprocessing comparison. Profiling measures sequential stages on the first
record; it excludes checkpoint writes and multiworker/storage contention. Normal
extraction vectorizes tensor conversion and normalization across each batch,
retaining the original PIL resize/crop and float32 normalization. Each worker
checks exact preprocessing equivalence on its first batch before saving features.

Resize/crop preprocessing now uses four image threads per record worker by
default (`--preprocess-workers 4`), preserving input order and the original PIL
operations. `--preprocess-workers 1` disables image threading for a direct profile
comparison. This setting can change across resumes; extracted feature semantics
are unchanged and the first-batch exact-reference check remains enabled. With two
record workers, four image threads each means eight image preprocessing threads
in total, in addition to PyTorch CPU threads. Benchmark 4 versus 8 image threads
if preprocessing still dominates; higher counts may contend for memory/CPU.

Pack completed ResNet checkpoints without importing PyTorch, loading ResNet,
downloading weights, starting model workers or using CUDA:

```powershell
python openworm/openworm_resnet_features.py --pack-only
```

This requires the original input archive for metadata and frame counts, plus the
matching `.work` manifest, completed JSON checkpoints and float32 feature arrays.
It checks completion, array headers and file lengths before packing. Missing or
incomplete checkpoints cause an error rather than starting inference. CPU ZIP
packing retains its byte progress bar/ETA and atomic final-file publication.

Final NPZ compression is parallel on CPU, including in `--pack-only` mode:

```powershell
python openworm/openworm_resnet_features.py --pack-only --pack-workers 4
```

`--pack-workers` defaults to 4 and controls CPU compression threads independently
of extraction `--workers`. Each compressor streams one checkpoint to a temporary
compressed part; one writer copies its compressed payload into the final archive
without recompression. The number of queued parts is bounded by the worker count;
parts are removed after merging and on exit. Allow temporary space for approximately
one compressed record per packing worker. Checkpoints remain unchanged. More
threads can improve CPU-bound compression but may slow a disk-bound run. Packing
progress measures source feature bytes compressed; the final merge can briefly
continue after it reaches 100%. Interrupted packing restarts from checkpoints.

`--pack-only` now skips the original video-array header scan. It reads record
names from the ZIP directory (without opening video entries), obtains frame
counts from feature checkpoint headers, validates completion markers and file
lengths, and then packs. The original archive is still required for its directory,
source identity check and labels/timestamps/other metadata. No original video
payloads are read during pack-only startup or packing.
