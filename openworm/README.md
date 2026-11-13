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
