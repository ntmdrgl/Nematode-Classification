"""Count and filter OpenWorm strain records.

Run without arguments to write strain_counts.csv from the two CSVs beside this
script. Only the Python standard library is required.
"""

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
MISSING_VALUES = {"", "nan", "-n/a-", "n/a", "na", "-"}
METADATA_FIELDS = ("record_id", "doi", "title", "publication_date")


def normalize(value):
    """Match the notebook's case/whitespace normalization and missing values."""
    value = (value or "").strip().lower()
    return "" if value in MISSING_VALUES else value


def read_csv(path, required):
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f"{path}: a row has more fields than the header")
        return rows, reader.fieldnames


def load_records(metadata_csv, parsed_csv):
    """Join by record_id when available, otherwise align the CSVs by row order."""
    metadata, _ = read_csv(metadata_csv, METADATA_FIELDS)
    parsed, fields = read_csv(parsed_csv, ("strain", "stage", "food", "total_time_s"))
    by_id = {}
    for row in metadata:
        record_id = (row["record_id"] or "").strip()
        if not record_id.isdigit() or record_id in by_id:
            raise ValueError(f"Metadata record_id must be unique and numeric: {record_id!r}")
        by_id[record_id] = row

    has_ids = "record_id" in fields
    if not has_ids and len(parsed) != len(metadata):
        raise ValueError(
            "Parsed and metadata row counts differ; cannot align by row order. "
            "Export record_id in the parsed CSV to join by ID."
        )

    records = []
    for index, row in enumerate(parsed):
        if has_ids:
            record_id = (row["record_id"] or "").strip()
            if record_id not in by_id:
                raise ValueError(f"Parsed record_id {record_id!r} is absent from metadata")
            meta = by_id[record_id]
        else:
            meta = metadata[index]
        record = {**row, **{field: meta[field] for field in METADATA_FIELDS}}
        record["strain_norm"] = normalize(row["strain"])
        records.append(record)
    return records


def filter_records(records, args):
    selected = []
    categories = {key: getattr(args, key) for key in ("strain", "sex", "stage", "food")}
    for record in records:
        if any(
            target is not None and normalize(record.get(key)) != normalize(target)
            for key, target in categories.items()
        ):
            continue
        if args.min_time is not None or args.max_time is not None:
            try:
                duration = float(record.get("total_time_s") or "nan")
            except ValueError:
                continue
            if not math.isfinite(duration):
                continue
            if args.min_time is not None and duration < args.min_time:
                continue
            if args.max_time is not None and duration > args.max_time:
                continue
        selected.append(record)
    return selected


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def strain_group(record):
    strain = normalize(record.get("strain"))
    return f"n2 {normalize(record.get('sex')) or 'unknown'}" if strain == "n2" else strain


def matches_download_filter(record):
    """The rightmost count column's record-level selection, shared by downloads."""
    try:
        duration = float(record.get("total_time_s") or "nan")
        fps = float(record.get("frames_per_second") or "nan")
        skeletons = float(record.get("number_of_segmented_skeletons") or "nan")
    except ValueError:
        return False
    return (bool(strain_group(record)) and normalize(record.get("stage")) == "adult"
            and normalize(record.get("food")) == "op50"
            and normalize(record.get("habituation")) in {"", "30m wait"}
            and 850 <= duration <= 950 and math.isfinite(fps) and fps > 0
            and math.isfinite(skeletons) and skeletons >= 0
            and skeletons / (duration * fps) >= 0.9)


def write_strain_counts(records, output):
    """Count total and adult fed records per strain, splitting N2 by sex."""
    counts = Counter()
    adult_fed_counts = Counter()
    adult_fed_700_1000s_counts = Counter()
    adult_fed_850_950s_counts = Counter()
    adult_fed_op50_850_950s_30m_wait_counts = Counter()
    adult_fed_op50_850_950s_30m_wait_min90pct_skeletonized_counts = Counter()
    for record in records:
        strain = record["strain_norm"]
        if not strain:
            continue
        if strain == "n2":
            sex = normalize(record.get("sex")) or "unknown"
            strain = f"n2 {sex}"
        counts[strain] += 1
        food = normalize(record.get("food"))
        if normalize(record.get("stage")) == "adult" and food and food != "no food":
            adult_fed_counts[strain] += 1
            try:
                duration = float(record.get("total_time_s") or "nan")
            except ValueError:
                duration = math.nan
            if 700 <= duration <= 1000:
                adult_fed_700_1000s_counts[strain] += 1
            if 850 <= duration <= 950:
                adult_fed_850_950s_counts[strain] += 1
                if food == "op50" and normalize(record.get("habituation")) in {"", "30m wait"}:
                    adult_fed_op50_850_950s_30m_wait_counts[strain] += 1
                    if matches_download_filter(record):
                        adult_fed_op50_850_950s_30m_wait_min90pct_skeletonized_counts[strain] += 1
    count_columns = {
        "count": counts,
        "adult_fed_count": adult_fed_counts,
        "adult_fed_700_1000s_count": adult_fed_700_1000s_counts,
        "adult_fed_850_950s_count": adult_fed_850_950s_counts,
        "adult_fed_op50_850_950s_30m_wait_count": adult_fed_op50_850_950s_30m_wait_counts,
        "adult_fed_op50_850_950s_30m_wait_min90pct_skeletonized_count":
            adult_fed_op50_850_950s_30m_wait_min90pct_skeletonized_counts,
    }
    sort_counts = count_columns[next(reversed(count_columns))]
    ordered = sorted(counts, key=lambda strain: (-sort_counts[strain], strain))
    write_csv(output, ("strain", *count_columns),
              ({"strain": strain, **{column: values[strain] for column, values in count_columns.items()}}
               for strain in ordered))
    return counts


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path,
                        default=DATA_DIR / "openworm_movement_zenodo_metadata.csv")
    parser.add_argument("--parsed-csv", type=Path,
                        default=DATA_DIR / "openworm_parsed_all_fields.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "strain_counts.csv",
                        help="Strain counts CSV (default: beside this script)")
    for category in ("strain", "sex", "stage", "food"):
        parser.add_argument(f"--{category}", help=f"Filter by {category}, ignoring case")
    parser.add_argument("--min-time", type=float, help="Inclusive minimum duration in seconds")
    parser.add_argument("--max-time", type=float, help="Inclusive maximum duration in seconds")
    parser.add_argument("--filtered-output", type=Path, help="Also save the selected record table")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for value in (args.min_time, args.max_time):
        if value is not None and not math.isfinite(value):
            parser.error("Time limits must be finite numbers")
    if args.min_time is not None and args.max_time is not None and args.min_time > args.max_time:
        parser.error("--min-time must not exceed --max-time")

    records = load_records(args.metadata_csv, args.parsed_csv)
    for column in ("sex", "stage", "food"):
        if getattr(args, column) is not None and records and column not in records[0]:
            parser.error(f"Parsed CSV is missing filter column {column!r}")
    if (args.min_time is not None or args.max_time is not None) and records and "total_time_s" not in records[0]:
        parser.error("Parsed CSV is missing filter column 'total_time_s'")
    selected = filter_records(records, args)
    # Prevent output options from overwriting an input or another output.
    outputs = [args.output] + ([args.filtered_output] if args.filtered_output else [])
    paths = [path.resolve() for path in outputs]
    inputs = {args.metadata_csv.resolve(), args.parsed_csv.resolve()}
    if len(paths) != len(set(paths)) or any(path in inputs for path in paths):
        parser.error("Output paths must be distinct from each other and the input CSVs")
    counts = write_strain_counts(selected, args.output)
    print(f"Loaded {len(records):,} records; selected {len(selected):,}.")
    print(f"Wrote {len(counts):,} strain groups ({sum(counts.values()):,} records) to {args.output.resolve()}")
    print(f"Excluded {len(selected) - sum(counts.values()):,} records with missing strain names.")
    if args.filtered_output:
        fields = list(records[0]) if records else [*METADATA_FIELDS, "strain", "strain_norm"]
        write_csv(args.filtered_output, fields, selected)
        print(f"Selected records: {args.filtered_output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
