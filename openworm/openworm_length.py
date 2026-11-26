"""Plot record durations from the labeled OpenWorm NPZ datasets.

Run from the repository root:
    python openworm/openworm_length.py
    python openworm/openworm_length.py --dataset eigenworms
    python openworm/openworm_length.py --dataset skeleton_angles --bins 40
    python openworm/openworm_length.py --dataset manual_features --data-dir D:/data
    python openworm/openworm_length.py --dataset matlab_features

Defaults: all available datasets in D:/data; figures in openworm/figures beside
this script. Use --figures-dir to change the figure destination. Missing NPZs
are skipped. Each dataset gets its own histogram and printed min/max/median.
Each dataset also gets a binned scatter chart: record counts in 5-second bins
from 650 to 950 seconds, plotted at bin centers (empty bins are included).
openworm_length_5s_bin_counts.csv in the figures directory contains the same
bin counts, with one column per available selected dataset. Bins include their
lower edge and exclude their upper edge, except the final bin includes 950 s.
Records strictly shorter than 850 seconds are also saved together in
openworm_records_under_850s.csv in the figures directory, one row per dataset
and record (the same record may appear in multiple datasets).

Length in seconds = number of time samples * 0.04 (25 Hz). This is the duration
represented by the preprocessed record, including real, interpolated and zeroed
samples, rather than a measurement of the original video file. The elapsed time
between the first and last sample is one sample interval shorter. NPZ array
headers are read without decompressing the full feature arrays.
"""

# Allow both `python openworm/<script>.py` and package imports/module execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from openworm.openworm_dataset import DATASET_NAMES, OUTPUTS

FIGURES = Path(__file__).resolve().parent / "figures"
SAMPLE_SECONDS = 0.04


def record_lengths(path):
    """Read numeric record shapes directly from their NPY headers in the ZIP."""
    with np.load(path, allow_pickle=False) as archive:
        keys = archive["record_keys"]
        if keys.ndim != 1 or not len(keys) or len(set(keys)) != len(keys):
            raise ValueError(f"Invalid or empty record_keys: {path}")
    lengths = []
    with ZipFile(path) as archive:
        for key in tqdm(keys, desc=path.stem, unit="record", smoothing=0):
            with archive.open(str(key) + ".npy") as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError(f"Unsupported NPY version {version}: {path}/{key}")
            if len(shape) != 2 or min(shape) < 1 or dtype.hasobject or dtype.kind not in "fiu":
                raise ValueError(f"Invalid record array: {path}/{key}")
            lengths.append(shape[0] * SAMPLE_SECONDS)
    return np.asarray(lengths)


def plot_lengths(lengths, dataset, figures_dir, bins=30):
    minimum, maximum, median = float(lengths.min()), float(lengths.max()), float(np.median(lengths))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lengths, bins=bins, color="#4477AA", edgecolor="white")
    ax.axvline(median, color="#CC3311", linestyle="--", label=f"Median: {median:,.2f} s")
    ax.set(title=f"OpenWorm {dataset.replace('_', ' ')} record lengths",
           xlabel="Record length (seconds)", ylabel="Number of records")
    ax.text(.98, .95, f"Records: {len(lengths):,}\nMin: {minimum:,.2f} s\nMax: {maximum:,.2f} s",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    ax.legend(loc="upper left")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    destination = figures_dir / f"openworm_{dataset}_length_histogram.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(f"{dataset}: {len(lengths):,} records; min {minimum:,.2f} s; "
          f"max {maximum:,.2f} s; median {median:,.2f} s\nSaved {destination}")
    return destination


def plot_binned_scatter(lengths, dataset, figures_dir):
    edges = np.arange(650, 951, 5)
    counts, _ = np.histogram(lengths, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(centers, counts, s=30, color="#4477AA", zorder=3)
    ax.set(title=f"OpenWorm {dataset.replace('_', ' ')} record lengths (5-second bins)",
           xlabel="Record length (seconds; bin centers)", ylabel="Number of records",
           xlim=(650, 950), ylim=(0, max(1, counts.max() * 1.12)))
    ax.set_xticks(np.arange(650, 951, 25))
    ax.set_axisbelow(True)
    ax.grid(alpha=.2)
    outside = int(np.count_nonzero((lengths < 650) | (lengths > 950)))
    ax.text(.02, .96, f"Records: {len(lengths):,}\nOutside 650–950 s: {outside:,}",
            transform=ax.transAxes, va="top")
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    destination = figures_dir / f"openworm_{dataset}_length_binned_scatter.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(f"Saved {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("D:/data"))
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    parser.add_argument("--dataset", choices=("all", *DATASET_NAMES), default="all")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive")
    try:
        found = False
        short_records = []
        bin_edges = np.arange(650, 951, 5)
        bin_counts = {}
        for dataset, filename in zip(DATASET_NAMES, OUTPUTS):
            if args.dataset not in ("all", dataset):
                continue
            path = args.data_dir / filename
            if not path.exists():
                print(f"Skipping missing dataset: {path}")
                continue
            lengths = record_lengths(path)
            bin_counts[dataset] = np.histogram(lengths, bins=bin_edges)[0]
            with np.load(path, allow_pickle=False) as archive:
                labels, ids = archive["labels"], archive["record_ids"]
                if labels.ndim != 1 or ids.ndim != 1 or not len(labels) == len(ids) == len(lengths):
                    raise ValueError(f"Record metadata does not match lengths: {path}")
                for i in np.flatnonzero(lengths < 850):
                    short_records.append(dict(dataset=dataset, strain=str(labels[i]),
                                              record_id=str(ids[i]), length_seconds=f"{lengths[i]:.2f}"))
            plot_lengths(lengths, dataset, args.figures_dir, args.bins)
            plot_binned_scatter(lengths, dataset, args.figures_dir)
            found = True
        if not found:
            parser.exit(1, f"No selected datasets found in {args.data_dir}\n")
        bins_destination = args.figures_dir / "openworm_length_5s_bin_counts.csv"
        with bins_destination.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["bin_start_seconds", "bin_end_seconds", *bin_counts])
            for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                writer.writerow([start, end, *[counts[i] for counts in bin_counts.values()]])
        print(f"Saved {bins_destination}")
        destination = args.figures_dir / "openworm_records_under_850s.csv"
        short_records.sort(key=lambda row: (row["dataset"], float(row["length_seconds"]),
                                            row["strain"], row["record_id"]))
        with destination.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=("dataset", "strain", "record_id", "length_seconds"))
            writer.writeheader()
            writer.writerows(short_records)
        unique = {(row["strain"], row["record_id"]) for row in short_records}
        print(f"Saved {destination}: {len(short_records):,} dataset-record rows, "
              f"{len(unique):,} distinct records shorter than 850 seconds")
    except (OSError, ValueError, KeyError, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
