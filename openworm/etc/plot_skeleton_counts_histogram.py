"""Plot skeleton counts or percentages in strain groups with >=10 videos.

Uses adult_fed_op50_850_950s_30m_wait_count without the 20,000-skeleton filter.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openworm_strain_filter import load_records, normalize

COLUMN = "adult_fed_op50_850_950s_30m_wait_count"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--metric", choices=("skeleton-count", "skeleton-percent"), default="skeleton-count")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_percent = args.metric == "skeleton-percent"
    if args.output is None:
        filename = "skeleton_percent_histogram.png" if plot_percent else "skeleton_counts_histogram.png"
        args.output = ROOT / "figures" / filename
    if args.min_count < 1:
        parser.error("--min-count must be positive")
    with (ROOT / "strain_counts.csv").open(encoding="utf-8", newline="") as source:
        counts = list(csv.DictReader(source))
    expected = {row["strain"]: int(row[COLUMN]) for row in counts
                if int(row[COLUMN]) >= args.min_count}
    records = load_records(ROOT / "openworm_movement_zenodo_metadata.csv",
                           ROOT / "openworm_parsed_all_fields.csv")
    selected = []
    observed = Counter()
    for record in records:
        strain = record["strain_norm"]
        if strain == "n2":
            strain = f"n2 {normalize(record.get('sex')) or 'unknown'}"
        if strain not in expected:
            continue
        if normalize(record.get("stage")) != "adult" or normalize(record.get("food")) != "op50":
            continue
        if normalize(record.get("habituation")) not in {"", "30m wait"}:
            continue
        try:
            duration = float(record.get("total_time_s") or "nan")
        except ValueError:
            continue
        if not 850 <= duration <= 950:
            continue
        skeletons = float(record.get("number_of_segmented_skeletons") or "nan")
        if not math.isfinite(skeletons) or skeletons < 0 or not skeletons.is_integer():
            raise ValueError(f"Record {record['record_id']} has an invalid skeleton count")
        if plot_percent:
            fps = float(record.get("frames_per_second") or "nan")
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Record {record['record_id']} has an invalid frame rate")
            selected.append(100 * skeletons / (duration * fps))
        else:
            selected.append(int(skeletons))
        observed[strain] += 1
    if dict(observed) != expected:
        raise ValueError("Selected records do not match the qualifying strain-count table")
    if not selected:
        raise ValueError("No qualifying records")

    values = np.asarray(selected)
    bin_width = 2 if plot_percent else 1000
    bins = np.arange(0, (values.max() // bin_width + 2) * bin_width, bin_width)
    frequencies, _ = np.histogram(values, bins=bins)
    if int(frequencies.sum()) != len(selected):
        raise ValueError("Histogram bins do not include every selected record")
    median = float(np.median(values))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.78)
    title = "Skeleton percentage per record" if plot_percent else "Segmented skeletons per record"
    fig.text(0.10, 0.93, title, fontsize=19, weight="bold")
    fig.text(0.10, 0.88,
             f"{len(selected):,} records across {len(expected):,} strain groups with at least {args.min_count} qualifying videos",
             fontsize=11, color="#374151")
    fig.text(0.10, 0.835, "Adult · OP50 · 850–950 seconds · 30-minute wait or blank habituation",
             fontsize=10, color="#4B5563")
    ax.hist(values, bins=bins, color="#3979B7", edgecolor="white", linewidth=0.5)
    ax.axvline(median, color="#B45309", linestyle="--", linewidth=1.7)
    median_label = f"Median: {median:.2f}%" if plot_percent else f"Median: {median:,.0f}"
    ax.text(median - (1 if plot_percent else 400), 0.96, median_label,
            transform=ax.get_xaxis_transform(), ha="right", va="top", color="#92400E")
    xlabel = "Skeleton percentage (%)" if plot_percent else "Number of segmented skeletons per record"
    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel("Number of records", labelpad=10)
    ax.set_xlim(0, bins[-1])
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#9CA3AF")
    if plot_percent:
        fig.text(0.10, 0.065, "Percentage = 100 × skeleton count / (video length in seconds × frame rate).",
                 fontsize=9, color="#4B5563")
        fig.text(0.10, 0.035,
                 "Bins: 2 percentage points. No 20,000-skeleton minimum. Values slightly above 100% are retained.",
                 fontsize=9, color="#4B5563")
    else:
        fig.text(0.10, 0.04,
                 "Bin width: 1,000 skeletons. One observation per recording; N2 sexes are separate strain groups.",
                 fontsize=9, color="#4B5563")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(json.dumps(dict(records=len(selected), strain_groups=len(expected),
                          metric=args.metric, minimum=float(values.min()), maximum=float(values.max()),
                          median=median, histogram_total=int(frequencies.sum()),
                          above_100_percent=int(np.count_nonzero(values > 100)) if plot_percent else None,
                          output=str(args.output.resolve())), indent=2))


if __name__ == "__main__":
    main()
