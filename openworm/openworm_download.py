"""Resume selected OpenWorm downloads into E:/openworm/<strain>/record_<id>.

Select adult, OP50, 850–950 s, 30m/blank wait, >=90% skeletonization records
from strain groups with >=10 such records. Default: video and feature HDF5 files.
WCON is always excluded. Four records download simultaneously by default.

Run these commands from the repository root with the virtual environment active:

    # Preview selection, download size and destination without writing files.
    python openworm/openworm_download.py --dry-run

    # Download video + feature HDF5 files to E:/openworm (four workers).
    python openworm/openworm_download.py
    python openworm/openworm_download.py --workers 4

    # Download only feature HDF5 files, or only video HDF5 files.
    python openworm/openworm_download.py --file-type features
    python openworm/openworm_download.py --file-type video

    # Use fewer simultaneous downloads, or download sequentially.
    python openworm/openworm_download.py --workers 2
    python openworm/openworm_download.py --workers 1

    # Limit a run to 10 records, or select one strain group.
    python openworm/openworm_download.py --max-records 10
    python openworm/openworm_download.py --strain cb5
    python openworm/openworm_download.py --strain "n2 male" --file-type features

    # Override destination, minimum records per group, or retry attempts.
    python openworm/openworm_download.py --output-dir "E:/openworm"
    python openworm/openworm_download.py --min-count 20 --retries 8

    # Show all command-line options.
    python openworm/openworm_download.py --help

Group-count eligibility is evaluated before --max-records is applied. Each
strain has its own folder, containing record_<id> subfolders. N2 sexes are
separate groups. Inputs default to the two CSVs beside this script.

To stop, press Ctrl+C and wait for active reads to exit (up to 60 seconds).
Rerun the same command to resume .part files and skip size-verified completed
files. Changing --workers preserves this progress. The display includes an
overall ETA and an ETA for each active record; failures are logged in
download_log.jsonl under the output directory.
"""

import argparse
import ast
import http.client
import json
import re
import shutil
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from queue import Queue, Empty
from urllib.error import HTTPError
from urllib.parse import quote, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from tqdm import tqdm

try:
    from .openworm_strain_filter import load_records, matches_download_filter, strain_group, read_csv
except ImportError:
    from openworm_strain_filter import load_records, matches_download_filter, strain_group, read_csv

ROOT = Path(__file__).resolve().parent
OUTPUT = Path("E:/openworm")
BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, ETA {remaining}, {rate_fmt}]{postfix}"


def list_cell(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        result = ast.literal_eval(value)
    if not isinstance(result, list):
        raise ValueError("File metadata must be a list")
    return result


def safe_name(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if not name or name.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError(f"Unsafe filename: {name!r}")
    return name


def encode_download_url(url):
    """Encode raw spaces/Unicode while preserving existing percent escapes."""
    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        raise ValueError("Download URL contains invalid control characters")
    parts = urlsplit(url)
    path = quote(parts.path, safe="/%:@!$&'()*+,;=-._~")
    query = quote(parts.query, safe="%=&/:?@!$'()*+,;+-._~")
    return urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def build_plan(metadata_csv, parsed_csv, output_dir, min_count=10, kind="video-features", strain=None, max_records=None):
    records = load_records(metadata_csv, parsed_csv)
    selected = [row for row in records if matches_download_filter(row)]
    counts = Counter(strain_group(row) for row in selected)
    selected = [row for row in selected if counts[strain_group(row)] >= min_count
                and (strain is None or strain_group(row) == strain.strip().lower())]
    selected.sort(key=lambda row: (strain_group(row), int(row["record_id"])))
    if max_records is not None:
        selected = selected[:max_records]
    metadata, _ = read_csv(metadata_csv, ("record_id", "file_names", "file_sizes", "file_links"))
    by_id = {row["record_id"]: row for row in metadata}
    plan = []
    paths = set()
    for row in selected:
        group, record_id = strain_group(row), row["record_id"]
        meta = by_id[record_id]
        names, sizes, urls = [list_cell(meta[key]) for key in ("file_names", "file_sizes", "file_links")]
        if not names or not len(names) == len(sizes) == len(urls):
            raise ValueError(f"Incomplete file metadata for record {record_id}")
        files = []
        for name, size, url in zip(names, sizes, urls):
            is_features = name.endswith(("_features.hdf5", "_features.h5"))
            is_video = name.endswith((".hdf5", ".h5")) and not is_features
            if not {"video-features": is_video or is_features, "video": is_video, "features": is_features}[kind]:
                continue
            if urlparse(url).scheme != "https" or int(size) <= 0:
                raise ValueError(f"Invalid file metadata for {record_id}: {name}")
            path = (output_dir / safe_name(group) / f"record_{record_id}" / safe_name(name)).resolve()
            if not path.is_relative_to(output_dir.resolve()) or path in paths:
                raise ValueError(f"Unsafe or duplicate destination: {path}")
            paths.add(path)
            files.append(dict(path=path, url=encode_download_url(url), size=int(size)))
        if not files:
            raise ValueError(f"Record {record_id} has no files of type {kind}")
        plan.append(dict(record_id=record_id, strain=group, files=files))
    return plan


def local_bytes(file):
    path, size = file["path"], file["size"]
    if path.exists():
        if path.stat().st_size != size:
            raise ValueError(f"Existing final file has wrong size; rename it before retrying: {path}")
        return size
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f"Partial file exceeds expected size: {partial}")
    return offset


class DownloadCancelled(Exception):
    pass


def transfer(file, advance, retries=6, opener=urlopen, pause=time.sleep, stop_event=None):
    """Keep .part files after errors; only promote exact-size completed files."""
    path, size = file["path"], file["size"]
    request_url = encode_download_url(file["url"])
    offset = local_bytes(file)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(retries):
        if stop_event is not None and stop_event.is_set():
            raise DownloadCancelled()
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            if offset < size:
                headers = {"User-Agent": "OpenWormDownloader/1.0", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with opener(Request(request_url, headers=headers), timeout=60) as response:
                    if response.status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not match or int(match[1]) != offset or int(match[3]) != size or int(match[2]) != size - 1:
                            raise ValueError("Server returned an unexpected byte range")
                    elif response.status == 200:
                        if offset:
                            # Server ignored Range: restart instead of appending duplicate bytes.
                            with partial.open("wb"):
                                pass
                            advance(-offset)
                            offset = 0
                    else:
                        raise ValueError(f"Unexpected HTTP status: {response.status}")
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) != size - offset:
                        raise ValueError("Response length differs from the metadata size")
                    with partial.open("ab") as destination:
                        while True:
                            if stop_event is not None and stop_event.is_set():
                                raise DownloadCancelled()
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            if offset + len(chunk) > size:
                                raise ValueError("Response exceeds expected file size")
                            destination.write(chunk)
                            offset += len(chunk)
                            advance(len(chunk))
                if offset != size:
                    raise OSError(f"Incomplete transfer: {offset}/{size} bytes")
            partial.replace(path)
            return
        except (OSError, http.client.HTTPException, ValueError) as error:
            if isinstance(error, (http.client.InvalidURL, ValueError)):
                raise
            if isinstance(error, HTTPError) and error.code not in {408, 429} and error.code < 500:
                raise
            if attempt == retries - 1:
                raise
            delay = min(2 ** (attempt + 1), 30)
            if isinstance(error, HTTPError):
                try:
                    delay = max(delay, float(error.headers.get("Retry-After", "0")))
                except ValueError:
                    pass
            tqdm.write(f"Retry {attempt + 1}/{retries - 1} in {delay:g}s: {path.name}: {error}")
            if stop_event is None:
                pause(delay)
            elif stop_event.wait(delay):
                raise DownloadCancelled()


@contextmanager
def download_lock(directory):
    """OS lock releases on Ctrl+C/crash; a leftover lock file is harmless."""
    with (directory / ".download.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another downloader is using this output directory") from error
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run(plan, output_dir, retries, workers=4):
    if workers < 1:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    with download_lock(output_dir):
        total = sum(file["size"] for record in plan for file in record["files"])
        initial = sum(local_bytes(file) for record in plan for file in record["files"])
        if shutil.disk_usage(output_dir).free < total - initial:
            raise OSError("Insufficient free disk space for this plan; use --file-type or --max-records")
        failures = 0
        pending = Queue()
        for record in plan:
            pending.put(record)
        stop_event = threading.Event()
        state_lock = threading.Lock()
        with (output_dir / "download_log.jsonl").open("a", encoding="utf-8") as log, \
                tqdm(total=total, initial=initial, desc="Everything", unit="B", unit_scale=True,
                     position=0, bar_format=BAR_FORMAT) as overall, \
                tqdm(total=len(plan), desc="Records", unit="record", position=1) as record_count:
            def worker(slot):
                nonlocal failures
                while not stop_event.is_set():
                    try:
                        record = pending.get_nowait()
                    except Empty:
                        return
                    files = record["files"]
                    record_errors = []
                    with tqdm(total=sum(file["size"] for file in files), initial=sum(local_bytes(file) for file in files),
                              desc=f"{record['strain']} / {record['record_id']}", unit="B", unit_scale=True,
                              position=2 + slot, leave=False, bar_format=BAR_FORMAT) as current:
                        def advance(amount):
                            with state_lock:
                                overall.update(amount)
                                current.update(amount)
                        for file in files:
                            if stop_event.is_set():
                                return
                            current.set_postfix_str(file["path"].name[:55], refresh=False)
                            try:
                                transfer(file, advance, retries=retries, stop_event=stop_event)
                            except DownloadCancelled:
                                return
                            except (OSError, http.client.HTTPException, ValueError) as error:
                                record_errors.append(f"{file['path'].name}: {error}")
                                tqdm.write(f"FAILED {record['record_id']}: {error}")
                    with state_lock:
                        if record_errors:
                            failures += 1
                        log.write(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                  record_id=record["record_id"], strain=record["strain"],
                                                  status="failed" if record_errors else "complete",
                                                  errors=record_errors)) + "\n")
                        log.flush()
                        record_count.update(1)

            executor = ThreadPoolExecutor(max_workers=workers)
            try:
                futures = [executor.submit(worker, slot) for slot in range(min(workers, len(plan)))]
                for future in as_completed(futures):
                    future.result()
            finally:
                stop_event.set()
                executor.shutdown(wait=True, cancel_futures=True)
            if failures:
                tqdm.write(f"{failures} records failed. Rerun the same command to resume them.")
            return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata-csv", type=Path, default=ROOT / "openworm_movement_zenodo_metadata.csv")
    parser.add_argument("--parsed-csv", type=Path, default=ROOT / "openworm_parsed_all_fields.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--file-type", choices=("video-features", "video", "features"), default="video-features")
    parser.add_argument("--strain", help="Exact group name, e.g. cb5 or 'n2 male'")
    parser.add_argument("--max-records", type=int, help="Optional cap after applying the group-count threshold")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4,
                        help="Simultaneous record downloads, one connection each (default: 4)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.min_count < 1 or args.retries < 1 or args.workers < 1 or (args.max_records is not None and args.max_records < 1):
        parser.error("Counts, retry limits and record caps must be positive")
    plan = build_plan(args.metadata_csv, args.parsed_csv, args.output_dir.resolve(), args.min_count,
                      args.file_type, args.strain, args.max_records)
    total = sum(file["size"] for record in plan for file in record["files"])
    existing = sum(local_bytes(file) for record in plan for file in record["files"])
    print(json.dumps(dict(strain_groups=len({record['strain'] for record in plan}), records=len(plan),
                          files=sum(len(record['files']) for record in plan), file_type=args.file_type, workers=args.workers,
                          total_GiB=round(total / 2**30, 2), remaining_GiB=round((total-existing) / 2**30, 2),
                          output_dir=str(args.output_dir.resolve())), indent=2), flush=True)
    if not plan:
        print("No qualifying records.")
    return 0 if args.dry_run or not plan else run(plan, args.output_dir.resolve(), args.retries, args.workers)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Partial downloads are saved; rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
