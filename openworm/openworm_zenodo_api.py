"""Generate OpenWorm metadata, parsed fields, and strain counts.

By default, fetch the entire Zenodo community. Use --from-metadata PATH to
rebuild from a saved metadata CSV without network access. Requires Python 3.10+.
"""

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from .openworm_strain_filter import load_records, write_csv, write_strain_counts
except ImportError:
    from openworm_strain_filter import load_records, write_csv, write_strain_counts


DATA_DIR = Path(__file__).resolve().parent
COMMUNITY = "open-worm-movement-database"
SEARCH_WINDOW = 10000
METADATA_FIELDS = (
    "record_id", "conceptrecid", "doi", "title", "publication_date", "description",
    "keywords", "subjects", "file_count", "file_names", "file_sizes", "file_links",
    "html", "api", "_text", "strain_tags",
)
REQUIRED_PARSED_FIELDS = {"strain", "sex", "stage", "food", "total_time_s"}
STRAIN_PATTERNS = {
    "N2": r"\bN2\b", "CB4856": r"\bCB4856\b",
    "gene_mutant": r"\b[a-z]{3,4}-\d+\b", "dauer": r"\bdauer\b",
    "adult": r"\badult\b", "larva": r"\bL[1-4]\b|\blarva\b",
}


class HtmlNode:
    def __init__(self, tag):
        self.tag = tag
        self.children = []

    def text(self):
        return "".join(child if isinstance(child, str) else child.text()
                       for child in self.children)


class DescriptionParser(HTMLParser):
    """Parse real nested HTML, including both <b> and <strong> field labels."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = HtmlNode("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = HtmlNode(tag)
        self.stack[-1].children.append(node)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input",
                       "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def clean_text(value):
    return re.sub(r"\s+", " ", value).strip()


def parse_description(description):
    parser = DescriptionParser()
    parser.feed(description or "")
    fields = {}

    def visit(node, section=""):
        nested_section = section
        if node.tag == "li":
            label = next((child for child in node.children
                          if isinstance(child, HtmlNode) and child.tag in {"b", "strong"}), None)
            if label is not None:
                key = re.sub(r"[^a-z0-9_]+", "", clean_text(label.text()).lower().replace(" ", "_"))
                tail = node.children[node.children.index(label) + 1:]
                value = clean_text("".join(
                    child if isinstance(child, str) else child.text() for child in tail
                    if isinstance(child, str) or child.tag not in {"ul", "ol"}
                )).lstrip(":").strip()
                if key in {"arena", "lab", "software"} and not value:
                    nested_section = f"{section}_{key}" if section else key
                elif key:
                    fields[f"{section}_{key}" if section else key] = value
        for child in node.children:
            if isinstance(child, HtmlNode):
                visit(child, nested_section)

    visit(parser.root)
    return fields


class ZenodoClient:
    def __init__(self, community=COMMUNITY, page_size=25):
        self.url = f"https://zenodo.org/api/communities/{community}/records"
        self.page_size = page_size
        self.next_request_at = 0

    def search(self, **params):
        request = Request(f"{self.url}?{urlencode(params)}", headers={
            "Accept": "application/json", "User-Agent": "OpenWormMetadataGenerator/1.0",
        })
        for attempt in range(6):
            delay = self.next_request_at - time.time()
            if delay > 0:
                time.sleep(delay)
            try:
                with urlopen(request, timeout=90) as response:
                    # Space anonymous calls to remain below 30 requests/minute.
                    self.next_request_at = time.time() + 2.1
                    if int(response.headers.get("X-RateLimit-Remaining", "1")) == 0:
                        reset = float(response.headers.get("X-RateLimit-Reset", "0"))
                        self.next_request_at = max(self.next_request_at, reset + 1)
                    data = json.load(response)
                total = data["hits"]["total"]
                if isinstance(total, dict):
                    if total.get("relation", "eq") != "eq":
                        raise ValueError("Zenodo returned an inexact total; refusing an incomplete export")
                    total = total["value"]
                return int(total), data["hits"]["hits"]
            except (HTTPError, URLError, TimeoutError) as error:
                if isinstance(error, HTTPError) and error.code != 429 and error.code < 500:
                    raise
                if attempt == 5:
                    raise
                delay = min(2 ** (attempt + 1), 30)
                if isinstance(error, HTTPError):
                    try:
                        delay = max(delay, float(error.headers.get("Retry-After", "0")))
                    except ValueError:
                        pass
                self.next_request_at = time.time() + delay
                print(f"Zenodo request failed; retrying in {delay:g}s: {error}", file=sys.stderr, flush=True)


def fetch_all_records(client, window=SEARCH_WINDOW):
    """Split disjoint record-ID ranges to stay below the deep paging limit."""
    collected = []
    seen = set()

    def fetch_range(lower, upper=None):
        query = f"recid:[{lower} TO {upper if upper is not None else '*'}]"
        params = dict(q=query, sort="newest", size=client.page_size, page=1)
        total, first_page = client.search(**params)
        if total > window:
            if not first_page or upper == lower:
                raise ValueError(f"Cannot split Zenodo query: {query}")
            pivot = (lower + upper) // 2 if upper is not None else max(int(row["id"]) for row in first_page)
            left = fetch_range(lower, pivot)
            right = fetch_range(pivot + 1, upper)
            if left + right != total:
                raise ValueError("Zenodo totals changed while fetching; rerun the export")
            return total
        fetched = 0
        for page in range(1, math.ceil(total / client.page_size) + 1):
            if page == 1:
                hits = first_page
            else:
                current_total, hits = client.search(**{**params, "page": page})
                if current_total != total:
                    raise ValueError("Zenodo totals changed while paging; rerun the export")
            expected_size = min(client.page_size, total - fetched)
            if len(hits) != expected_size:
                raise ValueError(f"Incomplete Zenodo page: {query}, page {page}")
            for record in hits:
                record_id = int(record["id"])
                if record_id in seen or record_id < lower or (upper is not None and record_id > upper):
                    raise ValueError(f"Duplicate or out-of-range Zenodo record: {record_id}")
                seen.add(record_id)
                collected.append(record)
            fetched += len(hits)
            print(f"Fetched {len(collected):,} records (range {lower}–{upper or '*'}, {fetched:,}/{total:,})", flush=True)
        return total

    total = fetch_range(1)
    if len(collected) != total:
        raise ValueError(f"Expected {total} records, fetched {len(collected)}")
    return sorted(collected, key=lambda row: int(row["id"]))


def metadata_row(record):
    md = record.get("metadata") or {}
    files = record.get("files") or []
    links = record.get("links") or {}
    keywords = md.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    subjects = md.get("subjects") or []
    row = dict(
        record_id=record["id"], conceptrecid=record.get("conceptrecid", ""),
        doi=md.get("doi", ""), title=md.get("title", ""),
        publication_date=md.get("publication_date", ""), description=md.get("description", ""),
        keywords="; ".join(keywords),
        subjects="; ".join(subject.get("term", "") for subject in subjects if isinstance(subject, dict)),
        file_count=len(files), file_names=json.dumps([file.get("key") for file in files]),
        file_sizes=json.dumps([file.get("size") for file in files]),
        file_links=json.dumps([(file.get("links") or {}).get("self") for file in files]),
        html=links.get("html") or links.get("self_html", ""), api=links.get("self", ""),
    )
    row["_text"] = "\n".join(row.get(field) or "" for field in ("title", "description", "keywords", "subjects"))
    row["strain_tags"] = json.dumps([name for name, pattern in STRAIN_PATTERNS.items()
                                     if re.search(pattern, row["_text"], flags=re.IGNORECASE)])
    return row


def load_metadata(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(METADATA_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError("Metadata CSV contains a row with extra fields")
        return rows


def generate_csvs(metadata, output_dir):
    ids = [str(row["record_id"]).strip() for row in metadata]
    if any(not record_id.isdigit() for record_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Metadata record IDs must be numeric and unique")
    parsed = [{**parse_description(row["description"]), "record_id": record_id}
              for row, record_id in zip(metadata, ids)]
    fields = ["record_id", *sorted(REQUIRED_PARSED_FIELDS | {
        key for row in parsed for key in row if key != "record_id"
    })]
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("openworm_movement_zenodo_metadata.csv", "openworm_parsed_all_fields.csv", "strain_counts.csv")
    # Validate staged files before writing outputs. Copy bytes into the targets
    # so existing workspace permissions survive Windows' private temp-file ACLs.
    with tempfile.TemporaryDirectory(prefix=".openworm-", dir=output_dir) as temporary:
        staging = Path(temporary)
        write_csv(staging / names[0], METADATA_FIELDS, metadata)
        write_csv(staging / names[1], fields, parsed)
        records = load_records(staging / names[0], staging / names[1])
        counts = write_strain_counts(records, staging / names[2])
        for name in names:
            shutil.copyfile(staging / name, output_dir / name)
    for name in names:
        print(f"Wrote {output_dir / name}")
    print(f"{len(metadata):,} records; {len(counts):,} strain groups; "
          f"{sum(not row.get('strain') for row in parsed):,} records without a single strain label.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-metadata", type=Path, help="Reparse a saved metadata CSV instead of fetching Zenodo")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--community", default=COMMUNITY)
    parser.add_argument("--page-size", type=int, default=25, help="Anonymous API page size (1–25)")
    args = parser.parse_args(argv)
    if not 1 <= args.page_size <= 25:
        parser.error("--page-size must be between 1 and 25 for anonymous Zenodo requests")
    metadata = (load_metadata(args.from_metadata) if args.from_metadata
                else [metadata_row(record) for record in fetch_all_records(ZenodoClient(args.community, args.page_size))])
    if not metadata:
        raise ValueError("No metadata records found; refusing to replace existing exports")
    generate_csvs(metadata, args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
