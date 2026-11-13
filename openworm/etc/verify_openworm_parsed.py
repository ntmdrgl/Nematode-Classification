"""Audit the parsed CSV against saved descriptions without changing source data."""

import argparse
import ast
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent
CORE_FIELDS = ("strain", "sex", "stage", "food", "total_time_s",
               "frames_per_second", "base_name", "timestamp", "worm_id",
               "days_of_adulthood")


class Node:
    def __init__(self, tag):
        self.tag = tag
        self.children = []

    def text(self):
        return "".join(child if isinstance(child, str) else child.text()
                       for child in self.children)


class DescriptionParser(HTMLParser):
    """Build a tree so nested lists and both bold HTML styles can be audited."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag)
        self.stack[-1].children.append(node)
        if tag not in {"br", "hr", "img", "input", "meta", "link", "wbr"}:
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


def clean(value):
    return re.sub(r"\s+", " ", value).strip()


def parse_independently(description):
    parser = DescriptionParser()
    parser.feed(description)
    fields = {}

    def visit(node, section=""):
        next_section = section
        if node.tag == "li":
            label = next((child for child in node.children
                          if isinstance(child, Node) and child.tag in {"b", "strong"}), None)
            if label:
                key = re.sub(r"[^a-z0-9_]+", "", clean(label.text()).lower().replace(" ", "_"))
                after_label = node.children[node.children.index(label) + 1:]
                value = clean("".join(
                    child if isinstance(child, str) else child.text()
                    for child in after_label
                    if isinstance(child, str) or child.tag not in {"ul", "ol"}
                )).lstrip(":").strip()
                if key in {"lab", "software", "arena"} and not value:
                    next_section = key
                else:
                    fields[f"{section}_{key}" if section else key] = value
        for child in node.children:
            if isinstance(child, Node):
                visit(child, next_section)

    visit(parser.root)
    return fields


def original_notebook_parser():
    notebook = json.loads((ROOT / "old/openworm_zenodo_api_notebook.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "def parse_description_all_fields(" in "".join(cell["source"]))
    tree = ast.parse(source)
    # Replay only the original parser definitions and regex constants, not the
    # notebook's imports, network requests, DataFrame operations or file writes.
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   or (isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name)
                               and target.id in {"LI_RE", "BOLD_KEY_RE", "TAG_RE"}
                               for target in node.targets))]
    namespace = {"re": re}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook_parser", "exec"), namespace)
    return namespace["parse_description_all_fields"]


def read_rows(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def get_json(url):
    with urlopen(url, timeout=45) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Check live community total and sample records")
    args = parser.parse_args()
    metadata = read_rows("openworm_movement_zenodo_metadata.csv")
    parsed = read_rows("openworm_parsed_all_fields.csv")
    if len(metadata) != len(parsed):
        raise ValueError("CSV row counts differ")
    original = original_notebook_parser()
    original_mismatches = Counter()
    core_mismatches = Counter()
    recovered = Counter()
    recovered_durations = Counter()
    populated_strain_mismatches = Counter()
    missing_records = []
    examples = []
    for meta, row in zip(metadata, parsed):
        replayed = original(meta["description"])
        independent = parse_independently(meta["description"])
        for field in set(row) | set(replayed):
            if (row.get(field) or "") != (replayed.get(field) or ""):
                original_mismatches[field] += 1
        for field in CORE_FIELDS:
            actual, expected = row.get(field) or "", independent.get(field) or ""
            if actual != expected:
                core_mismatches[field] += 1
                if row["strain"]:
                    populated_strain_mismatches[field] += 1
                if not actual and expected:
                    recovered[field] += 1
                if len(examples) < 12:
                    examples.append(dict(record_id=meta["record_id"], field=field,
                                         exported=actual, description=expected))
        if not row["strain"]:
            recovered_durations[independent.get("total_time_s", "")] += 1
            missing_records.append(dict(record_id=meta["record_id"], title=meta["title"],
                                        recovered_strain=independent.get("strain", ""),
                                        recovered_stage=independent.get("stage", ""),
                                        recovered_sex=independent.get("sex", ""),
                                        recovered_food=independent.get("food", "")))

    def duplicates(field):
        counts = Counter(row[field] for row in metadata if row[field])
        return sum(count - 1 for count in counts.values())

    worm_ids = Counter(row["worm_id"] for row in parsed if row["worm_id"])
    base_names = Counter(row["base_name"] for row in parsed if row["base_name"])
    report = dict(
        audited_at_utc=datetime.now(timezone.utc).isoformat(),
        metadata_rows=len(metadata), parsed_rows=len(parsed),
        duplicate_record_ids=duplicates("record_id"),
        duplicate_concept_ids=duplicates("conceptrecid"),
        original_parser_mismatches=dict(original_mismatches),
        independent_core_mismatches=dict(core_mismatches),
        independent_core_mismatches_in_rows_with_strain=dict(populated_strain_mismatches),
        missing_values_recovered_by_independent_parser=dict(recovered),
        records_missing_strain=len(missing_records),
        recovered_strain_counts=dict(Counter(row["recovered_strain"] for row in missing_records)),
        recovered_duration_values=dict(recovered_durations),
        populated_base_names=sum(base_names.values()),
        duplicate_base_names=sum(count - 1 for count in base_names.values()),
        rows_with_worm_id=sum(worm_ids.values()), unique_populated_worm_ids=len(worm_ids),
        worm_ids_seen_in_multiple_records=sum(count > 1 for count in worm_ids.values()),
        max_records_for_one_worm_id=max(worm_ids.values(), default=0),
        file_count_distribution=dict(Counter(row["file_count"] for row in metadata)),
        source_nested_field_example={key: value for key, value in parse_independently(metadata[0]["description"]).items()
                                     if key.startswith(("arena_", "lab_", "software_"))},
        mismatch_examples=examples,
    )
    if args.live:
        community_url = "https://zenodo.org/api/communities/open-worm-movement-database/records?size=1"
        live = get_json(community_url)
        report["live_community_url"] = community_url
        report["live_community_total"] = live["hits"]["total"]
        sample_ids = {metadata[0]["record_id"], metadata[-1]["record_id"], "4074963", "1745040"}
        sample_ids.add(next(meta["record_id"] for meta, row in zip(metadata, parsed) if row["worm_id"]))
        report["live_samples"] = []
        for record_id in sorted(sample_ids, key=int):
            live_record = get_json(f"https://zenodo.org/api/records/{record_id}")
            saved = next(row for row in metadata if row["record_id"] == record_id)
            live_md = live_record["metadata"]
            report["live_samples"].append(dict(
                record_id=record_id,
                title_matches=live_md["title"] == saved["title"],
                description_matches=live_md["description"] == saved["description"],
                file_count_matches=len(live_record.get("files", [])) == int(saved["file_count"]),
            ))
    output = REPORT_DIR / "parsing_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (REPORT_DIR / "parsing_audit_missing_strains.csv").open("w", encoding="utf-8", newline="") as destination:
        fields = ("record_id", "title", "recovered_strain", "recovered_stage", "recovered_sex", "recovered_food")
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(missing_records)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"Audit saved to {output}")


if __name__ == "__main__":
    main()
