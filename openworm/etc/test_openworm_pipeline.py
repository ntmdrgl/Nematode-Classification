"""Local regression checks for the OpenWorm export/count pipeline."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openworm_zenodo_api as generator
import openworm_strain_filter as consumer


class PipelineTests(unittest.TestCase):
    def test_html_labels_entities_and_nested_sections(self):
        description = """<ul>
        <li><b>strain</b>: N2</li><li><strong>sex</strong>&nbsp;: male</li>
        <li><strong>stage</strong>: adult</li><li><b>food</b>: A &amp; B</li>
        <li><strong>total time (s)</strong>: 700</li>
        <li><b>lab</b>: <ul><li><b>name</b>: Example lab</li>
          <li><strong>location</strong>: A &lt; B</li></ul></li>
        <li><b>software</b>: <ul><li><b>name</b>: tierpsy</li>
          <li><b>version</b>: 1</li></ul></li>
        <li><b>arena</b>: <ul><li><b>style</b>: petri</li></ul></li>
        <li><strong>number_worms:</strong>40</li>
        <li><strong>strains</strong>: N2 and DA609</li></ul>"""
        self.assertEqual(generator.parse_description(description), {
            "strain": "N2", "sex": "male", "stage": "adult", "food": "A & B",
            "total_time_s": "700", "lab_name": "Example lab", "lab_location": "A < B",
            "software_name": "tierpsy", "software_version": "1", "arena_style": "petri",
            "number_worms": "40", "strains": "N2 and DA609",
        })
        self.assertNotIn("strain", generator.parse_description(
            "<ul><li><strong>strains</strong>: N2 and DA609</li></ul>"))

    def test_fetch_partitions_and_pages_without_assuming_newest_id_is_largest(self):
        records = [{"id": number} for number in (5, 2, 1, 8, 9, 3, 4, 7, 6)]

        class FakeClient:
            page_size = 2

            def search(self, q, sort, size, page):
                lower, upper = q.removeprefix("recid:[").removesuffix("]").split(" TO ")
                selected = [row for row in records if int(lower) <= row["id"]
                            and (upper == "*" or row["id"] <= int(upper))]
                if len(selected) > 4 and page > 1:
                    raise AssertionError("Deep paging limit exceeded")
                return len(selected), selected[(page - 1) * size:page * size]

        self.assertEqual(generator.fetch_all_records(FakeClient(), window=4),
                         [{"id": number} for number in range(1, 10)])

    def test_fetch_rejects_duplicates_and_incomplete_pages(self):
        class BrokenClient:
            page_size = 2

            def search(self, **params):
                return 2, [{"id": 1}, {"id": 1}]

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            generator.fetch_all_records(BrokenClient())

        class ShortClient:
            page_size = 2

            def search(self, **params):
                return 2, [{"id": 1}]

        with self.assertRaisesRegex(ValueError, "Incomplete"):
            generator.fetch_all_records(ShortClient())

    def test_exports_consumer_filters_and_count_boundaries(self):
        cases = [
            ("N2", "male", "adult", "OP50", "700"),
            ("N2", "hermaphrodite", "adult", "other food", "1000"),
            ("N2", "male", "adult", "no food", "800"),
            ("CB5", "hermaphrodite", "L4", "OP50", "800"),
            ("CB5", "hermaphrodite", "adult", "OP50", "699.99"),
            ("CB5", "hermaphrodite", "adult", "OP50", "1000.01"),
            ("CB5", "hermaphrodite", "adult", "", "800"),
            ("CB5", "hermaphrodite", "adult", "OP50", "invalid"),
        ]
        metadata = []
        for index, values in enumerate(cases, 1):
            pairs = zip(("strain", "sex", "stage", "food", "total time (s)"), values)
            description = "<ul>" + "".join(f"<li><strong>{key}</strong>: {value}</li>"
                                            for key, value in pairs) + "</ul>"
            metadata.append(generator.metadata_row({"id": index, "metadata": {
                "description": description, "title": r"Test C:\worms\video.hdf5"},
                "files": [{"key": "one.hdf5", "size": 123, "links": {"self": "example"}}]}))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator.generate_csvs(metadata, root)
            rows = consumer.load_records(root / "openworm_movement_zenodo_metadata.csv",
                                         root / "openworm_parsed_all_fields.csv")
            self.assertEqual(len(rows), len(cases))
            self.assertEqual(rows[0]["title"], r"Test C:\worms\video.hdf5")
            with (root / "strain_counts.csv").open(newline="") as source:
                counts = {row["strain"]: row for row in csv.DictReader(source)}
            self.assertEqual(counts["n2 male"], dict(strain="n2 male", count="2",
                adult_fed_count="1", adult_fed_700_1000s_count="1", adult_fed_850_950s_count="0",
                adult_fed_op50_850_950s_30m_wait_count="0",
                adult_fed_op50_850_950s_30m_wait_min90pct_skeletonized_count="0"))
            self.assertEqual(counts["n2 hermaphrodite"]["adult_fed_700_1000s_count"], "1")
            self.assertEqual(counts["cb5"], dict(strain="cb5", count="5",
                adult_fed_count="3", adult_fed_700_1000s_count="0", adult_fed_850_950s_count="0",
                adult_fed_op50_850_950s_30m_wait_count="0",
                adult_fed_op50_850_950s_30m_wait_min90pct_skeletonized_count="0"))
            args = consumer.build_parser().parse_args(["--strain", "n2", "--sex", "male",
                                                       "--min-time", "700", "--max-time", "1000"])
            self.assertEqual(len(consumer.filter_records(rows, args)), 2)
            parsed = list(reversed(generator.load_metadata(root / "openworm_movement_zenodo_metadata.csv")))
            self.assertEqual(json.loads(parsed[0]["file_names"]), ["one.hdf5"])
            # Sorting the parsed file must not change the association with metadata.
            with (root / "openworm_parsed_all_fields.csv").open(newline="") as source:
                table = list(csv.DictReader(source))
            consumer.write_csv(root / "openworm_parsed_all_fields.csv", table[0].keys(), reversed(table))
            joined = consumer.load_records(root / "openworm_movement_zenodo_metadata.csv",
                                           root / "openworm_parsed_all_fields.csv")
            self.assertEqual(joined[0]["record_id"], "8")
            self.assertEqual(joined[0]["strain"], "CB5")


if __name__ == "__main__":
    unittest.main()
