"""Exercise interrupted/resumed HTTP downloads against a local fixture server."""
import csv
import io
import json
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openworm_download as downloader

BODY = b"0123456789abcdefghij"


class Handler(BaseHTTPRequestHandler):
    requests = []
    parallel_barrier = threading.Barrier(2)

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/parallel"):
            self.parallel_barrier.wait(timeout=5)
        if self.path == "/missing":
            self.send_error(404)
            return
        offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        self.requests.append((self.path, offset))
        if self.path == "/ignore":
            offset = 0
        self.send_response(206 if offset else 200)
        if offset:
            start = offset + 1 if self.path == "/bad-range" else offset
            self.send_header("Content-Range", f"bytes {start}-{len(BODY)-1}/{len(BODY)}")
        self.send_header("Content-Length", str(len(BODY)-offset))
        self.end_headers()
        if self.path == "/cut" and offset == 0:
            self.wfile.write(BODY[:5])
        else:
            self.wfile.write(BODY[offset:])
        self.close_connection = True


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def transfer_case(self, endpoint, prefix=b"", retries=2):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "video.hdf5"
        partial = path.with_name(path.name + ".part")
        if prefix:
            partial.write_bytes(prefix)
        file = dict(path=path, size=len(BODY), url=f"http://127.0.0.1:{self.server.server_port}{endpoint}")
        updates = []
        downloader.transfer(file, updates.append, retries=retries, pause=lambda seconds: None)
        self.assertEqual(path.read_bytes(), BODY)
        self.assertFalse(partial.exists())
        self.assertEqual(sum(updates), len(BODY)-len(prefix))
        return file

    def test_resume(self):
        self.transfer_case("/range", BODY[:7])
        self.assertEqual(Handler.requests[-1], ("/range", 7))

    def test_parallel_run_resumes_and_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = [dict(record_id=str(i), strain="cb5", files=[dict(
                path=root / f"record_{i}" / "video.hdf5", size=len(BODY),
                url=f"http://127.0.0.1:{self.server.server_port}/parallel{i}")]) for i in range(3)]
            for record in plan:
                record["files"][0]["path"].parent.mkdir()
            plan[0]["files"][0]["path"].write_bytes(BODY)
            partial = plan[1]["files"][0]["path"].with_name("video.hdf5.part")
            partial.write_bytes(BODY[:7])
            before = len(Handler.requests)
            with patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(downloader.run(plan, root, retries=1, workers=2), 0)
            requests = Handler.requests[before:]
            self.assertEqual(set(requests), {("/parallel1", 7), ("/parallel2", 0)})
            for record in plan:
                self.assertEqual(record["files"][0]["path"].read_bytes(), BODY)
            logs = [json.loads(line) for line in (root / "download_log.jsonl").read_text().splitlines()]
            self.assertEqual({row["record_id"] for row in logs}, {"0", "1", "2"})
            self.assertTrue(all(row["status"] == "complete" for row in logs))
            with patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(downloader.run(plan, root, retries=1, workers=2), 0)
            self.assertEqual(len(Handler.requests), before + 2)

    def test_parallel_failure_does_not_stop_other_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = [dict(record_id=str(i), strain="cb5", files=[dict(
                path=root / f"record_{i}" / "video.hdf5", size=len(BODY),
                url=f"http://127.0.0.1:{self.server.server_port}/{endpoint}")])
                    for i, endpoint in enumerate(("missing", "range"))]
            with patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(downloader.run(plan, root, retries=1, workers=2), 1)
            self.assertEqual(plan[1]["files"][0]["path"].read_bytes(), BODY)
            logs = [json.loads(line) for line in (root / "download_log.jsonl").read_text().splitlines()]
            self.assertEqual({row["status"] for row in logs}, {"failed", "complete"})

    def test_cancel_preserves_partial_without_starting_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.hdf5"
            partial = path.with_name(path.name + ".part")
            partial.write_bytes(BODY[:7])
            event = threading.Event()
            event.set()
            with self.assertRaises(downloader.DownloadCancelled):
                downloader.transfer(dict(path=path, size=len(BODY), url="http://unused/file"),
                                    lambda amount: None, stop_event=event)
            self.assertEqual(partial.read_bytes(), BODY[:7])

    def test_filename_spaces_and_unicode(self):
        self.transfer_case("/mec-12 (e1605) café.hdf5", BODY[:7])
        self.assertEqual(Handler.requests[-1], ("/mec-12%20(e1605)%20caf%C3%A9.hdf5", 7))

    def test_existing_url_encoding_is_preserved(self):
        self.transfer_case("/mec-12%20(e1605)%20caf%C3%A9.hdf5")
        self.assertEqual(Handler.requests[-1], ("/mec-12%20(e1605)%20caf%C3%A9.hdf5", 0))

    def test_url_control_characters_rejected(self):
        with self.assertRaisesRegex(ValueError, "control characters"):
            downloader.encode_download_url("https://zenodo.org/file\nhere")

    def test_restart_when_server_ignores_range(self):
        self.transfer_case("/ignore", BODY[:7])

    def test_interrupted_transfer_retries_from_saved_offset(self):
        self.transfer_case("/cut")
        self.assertEqual(Handler.requests[-2:], [("/cut", 0), ("/cut", 5)])

    def test_complete_file_skips_network(self):
        file = self.transfer_case("/range")
        before = len(Handler.requests)
        downloader.transfer(file, lambda count: self.fail("Unexpected progress"))
        self.assertEqual(len(Handler.requests), before)

    def test_completed_partial_promoted_without_network(self):
        before = len(Handler.requests)
        self.transfer_case("/range", BODY)
        self.assertEqual(len(Handler.requests), before)

    def test_invalid_range_preserves_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "file"
            partial = path.with_name(path.name + ".part")
            partial.write_bytes(BODY[:7])
            file = dict(path=path, size=len(BODY), url=f"http://127.0.0.1:{self.server.server_port}/bad-range")
            with self.assertRaisesRegex(ValueError, "unexpected byte range"):
                downloader.transfer(file, lambda amount: None, retries=1)
            self.assertEqual(partial.read_bytes(), BODY[:7])
            self.assertFalse(path.exists())

    def test_plan_matches_count_csv(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            plan = downloader.build_plan(root / "openworm_movement_zenodo_metadata.csv",
                                         root / "openworm_parsed_all_fields.csv", Path(temporary))
        with (root / "strain_counts.csv").open(newline="") as source:
            column = "adult_fed_op50_850_950s_30m_wait_min90pct_skeletonized_count"
            expected = {row["strain"]: int(row[column]) for row in csv.DictReader(source) if int(row[column]) >= 10}
        self.assertEqual(dict(Counter(record["strain"] for record in plan)), expected)
        self.assertTrue(all(len(record["files"]) == 2 for record in plan))
        self.assertTrue(all(file["path"].suffix in {".hdf5", ".h5"}
                            for record in plan for file in record["files"]))


if __name__ == "__main__":
    unittest.main()
