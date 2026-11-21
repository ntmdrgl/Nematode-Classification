"""Raw ZIP64 merging and independent compressed-video workers."""
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from zipfile import ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA, ZipFile

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openworm.openworm_dataset import write_array
import openworm.openworm_dataset as dataset
import openworm.openworm_trim as trim
import openworm.utils.openworm_video_dataset as video_dataset
from openworm.utils.openworm_video_workers import packed_records
from openworm.utils.openworm_zip import append_entry, append_archive
import test_openworm_video_dataset as fixtures


def payload(path, name):
    with ZipFile(path) as archive:
        info = archive.getinfo(name)
        with open(path, "rb") as source:
            source.seek(info.header_offset)
            header = source.read(30)
            names, extra = struct.unpack_from("<HH", header, 26)
            source.seek(names + extra, 1)
            return info.compress_type, info.CRC, info.file_size, source.read(info.compress_size)


class NonSeekable(io.BytesIO):
    def seek(self, *args):
        raise io.UnsupportedOperation("non-seekable ZIP writer")


def identity_pack(index, arguments, directory, report):
    marker = Path(directory) / f"{index}.pid"
    marker.write_text(str(os.getpid()))
    deadline = time.monotonic() + 15
    while len(list(Path(directory).glob("*.pid"))) < 2:
        if time.monotonic() > deadline:
            raise RuntimeError("Independent compression worker did not start")
        time.sleep(.01)
    report(1, 1)
    return index, marker, os.getpid()


def refill_pack(index, arguments, directory, report):
    directory = Path(directory)
    if index == 0:
        deadline = time.monotonic() + 15
        while not (directory / "third-started").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Worker refill stalled behind the first record")
            time.sleep(.01)
    if index == 2:
        (directory / "third-started").touch()
    report(1, 1)
    return index, directory / f"{index}.part", 0.


class RawZipTests(unittest.TestCase):
    def test_all_codecs_rename_zip64_and_preserve_compressed_payload(self):
        values = np.arange(240, dtype=">u2").reshape(10, 4, 6)
        for method in (ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as d:
                root = Path(d); source = root / "source.npz"; target = root / "target.npz"
                with ZipFile(source, "w", compression=method) as archive:
                    write_array(archive, "mask", values)
                with ZipFile(source) as incoming, ZipFile(target, "w", compression=ZIP_DEFLATED) as outgoing:
                    write_array(outgoing, "labels", ["strain"])
                    append_entry(outgoing, incoming, "mask.npy", "data_000000.npy")
                    write_array(outgoing, "record_ids", ["record_1"])
                self.assertEqual(payload(source, "mask.npy"), payload(target, "data_000000.npy"))
                with np.load(target, allow_pickle=False) as archive, ZipFile(target) as check:
                    np.testing.assert_array_equal(archive["data_000000"], values)
                    self.assertEqual(archive["data_000000"].dtype, values.dtype)
                    self.assertIsNone(check.testzip())

    def test_data_descriptors_and_unicode_names(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); stream = NonSeekable()
            values = np.arange(12, dtype="u1").reshape(2, 2, 3)
            with ZipFile(stream, "w", compression=ZIP_LZMA) as output:
                write_array(output, "mask", values)
            source = root / "source.npz"; source.write_bytes(stream.getvalue())
            target = root / "target.npz"
            with ZipFile(source) as incoming, ZipFile(target, "w") as outgoing:
                self.assertTrue(incoming.getinfo("mask.npy").flag_bits & 8)
                append_entry(outgoing, incoming, "mask.npy", "frames_\u00b5.npy")
            with np.load(target, allow_pickle=False) as archive, ZipFile(target) as check:
                np.testing.assert_array_equal(archive["frames_\u00b5"], values)
                self.assertFalse(check.getinfo("frames_\u00b5.npy").flag_bits & 8)
                self.assertIsNone(check.testzip())

    def test_zip64_central_offsets_are_regenerated(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d, patch.object(zipfile, "ZIP64_LIMIT", 64):
            root = Path(d); source = root / "source.npz"; target = root / "target.npz"
            with ZipFile(source, "w", compression=ZIP_BZIP2) as archive:
                write_array(archive, "first", np.arange(500))
                write_array(archive, "second", np.arange(200))
            with ZipFile(target, "w", compression=ZIP_DEFLATED) as output:
                write_array(output, "prefix", np.zeros(1000))
                append_archive(output, source)
            with np.load(target, allow_pickle=False) as archive, ZipFile(target) as check:
                np.testing.assert_array_equal(archive["first"], np.arange(500))
                np.testing.assert_array_equal(archive["second"], np.arange(200))
                self.assertIsNone(check.testzip())

    def test_duplicate_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / "source.npz"; target = root / "target.npz"
            with ZipFile(source, "w") as archive:
                write_array(archive, "mask", np.ones((1, 2, 3), dtype="u1"))
            with ZipFile(source) as incoming, ZipFile(target, "w") as outgoing:
                append_entry(outgoing, incoming, "mask.npy")
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    append_entry(outgoing, incoming, "mask.npy")


class ParallelArchiveTests(unittest.TestCase):
    def test_pack_jobs_run_in_distinct_processes(self):
        with tempfile.TemporaryDirectory() as directory, tqdm(total=2, disable=True) as overall:
            jobs = [(('strain', f'record_{i}'), ()) for i in range(2)]
            results = list(packed_records(identity_pack, jobs, 2, directory, overall))
            pids = {pid for _, _, pid in results}
            self.assertEqual(len(pids), 2)
            self.assertNotIn(os.getpid(), pids)

    def test_next_job_starts_without_waiting_for_first(self):
        with tempfile.TemporaryDirectory() as directory, tqdm(total=3, disable=True) as overall:
            jobs = [(('strain', f'record_{i}'), ()) for i in range(3)]
            results = list(packed_records(refill_pack, jobs, 2, directory, overall))
            self.assertEqual({i for i, _, _ in results}, {0, 1, 2})

    def test_dataset_never_reencodes_compressed_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            expected = fixtures.VideoDatasetTests().video_fixtures(root)
            with patch.object(video_dataset, "copy_images", side_effect=AssertionError("Unexpected image recompression")):
                output, = dataset.build_datasets(root, "video", workers=1)
            for index, strain in enumerate(expected):
                source = root / f"video/{strain}/record_1.npz"
                self.assertEqual(payload(source, "mask.npy"), payload(output, f"data_{index:06d}.npy"))
            self.assertEqual(list(root.glob(".openworm-video-*")), [])

    def test_legacy_stored_images_get_lossless_compression(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); values = np.arange(30, dtype="u1").reshape(5, 2, 3)
            fixtures.mask_export(root, "strain", "record_1", values, method=ZIP_STORED)
            output, = dataset.build_datasets(root, "video", workers=2)
            with np.load(output, allow_pickle=False) as archive, ZipFile(output) as check:
                np.testing.assert_array_equal(archive["data_000000"], values)
                self.assertEqual(check.getinfo("data_000000.npy").compress_type, ZIP_BZIP2)
            self.assertEqual(list(root.glob(".openworm-video-*")), [])

    def test_trim_parallel_matches_serial_all_pixels_and_alignment_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixtures.VideoTrimTests().fixtures(root)
            with patch.object(trim, "MIN_SAMPLES", 3):
                serial, = trim.trim_datasets(root, root / "serial", workers=1, dataset="video")
                parallel, = trim.trim_datasets(root, root / "parallel", workers=2, dataset="video")
            with np.load(serial, allow_pickle=False) as expected, np.load(parallel, allow_pickle=False) as actual, \
                    ZipFile(serial) as left, ZipFile(parallel) as right:
                self.assertEqual(set(expected.files), set(actual.files))
                for key in expected.files:
                    np.testing.assert_array_equal(actual[key], expected[key])
                    self.assertEqual(actual[key].dtype, expected[key].dtype)
                    self.assertEqual(left.getinfo(key + ".npy").compress_type, right.getinfo(key + ".npy").compress_type)
                self.assertIsNone(right.testzip())
            self.assertEqual(list(root.rglob(".openworm-video-*")), [])

    def test_failed_worker_preserves_completed_outputs_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixtures.VideoDatasetTests().video_fixtures(root)
            output, = dataset.build_datasets(root, "video", workers=2)
            before = output.read_bytes()
            source = root / "video/a/record_1.npz"
            with np.load(source, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["strain"] = "wrong"
            np.savez_compressed(source, **arrays)
            with self.assertRaises(ValueError):
                dataset.build_datasets(root, "video", workers=2)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(list(root.glob("*.part")), [])
            self.assertEqual(list(root.glob(".openworm-video-*")), [])


if __name__ == "__main__":
    unittest.main()
