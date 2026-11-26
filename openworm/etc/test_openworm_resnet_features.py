"""Dependency-free checks for ResNet archive inspection and size estimation."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

SCRIPT = Path(__file__).resolve().parents[1] / 'openworm_resnet_features.py'
spec = importlib.util.spec_from_file_location('resnet_features', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def npy(shape, fortran=False):
    text = repr(dict(descr='|u1', fortran_order=fortran, shape=shape)).encode() + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(text)) + text


class ArchiveTests(unittest.TestCase):
    def test_inspects_shapes_without_loading_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'video.npz'
            with ZipFile(path, 'w') as archive:
                archive.writestr('data_000000.npy', npy((10, 80, 90)))
                archive.writestr('labels.npy', b'ignored')
            self.assertEqual(module.inspect(path), [('data_000000', (10, 80, 90), '|u1')])

    def test_rejects_fortran_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'video.npz'
            with ZipFile(path, 'w') as archive:
                archive.writestr('data_000000.npy', npy((10, 80, 90), True))
            with self.assertRaises(ValueError):
                module.inspect(path)

    def test_pack_validation_accepts_complete_and_rejects_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            key = 'data_000000'
            records = [(key, (2, 80, 90), '|u1')]
            info = repr(dict(descr='<f4', fortran_order=False, shape=(2, 512))).encode() + b'\n'
            payload = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(info)) + info
            path = work / (key + '.npy')
            path.write_bytes(payload + bytes(2 * 512 * 4))
            module.save_json(work / (key + '.json'), {'frames': 2})
            module.validate_checkpoints(work, records)
            module.save_json(work / (key + '.json'), {'frames': 1})
            with self.assertRaisesRegex(ValueError, 'Incomplete checkpoint'):
                module.validate_checkpoints(work, records)
            module.save_json(work / (key + '.json'), {'frames': 2})
            path.write_bytes(payload + bytes(4))
            with self.assertRaisesRegex(ValueError, 'Invalid feature checkpoint'):
                module.validate_checkpoints(work, records)

    def test_parallel_pack_roundtrip_and_cleanup(self):
        class Progress:
            n = 0
            def update(self, amount):
                self.n += amount
        for workers in (1, 3):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                payloads = {f'data_{i:06d}': bytes(range(256)) * (i + 1) for i in range(7)}
                for key, payload in payloads.items():
                    (work / (key + '.npy')).write_bytes(payload)
                records = [(key, (1, 1, 1), '|u1') for key in payloads]
                output = work / 'final.npz'
                progress = Progress()
                with ZipFile(output, 'w') as target:
                    module.pack_feature_records(target, work, records, workers, progress)
                with ZipFile(output) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertEqual(len(archive.namelist()), len(payloads))
                    for key, payload in payloads.items():
                        self.assertEqual(archive.read(key + '.npy'), payload)
                self.assertEqual(progress.n, sum(map(len, payloads.values())))
                self.assertFalse(list(work.glob('.pack-*')))
                self.assertNotIn('torch', module.sys.modules)

    def test_checkpoint_records_never_opens_source_video_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / 'source.npz'
            key = 'data_000000'
            with ZipFile(source, 'w') as archive:
                archive.writestr(key + '.npy', b'not a readable video array')
            info = repr(dict(descr='<f4', fortran_order=False, shape=(3, 512))).encode() + b'\n'
            feature = work / (key + '.npy')
            feature.write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(info)) + info + bytes(3 * 512 * 4))
            with patch.object(module.ZipFile, 'open', side_effect=AssertionError('Video entry opened')):
                self.assertEqual(module.checkpoint_records(source, work), [(key, (3,), '<f4')])
                feature.unlink()
                with self.assertRaisesRegex(ValueError, 'Missing checkpoint'):
                    module.checkpoint_records(source, work)

    def test_checkpoint_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            module.save_json(path, {'frames': 32})
            module.save_json(path, {'frames': 64})
            self.assertEqual(module.json.loads(path.read_text()), {'frames': 64})
            self.assertFalse(path.with_suffix('.tmp').exists())


if __name__ == '__main__':
    unittest.main()
