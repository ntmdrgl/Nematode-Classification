"""Direct final-only HDF5 pipeline, resume journals, and numerical equivalence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZIP_BZIP2
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import openworm.openworm_pipeline as pipeline
from openworm.openworm_dataset import write_array
from openworm.openworm_skeleton_angles import read_record, preprocess, spatial_resample
from openworm.openworm_manual_features import feature_names, preprocess_feature, ORIENTATIONS
from openworm.utils.openworm_video_trim import prepare_hdf5_record


def fake_matlab(video, query_times, matlab=None, frame_workers=1, pool_type='threads', progress_callback=None, temporary_dir=None):
    if progress_callback:
        progress_callback('complete', len(query_times), len(query_times))
    return np.asarray(query_times)[:, None] + np.arange(16)[None, :]


class PipelineTests(unittest.TestCase):
    def fixtures(self, root):
        raw_root = root / 'original hdf5'
        for label, record, count in [('a', 'record_0', 10), ('b', 'record_1', 30), ('c', 'record_2', 32)]:
            folder = raw_root / label / record
            folder.mkdir(parents=True)
            names = ['head_orientation'] + [f'manual_{i}' for i in range(46)] + pipeline.EIGEN
            table = np.zeros(count, dtype=[('timestamp', 'f8'), ('worm_index', 'f8')] + [(n, 'f8') for n in names])
            table['timestamp'] = np.arange(count)
            table['worm_index'] = 1
            for i, name in enumerate(names):
                table[name] = np.arange(count) + i
            table[names[1]][3] = np.nan
            y = np.linspace(0, 10, 49)
            skeletons = np.array([np.column_stack((np.sin(y / 3) * (1 + t/100), y)) for t in range(count)])
            skeletons[10:20] = np.nan
            with h5py.File(folder / 'worm_features.hdf5', 'w') as output:
                output.create_dataset('coordinates/skeletons', data=skeletons)
                output.create_dataset('features_timeseries', data=table).attrs['fps'] = 25.
            raw = np.r_[np.arange(10), np.arange(20, 40)]
            images = np.repeat((raw + 1).astype('>u2'), 6).reshape(-1, 2, 3)
            with h5py.File(folder / 'worm.hdf5', 'w') as output:
                output.create_dataset('mask', data=images).attrs['fps'] = 25.
                output.create_dataset('timestamp/raw', data=raw)
                output.create_dataset('timestamp/time', data=raw * .04)
                output.create_dataset('full_data', data=np.full((2, 2, 3), 999, dtype='u2'))
        return raw_root

    def run_pipeline(self, root, raw, workers=1, dataset='all', **options):
        with patch.object(pipeline, 'MIN_SAMPLES', 10), patch.object(pipeline, 'rebuild_record', side_effect=fake_matlab):
            return pipeline.run_pipeline(raw, root/'data', root/'out', root/'ssd', workers=workers,
                                         matlab_workers=1, frame_workers=1, dataset=dataset, **options)

    def test_final_only_matches_existing_preprocessing_with_serial_and_process_workers(self):
        for workers in (1, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as d:
                root = Path(d); raw = self.fixtures(root)
                original = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in raw.rglob('*.hdf5')}
                paths = self.run_pipeline(root, raw, workers)
                self.assertEqual(len(paths), 5)
                self.assertEqual(len(list((root/'out').iterdir())), 5)
                self.assertEqual(list((root/'data').rglob('*.npz')), [])
                self.assertEqual(list((root/'data').rglob('*.csv')), [])
                self.assertEqual(list((root/'ssd').rglob('*.hdf5')), [])
                self.assertEqual(list((root/'ssd').rglob('*.part')), [])
                for dataset, path in zip(pipeline.DATASETS, paths):
                    with np.load(path, allow_pickle=False) as archive:
                        self.assertEqual(archive['labels'].tolist(), ['b', 'c'])
                        self.assertEqual(archive['record_ids'].tolist(), ['record_1', 'record_2'])
                        np.testing.assert_allclose(archive['time_s'], np.arange(30)*.04)
                        for i, label in enumerate(('b', 'c')):
                            features = raw / label / f'record_{i+1}' / 'worm_features.hdf5'
                            times, skeletons = read_record(features)
                            if dataset == 'skeleton_angles':
                                expected = preprocess(times, skeletons)[1][:30]
                            elif dataset in ('manual_features', 'eigenworms'):
                                valid = np.isfinite(spatial_resample(skeletons)).all(axis=(1,2))
                                with h5py.File(features) as source:
                                    table = source['features_timeseries'][:]
                                names = pipeline.EIGEN if dataset == 'eigenworms' else [n for n in feature_names(table) if n not in pipeline.EIGEN]
                                expected = np.column_stack([preprocess_feature(times, table[n], valid, n in ORIENTATIONS)[1] for n in names])[:30]
                            elif dataset == 'matlab_features_new':
                                expected = np.arange(30)[:,None]*.04 + np.arange(16)[None,:]
                            else:
                                video = features.with_name('worm.hdf5')
                                indices, _ = prepare_hdf5_record(video, (label,f'record_{i+1}'), features,
                                                                 30 if i==0 else 32, 30, 'nearest', None, video)
                                with h5py.File(video) as source:
                                    pixels = source['mask'][:]
                                expected = np.zeros((30,2,3), dtype=pixels.dtype)
                                valid = indices >= 0
                                expected[valid] = pixels[indices[valid]]
                                self.assertTrue((~valid).any())
                                with ZipFile(path) as zipped:
                                    self.assertEqual(zipped.getinfo(f'data_{i:06d}.npy').compress_type, ZIP_BZIP2)
                            actual = archive[f'data_{i:06d}']
                            self.assertEqual(actual.shape, expected.shape)
                            self.assertEqual(actual.dtype, expected.dtype)
                            np.testing.assert_allclose(actual, expected, atol=1e-12)
                self.assertEqual(original, {p: (p.stat().st_size,p.stat().st_mtime_ns) for p in raw.rglob('*.hdf5')})
                mtimes = [p.stat().st_mtime_ns for p in paths]
                with patch.object(pipeline, 'pack_record', side_effect=AssertionError('must skip')):
                    self.run_pipeline(root, raw, workers=1)
                self.assertEqual(mtimes, [p.stat().st_mtime_ns for p in paths])

    def test_video_only_matches_existing_skeleton_reference_and_auto_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=self.fixtures(root)
            out=root/'out'; out.mkdir()
            reference=out/'openworm_skeleton_angles_trimmed_890_dataset.npz'
            with ZipFile(reference,'w') as archive:
                pipeline.write_metadata(archive,pipeline.ANGLES,[('c','record_2')])
                write_array(archive,'data_000000',np.zeros((12,48)))
            with patch.object(pipeline,'discover',side_effect=AssertionError('must use exact existing reference')):
                paths=self.run_pipeline(root,raw,workers=2,dataset='video')
                with np.load(paths[0],allow_pickle=False) as archive:
                    self.assertEqual(archive['labels'].tolist(),['c'])
                    self.assertEqual(archive['record_ids'].tolist(),['record_2'])
                    self.assertEqual(archive['data_000000'].shape,(12,2,3))
                with patch.object(pipeline,'pack_record',side_effect=AssertionError('must skip matching final')):
                    self.run_pipeline(root,raw,dataset='video',reference_skeleton=reference)

    def test_video_only_cli_uses_process_workers_and_saves_no_other_datasets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = self.fixtures(root)
            args = ['pipeline', '--dataset', 'video', '--hdf5-dir', str(raw),
                    '--data-dir', str(root/'data'), '--output-dir', str(root/'out'),
                    '--staging-dir', str(root/'ssd'), '--workers', '2']
            with patch.object(sys, 'argv', args), patch.object(pipeline, 'MIN_SAMPLES', 10), patch.object(
                    pipeline, 'rebuild_record', side_effect=AssertionError('video must not launch MATLAB')):
                pipeline.main()
            files = list((root/'out').iterdir())
            self.assertEqual([p.name for p in files], ['openworm_video_trimmed_890_dataset.npz'])
            with np.load(files[0], allow_pickle=False) as source:
                self.assertEqual(source['data_000000'].shape, (30,2,3))
                self.assertEqual(source['labels'].tolist(), ['b','c'])
                self.assertTrue((source['source_index_000000'] < 0).any())
            self.assertEqual(list((root/'ssd').rglob('*.hdf5')), [])

    def test_video_only_resumes_same_partial_file_and_skips_committed_record(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = self.fixtures(root)
            original_commit = pipeline.Checkpoint.commit
            def interrupt(checkpoint, index, chunk):
                original_commit(checkpoint, index, chunk)
                raise KeyboardInterrupt()
            with patch.object(pipeline.Checkpoint, 'commit', interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_pipeline(root, raw, dataset='video')
            partial = root/'out/openworm_video_trimmed_890_dataset.npz.part'
            self.assertTrue(partial.is_file())
            with np.load(partial, allow_pickle=False) as source:
                saved = source['data_000000'].copy()
            original_pack = pipeline.pack_record
            def guard(slot, args, directory, report):
                self.assertEqual(args[1], 'video')
                self.assertEqual(args[2], 1, 'reprocessed already-committed record')
                return original_pack(slot, args, directory, report)
            with patch.object(pipeline, 'pack_record', guard):
                paths = self.run_pipeline(root, raw, dataset='video')
            self.assertEqual(len(paths), 1)
            self.assertFalse(partial.exists())
            with np.load(paths[0], allow_pickle=False) as source:
                np.testing.assert_array_equal(source['data_000000'], saved)
                self.assertEqual(source['record_keys'].tolist(), ['data_000000','data_000001'])
            mtime = paths[0].stat().st_mtime_ns
            with patch.object(pipeline, 'pack_record', side_effect=AssertionError('must skip completed video')):
                self.run_pipeline(root, raw, dataset='video')
            self.assertEqual(paths[0].stat().st_mtime_ns, mtime)

    def test_resume_skips_committed_records_after_interruption(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = self.fixtures(root)
            original_commit = pipeline.Checkpoint.commit
            def interrupt(checkpoint, index, chunk):
                original_commit(checkpoint, index, chunk)
                raise KeyboardInterrupt()
            with patch.object(pipeline.Checkpoint, 'commit', interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_pipeline(root, raw)
            self.assertEqual(list((root/'ssd').rglob('*.hdf5')), [])
            original_pack = pipeline.pack_record
            def guard(slot, args, directory, report):
                if args[1] == 'skeleton_angles' and args[2] == 0:
                    self.fail('reprocessed committed record')
                return original_pack(slot, args, directory, report)
            with patch.object(pipeline, 'pack_record', guard):
                paths = self.run_pipeline(root, raw)
            self.assertTrue(all(p.is_file() for p in paths))

    def test_resume_reuses_finished_worker_chunk_before_commit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = self.fixtures(root)
            with patch.object(pipeline.Checkpoint, 'commit', side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_pipeline(root, raw)
            self.assertEqual(len(list((root/'ssd').rglob('*.npz.part'))), 1)
            original_pack = pipeline.pack_record
            def guard(slot, args, directory, report):
                if args[1] == 'skeleton_angles' and args[2] == 0:
                    self.fail('reprocessed finished worker chunk')
                return original_pack(slot, args, directory, report)
            with patch.object(pipeline, 'pack_record', guard):
                self.run_pipeline(root, raw)
            self.assertEqual(list((root/'ssd').rglob('*.part')), [])

    def test_invalid_or_legacy_partial_is_preserved_without_destroying_bytes(self):
        for kind in ('empty', 'not_zip', 'truncated_zip', 'legacy_zip'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root=Path(d); destination=root/'final.npz'; partial=destination.with_suffix('.npz.part')
                if kind in ('truncated_zip','legacy_zip'):
                    with ZipFile(partial,'w') as archive:
                        write_array(archive,'data_000000',np.ones((4,2)))
                    if kind=='truncated_zip':
                        partial.write_bytes(partial.read_bytes()[:-22])
                else:
                    partial.write_bytes(b'' if kind=='empty' else b'unfinished older output')
                before=partial.read_bytes()
                checkpoint=pipeline.Checkpoint(destination,root,'signature')
                preserved=list(root.glob('final.npz.part.preserved-*'))
                self.assertEqual(len(preserved),1)
                self.assertEqual(preserved[0].read_bytes(),before)
                self.assertFalse(partial.exists())
                self.assertEqual(checkpoint.completed(),set())
                with ZipFile(partial,'w') as archive:
                    write_array(archive,'pipeline_signature','signature')
                self.assertEqual(checkpoint.completed(),set())

    def test_different_valid_checkpoint_is_not_quarantined_or_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); destination=root/'final.npz'; partial=destination.with_suffix('.npz.part')
            with ZipFile(partial,'w') as archive:
                write_array(archive,'pipeline_signature','different signature')
            before=partial.read_bytes()
            with self.assertRaisesRegex(ValueError,'different HDF5 inputs/settings'):
                pipeline.Checkpoint(destination,root,'signature')
            self.assertEqual(partial.read_bytes(),before)
            self.assertEqual(list(root.glob('*.preserved-*')),[])

    def test_video_run_restarts_unreadable_partial_and_finishes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=self.fixtures(root); (root/'out').mkdir()
            partial=root/'out/openworm_video_trimmed_890_dataset.npz.part'
            partial.write_bytes(b'partial ZIP without central directory')
            paths=self.run_pipeline(root,raw,dataset='video')
            self.assertEqual(len(paths),1)
            self.assertTrue(paths[0].is_file())
            self.assertFalse(partial.exists())
            preserved=list((root/'out').glob('*.preserved-*'))
            self.assertEqual(len(preserved),1)
            self.assertEqual(preserved[0].read_bytes(),b'partial ZIP without central directory')
            with np.load(paths[0],allow_pickle=False) as archive:
                self.assertEqual(archive['data_000000'].shape,(30,2,3))

    def test_zip_journal_recovers_killed_append_and_preserves_chunk_on_write_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cp = pipeline.Checkpoint(root/'final.npz', root, 'signature')
            with ZipFile(cp.part, 'w') as output:
                write_array(output, 'pipeline_signature', 'signature')
            before = cp.part.read_bytes()
            with ZipFile(cp.part) as archive:
                offset = archive.start_dir
            cp.tail.write_bytes(before[offset:])
            pipeline.atomic_json(cp.journal, dict(signature='signature',offset=offset,tail_size=len(before)-offset))
            with cp.part.open('r+b') as output:
                output.seek(offset)
                output.write(b'interrupted append overwritten central directory')
            cp = pipeline.Checkpoint(root/'final.npz', root, 'signature')
            self.assertEqual(cp.part.read_bytes(), before)
            chunk = root/'chunk.npz'
            with ZipFile(chunk,'w') as output:
                write_array(output,'data_000000',np.ones((4,2)))
            with patch('openworm.utils.openworm_zip.append_entry',side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError,'disk full'):
                    cp.commit(0, chunk)
            self.assertTrue(chunk.exists())
            self.assertEqual(cp.part.read_bytes(), before)
            self.assertEqual(cp.completed(),set())
            cp.commit(0,chunk)
            self.assertEqual(cp.completed(),{0})
            self.assertFalse(chunk.exists())

    def test_existing_compatible_non_pipeline_outputs_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=self.fixtures(root)
            paths=self.run_pipeline(root,raw)
            for path in paths:
                with ZipFile(path) as source:
                    entries={n:source.read(n) for n in source.namelist() if not n.startswith('pipeline_')}
                with ZipFile(path,'w',compression=ZIP_BZIP2) as output:
                    for name,value in entries.items():
                        output.writestr(name,value)
            with patch.object(pipeline,'pack_record',side_effect=AssertionError('must skip compatible output')):
                self.run_pipeline(root,raw)


if __name__=='__main__':
    unittest.main()
