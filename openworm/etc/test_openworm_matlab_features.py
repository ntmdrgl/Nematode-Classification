"""Validate source parity, MAT interchange and the shared gap rules without MATLAB."""
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading
import h5py

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import openworm.openworm_matlab_features as script


class MatlabFeaturesTests(unittest.TestCase):
    def test_existing_csv_skips_matlab_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / 'input/strain/record_1'
            record.mkdir(parents=True)
            output = root / 'output/strain/record_1.csv'
            output.parent.mkdir(parents=True)
            output.write_bytes(b'existing record')
            with patch.object(sys, 'argv', ['script', '--input-dir', str(root / 'input'),
                                            '--output-dir', str(root / 'output')]), \
                    patch.object(script, 'matlab_executable', side_effect=AssertionError('must skip')):
                with patch('builtins.print') as output_print:
                    self.assertEqual(script.main(), 0)
                self.assertEqual(output_print.call_count, 1)
                self.assertIn('skipped 1', output_print.call_args.args[0])
            with patch.object(script.subprocess, 'run', side_effect=AssertionError('must skip')):
                message = script.process_record(record / 'worm.hdf5', root / 'output', 'matlab')
            self.assertIn('SKIP', message)
            self.assertEqual(output.read_bytes(), b'existing record')

    def test_failed_csv_write_does_not_publish_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = MagicMock()
            writer.writerow.side_effect = [None, OSError('test write failed')]
            with patch.object(script.subprocess, 'run'), \
                    patch.object(script, 'loadmat', return_value=self.raw()), \
                    patch.object(script.csv, 'writer', return_value=writer):
                with self.assertRaisesRegex(OSError, 'test write failed'):
                    script.process_record(root / 'strain/record_1/worm.hdf5', root / 'output', 'matlab')
            self.assertEqual(list((root / 'output').rglob('*.csv')), [])
            self.assertEqual(list((root / 'output').rglob('*.part')), [])

    def test_csv_published_after_all_rows_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output/strain/record_1.csv'
            replace = script.os.replace
            def publish(temporary, destination):
                self.assertFalse(output.exists())
                self.assertEqual(len(Path(temporary).read_text().splitlines()), 17)
                replace(temporary, destination)
            with patch.object(script.subprocess, 'run'), \
                    patch.object(script, 'loadmat', return_value=self.raw()), \
                    patch.object(script.os, 'replace', side_effect=publish):
                script.process_record(root / 'strain/record_1/worm.hdf5', root / 'output', 'matlab')
            self.assertTrue(output.is_file())

    def test_streamed_frame_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = MagicMock()
            child.stdout = iter(['Progress (  1/1  ): 6.2% (1/16) | Elapsed: 00:00:01\n',
                                 'Progress (  1/1  ): 100.0% (16/16) | Elapsed: 00:00:02\n'])
            child.wait.return_value = 0
            progress = []
            with patch.object(script.subprocess, 'Popen') as launch, \
                    patch.object(script, 'loadmat', return_value=self.raw()):
                launch.return_value.__enter__.return_value = child
                script.process_record(root / 'strain/record_1/worm.hdf5', root / 'output', 'matlab',
                                      progress_callback=lambda done, total: progress.append((done, total)))
            self.assertEqual(progress, [(1, 16), (16, 16)])

    def test_concurrent_record_frame_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = []
            for i in range(2):
                video = root / f'strain/record_{i}/worm.hdf5'
                video.parent.mkdir(parents=True)
                with h5py.File(video, 'w') as source:
                    source.create_dataset('mask', shape=(10, 2, 2), dtype='uint8')
                videos.append(video)
            barrier = threading.Barrier(2)
            def worker(video, *args, show_progress, progress_callback):
                barrier.wait(timeout=5)
                progress_callback(1, 5)
                progress_callback(5, 5)
                return f'Completed {video}'
            self.assertEqual(script.run_records(videos, worker, (root, 'matlab', 5, 'old'),
                                                2, 'Test records'), 0)

    def test_variant_selects_matlab_function(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'test/record_1/worm.hdf5'
            for variant, function in (("old", "openworm_matlab_features"), ("new", "openworm_matlab_features_new")):
                with patch.object(script.subprocess, 'run') as run, \
                        patch.object(script, 'loadmat', return_value=self.raw()):
                    script.process_record(video, root / variant, 'matlab', variant=variant)
                self.assertIn(function + '(', run.call_args.args[0][-1])

    def test_new_variant_default_output_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / 'test/record_1'
            record.mkdir(parents=True)
            (record / 'worm.hdf5').touch()
            (record / 'worm_features.hdf5').touch()
            with patch.object(sys, 'argv', ['script', '--input-dir', directory, '--variant', 'new']), \
                    patch.object(script, 'matlab_executable', return_value='matlab'), \
                    patch.object(script, 'run_records', return_value=0) as run:
                self.assertEqual(script.main(), 0)
            self.assertEqual(run.call_args.args[2][0], (script.ROOT / 'data/matlab_features_new').resolve())
            self.assertEqual(run.call_args.args[2][3], 'new')

    def test_frame_worker_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(script.subprocess, 'run') as run, \
                    patch.object(script, 'loadmat', return_value=self.raw()):
                script.process_record(root / 'strain/record_1/worm.hdf5', root / 'output',
                                      'matlab', variant='new', frame_workers=3)
            self.assertIn("inf, 32, 3, 'Threads');", run.call_args.args[0][-1])

    def test_parallel_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / 'test/record_1'
            record.mkdir(parents=True)
            (record / 'worm.hdf5').touch()
            (record / 'worm_features.hdf5').touch()
            with patch.object(sys, 'argv', ['script', '--input-dir', directory, '--workers', '2']), \
                    patch.object(script, 'matlab_executable', return_value='matlab'), \
                    patch.object(script, 'run_records', return_value=0) as run:
                self.assertEqual(script.main(), 0)
            self.assertEqual(run.call_args.args[0], [(record / 'worm.hdf5').resolve()])
            self.assertIs(run.call_args.args[1], script.process_record)
            self.assertEqual(run.call_args.args[3], 2)

    def test_parallel_capture_preserves_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'test/record_1/worm.hdf5'
            video.parent.mkdir(parents=True)
            video.touch()
            with patch.object(script.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                    patch.object(script, 'loadmat', return_value=self.raw()):
                script.process_record(video, root / 'serial', 'matlab', show_progress=True)
                script.process_record(video, root / 'parallel', 'matlab', show_progress=False)
            self.assertEqual((root / 'serial/test/record_1.csv').read_bytes(),
                             (root / 'parallel/test/record_1.csv').read_bytes())

    def test_parallel_matlab_failure_includes_output(self):
        def fail(*args, **kwargs):
            kwargs['stdout'].write('MATLAB test failure')
            return SimpleNamespace(returncode=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(script.subprocess, 'run', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'MATLAB test failure'):
                    script.process_record(Path(directory) / 'test/record_1/worm.hdf5',
                                          Path(directory), 'matlab', show_progress=False)

    def raw(self):
        return dict(feature_names=np.array(script.FEATURES, dtype=object),
                    feature_values=np.ones((16,16)), observed_mask=np.ones((16,16), bool),
                    time_s=np.arange(16)*.04, frame_number=np.arange(16), fps=25.)

    def test_feature_names_match_original_preprocessing(self):
        text=(script.ROOT/'openworm/old/openworm_dataset_preprocessing.m').read_text()
        selected=re.search(r"features_local\s*=\s*\{(.*?)\};",text,re.S).group(1)
        self.assertEqual(re.findall(r"'([^']+)'",selected),script.FEATURES)

    def test_collection_calculations_preserved_verbatim(self):
        old=(script.ROOT/'openworm/old/openworm_dataset_collection_batch_clean.m').read_text()
        new=(script.ROOT/'openworm/utils/openworm_matlab_features.m').read_text()
        body=old[old.index('    % --- read file, preprocess info'):old.index('    feature_names = {')]
        body=body.replace('filenames_features(k)','featuresFile').replace('filenames(k)','videoFile')
        body=body.replace('min(num_frames, max_frames)','min(num_frames, maxFrames)')
        body=body.replace('k, min(numel(filenames), max_files),','1, 1,')
        # Undo the two I/O/cache-only optimizations before comparing every
        # original calculation and branch against the legacy frame loop.
        new = re.sub(r"        % OPTIMIZED_READ_BEGIN:.*?        % OPTIMIZED_READ_END",
                     "        frame = h5read(videoFile, '/mask', [1 1 i], [inf inf 1]);", new, flags=re.S)
        new = new.replace('closing_disk = strel(\'disk\', 3);', '')
        new = new.replace('closing_disk)', "strel('disk', 3))")
        new = re.sub(r"    block_frames =.*?    frame_block_end = 0;\n", '', new, flags=re.S)
        normalized = lambda text: '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        self.assertIn(normalized(body), normalized(new))

    def test_mat_roundtrip_gap_masks_and_columns(self):
        raw=self.raw()
        raw['observed_mask'][2:4,0]=False
        raw['observed_mask'][6:12,1]=False
        raw['feature_values'][2:4,0]=999  # Legacy-filled values must not hide missingness.
        raw['feature_values'][6:12,1]=999
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'features.mat'
            savemat(path,raw)
            times,values,status=script.preprocess_extracted(loadmat(path,simplify_cells=True))
        self.assertEqual(values.shape,(16,16))
        np.testing.assert_allclose(np.diff(times),.04)
        np.testing.assert_array_equal(values[2:4,0],1)
        self.assertTrue(np.all(status[2:4,0]=='interpolated'))
        np.testing.assert_array_equal(values[6:12,1],0)
        self.assertTrue(np.all(status[6:12,1]=='zeroed'))

    def test_dropped_frames_and_orientation_period(self):
        raw=self.raw()
        raw['feature_values'][0,15]=89
        raw['feature_values'][1,15]=-89
        raw['frame_number']=np.r_[0,np.arange(2,17)]
        raw['time_s']=raw['frame_number']*.04
        _,values,status=script.preprocess_extracted(raw)
        self.assertAlmostEqual(abs(values[1,15]),90)
        self.assertEqual(status[1,15],'interpolated')
        self.assertTrue(np.isfinite(values).all())

    def test_missing_video_skips_without_starting_matlab(self):
        with tempfile.TemporaryDirectory() as directory:
            record=Path(directory)/'test/record_1'
            record.mkdir(parents=True)
            (record/'worm_features.hdf5').touch()
            with patch.object(sys,'argv',['script','--input-dir',directory]), \
                    patch.object(script,'matlab_executable',side_effect=AssertionError('MATLAB must not start')):
                with patch('builtins.print') as output_print:
                    self.assertEqual(script.main(),0)
                self.assertEqual(output_print.call_count, 1)
                self.assertIn('skipped 1', output_print.call_args.args[0])


if __name__=='__main__':
    unittest.main()
