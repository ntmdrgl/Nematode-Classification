"""Validate source parity, MAT interchange and the shared gap rules without MATLAB."""
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openworm_matlab_features as script


class MatlabFeaturesTests(unittest.TestCase):
    def raw(self):
        return dict(feature_names=np.array(script.FEATURES, dtype=object),
                    feature_values=np.ones((16,16)), observed_mask=np.ones((16,16), bool),
                    time_s=np.arange(16)*.04, frame_number=np.arange(16), fps=25.)

    def test_feature_names_match_original_preprocessing(self):
        text=(script.ROOT/'openworm/old/openworm_dataset_preprocessing.m').read_text()
        selected=re.search(r"features_local\s*=\s*\{(.*?)\};",text,re.S).group(1)
        self.assertEqual(re.findall(r"'([^']+)'",selected),script.FEATURES)

    def test_collection_frame_loop_preserved_verbatim(self):
        old=(script.ROOT/'openworm/old/openworm_dataset_collection_batch_clean.m').read_text()
        new=(script.ROOT/'openworm/openworm_matlab_features.m').read_text()
        body=old[old.index('    % --- read file, preprocess info'):old.index('    feature_names = {')]
        body=body.replace('filenames_features(k)','featuresFile').replace('filenames(k)','videoFile')
        body=body.replace('min(num_frames, max_frames)','min(num_frames, maxFrames)')
        body=body.replace('k, min(numel(filenames), max_files),','1, 1,')
        self.assertIn(body.strip(),new)

    def test_mat_roundtrip_gap_masks_and_columns(self):
        raw=self.raw()
        raw['observed_mask'][2:4,0]=False
        raw['observed_mask'][6:12,1]=False
        raw['feature_values'][2:4,0]=999  # Legacy-filled values must not hide missingness.
        raw['feature_values'][6:12,1]=999
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'features.mat'
            savemat(path,raw)
            times,values,status=script.preprocess_extracted(loadmat(path,simplify_cells=True),1,0)
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
        _,values,status=script.preprocess_extracted(raw,1,0)
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
                self.assertEqual(script.main(),0)


if __name__=='__main__':
    unittest.main()
