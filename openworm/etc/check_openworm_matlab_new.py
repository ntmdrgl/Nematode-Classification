"""Bounded real-MATLAB regression check using a 65-frame local video fixture.

python openworm/etc/check_openworm_matlab_new.py --video "path/to/worm.hdf5"
Requires MATLAB and Image Processing Toolbox. Writes only temporary fixtures.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import subprocess

import h5py
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openworm.openworm_matlab_features import matlab_executable, matlab_string, preprocess_extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--frame-workers', type=int, default=1)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parents[1] / "utils"
    original = scripts / 'openworm_matlab_features.m'
    before = hashlib.sha256(original.read_bytes()).digest()
    features = args.video.with_name(args.video.stem + '_features.hdf5')
    with tempfile.TemporaryDirectory(prefix='openworm-new-check-') as directory:
        root = Path(directory)
        with h5py.File(args.video, 'r') as video, h5py.File(features, 'r') as source:
            raw = video['timestamp/raw'][:65].reshape(-1)
            timestamps = source['features_timeseries']['timestamp'][:]
            exists, indices = np.isin(raw, timestamps), np.searchsorted(timestamps, raw)
            assert exists.all(), 'Choose a fixture with complete initial feature coverage'
            table = source['features_timeseries'][indices]
            coords = {name: source['coordinates/' + name][indices]
                      for name in ('skeletons', 'dorsal_contours', 'ventral_contours')}
            scale = float(abs(video['mask'].attrs['microns_per_pixel']))
            table_attrs = dict(source['features_timeseries'].attrs)
            assert table_attrs['xy_units'] in (b'micrometers', 'micrometers')
            for variant in ('base', 'shifted', 'pixels', 'duplicates', 'extra_time', 'conflicting_time'):
                rows = np.arange(65)
                if variant == 'duplicates':
                    rows = np.r_[rows[:20], 19, rows[20:], 64]
                with h5py.File(root / f'{variant}.hdf5', 'w') as out:
                    mask = out.create_dataset('mask', data=video['mask'][:65][rows], compression='gzip')
                    mask[10:12] = 0  # Exercise skipped segmentation and tracking state.
                    for key, value in video['mask'].attrs.items():
                        mask.attrs[key] = value
                    frame_ids = raw[rows]
                    times = video['timestamp/time'][:65].reshape(-1)[rows]
                    if variant == 'extra_time':
                        frame_ids = np.r_[frame_ids, frame_ids[-1] + 1]
                        times = np.r_[times, times[-1] + .04]
                    if variant == 'conflicting_time':
                        times[20] = times[19]
                    out.create_dataset('timestamp/raw', data=frame_ids)
                    out.create_dataset('timestamp/time', data=times)
                    video.copy('stage_log', out)
                keep = np.ones(65, dtype=bool)
                if variant == 'shifted':
                    keep[:7] = False
                    keep[20:22] = False
                with h5py.File(root / f'{variant}_features.hdf5', 'w') as out:
                    ds = out.create_dataset('features_timeseries', data=table[keep])
                    for key, value in table_attrs.items():
                        ds.attrs[key] = value
                    if variant == 'pixels':
                        ds.attrs['xy_units'] = 'pixels'
                        ds.attrs['microns_per_pixel'] = scale
                    for name, values in coords.items():
                        values = values[keep].astype(float)
                        if variant == 'pixels':
                            values /= scale
                        out.create_dataset('coordinates/' + name, data=values)
        calls = [f'addpath({matlab_string(scripts)});']
        # Analytic chain-link cases: 4 axial links, 4 diagonals, and a mixed
        # open chain with 2 axial + 2 diagonal links (not closed perimeters).
        calls.append("s=false(9); s(5,2:6)=true; d=bwdistgeodesic(s,2,5,'quasi-euclidean'); "
                     "assert(abs(double(d(5,6))-4)<1e-5); "
                     "s=false(9); s(sub2ind(size(s),2:6,2:6))=true; "
                     "d=bwdistgeodesic(s,2,2,'quasi-euclidean'); "
                     "assert(abs(double(d(6,6))-4*sqrt(2))<1e-5); "
                     "s=false(9); s(sub2ind(size(s),[2 2 3 4 4],[2 3 4 5 6]))=true; "
                     "d=bwdistgeodesic(s,2,2,'quasi-euclidean'); "
                     "assert(abs(double(d(4,6))-(2+2*sqrt(2)))<1e-5);")
        for function, variant, output in (
                ('openworm_matlab_features', 'base', 'old'),
                ('openworm_matlab_features_new', 'base', 'new'),
                ('openworm_matlab_features_new', 'shifted', 'shifted'),
                ('openworm_matlab_features_new', 'pixels', 'pixels'),
                ('openworm_matlab_features_new', 'duplicates', 'duplicates'),
                ('openworm_matlab_features_new', 'extra_time', 'extra_time')):
            calls.append(f'{function}({matlab_string(root / (variant + ".hdf5"))}, '
                         f'{matlab_string(root / (output + ".mat"))}, {67 if variant == "duplicates" else 65}'
                         f'{", 32, " + str(args.frame_workers) if function.endswith("_new") else ""});')
        calls.append(f"rejected=false; try; openworm_matlab_features_new("
                     f"{matlab_string(root / 'conflicting_time.hdf5')}, "
                     f"{matlab_string(root / 'bad.mat')}, 65); catch e; "
                     "rejected=contains(e.message,'must increase'); end; assert(rejected);")
        subprocess.run([matlab_executable(), '-wait', '-noFigureWindows', '-batch', ' '.join(calls)], check=True)
        if args.frame_workers > 1:
            serial_calls = [f'addpath({matlab_string(scripts)});']
            for variant in ('base', 'shifted', 'pixels'):
                serial_calls.append(f'openworm_matlab_features_new({matlab_string(root / (variant + ".hdf5"))}, '
                                    f'{matlab_string(root / (variant + "_serial.mat"))}, 65, 32, 1);')
            subprocess.run([matlab_executable(), '-wait', '-noFigureWindows', '-batch', ' '.join(serial_calls)], check=True)
            for variant, output in (('base', 'new'), ('shifted', 'shifted'), ('pixels', 'pixels')):
                serial = loadmat(root / (variant + '_serial.mat'), simplify_cells=True)
                parallel = loadmat(root / (output + '.mat'), simplify_cells=True)
                for key in serial:
                    if not key.startswith('__'):
                        np.testing.assert_equal(serial[key], parallel[key], err_msg=f'{variant}/{key}')
        results = {name: loadmat(root / (name + '.mat'), simplify_cells=True)
                   for name in ('old', 'new', 'shifted', 'pixels', 'duplicates', 'extra_time')}
        for variant in ('duplicates', 'extra_time'):
            for key in ('feature_values', 'observed_mask', 'time_s', 'frame_number'):
                np.testing.assert_equal(results[variant][key], results['new'][key], err_msg=f'{variant}/{key}')
        old, new = results['old']['feature_values'], results['new']['feature_values']
        for column in (10, 11, 14):
            np.testing.assert_allclose(new[:, column], old[:, column] * scale, equal_nan=True)
        np.testing.assert_allclose(new[:, 13], old[:, 13] * scale**2, equal_nan=True)
        np.testing.assert_allclose(new[:, [9, 12, 15]], old[:, [9, 12, 15]], equal_nan=True)
        valid_length = results['new']['observed_mask'][:, 0].astype(bool)
        assert valid_length.any() and np.all(new[valid_length, 0] > 0)
        np.testing.assert_allclose(new[:, 5:8], old[:, 5:8] / results['old']['fps'],
                                   atol=.02, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(new, results['pixels']['feature_values'], atol=1e-8, equal_nan=True)
        shifted = results['shifted']
        missing = np.r_[np.arange(7), 20, 21]
        assert not shifted['observed_mask'][missing, 1:8].any()
        observed = shifted['observed_mask'][:, 1].astype(bool)
        expected = np.linalg.norm(coords['skeletons'][:, 0].astype(float) -
                                  coords['skeletons'][:, -1].astype(float), axis=1)
        np.testing.assert_allclose(shifted['feature_values'][observed, 1], expected[observed])
        # At valid adjacent image/coordinate rows, speed is distance / actual dt.
        centroid = np.concatenate(list(coords.values()), axis=1).astype(float).mean(axis=1)
        expected_speed = np.linalg.norm(np.diff(centroid, axis=0), axis=1) / np.diff(results['new']['time_s'])
        valid = results['new']['observed_mask'][1:, 2].astype(bool)
        np.testing.assert_allclose(new[1:, 2][valid], expected_speed[valid], atol=1e-8)
        for name in ('new', 'shifted', 'pixels'):
            grid, values, status = preprocess_extracted(results[name])
            assert values.shape == status.shape and values.shape[1] == 16
            assert np.isfinite(values).all() and np.allclose(np.diff(grid), .04)
        assert hashlib.sha256(original.read_bytes()).digest() == before
        print('PASS: physical units, net displacement, actual-time speeds, shifted/missing '
              'frame alignment, repeated frames, extra timestamps, conflicting-time rejection, '
              'pixel-coordinate conversion, Python preprocessing, and unchanged original.')


if __name__ == '__main__':
    main()
