"""Build only the five trimmed OpenWorm datasets directly from original HDF5s.

Run from the repository root; see openworm/README.md. No CSVs, downloads,
compressed video exports or untrimmed dataset archives are needed/generated.
"""

# Allow both `python openworm/<script>.py` and package imports/module execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from zipfile import BadZipFile, ZipFile, ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA

import h5py
import numpy as np
from tqdm import tqdm

from openworm.utils.openworm_alignment import reference_timestamps, sample_count, rebuild_record
from openworm.utils.openworm_archive_staging import StagedJobs
from openworm.openworm_dataset import ANGLES, EIGEN, ordered_results, write_array, write_metadata
from openworm.openworm_manual_features import feature_names, preprocess_feature, ORIENTATIONS
from openworm.openworm_matlab_features import FEATURES
from openworm.openworm_skeleton_angles import read_record, preprocess, spatial_resample
from openworm.openworm_trim import MIN_SAMPLES, inspect_dataset
from openworm.utils.openworm_video_dataset import load_array, video_metadata
from openworm.utils.openworm_video_staging import copy_video
from openworm.utils.openworm_video_trim import hdf5_plan, pack_trim_hdf5_record
from openworm.utils.openworm_video_workers import packed_records

DATASETS = ('skeleton_angles', 'manual_features', 'eigenworms', 'matlab_features_new', 'video')
VERSION = 1


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def inspect_source(features):
    features = Path(features)
    _, times, _, _ = reference_timestamps(features)
    with h5py.File(features, 'r') as source:
        names = feature_names(source['features_timeseries'])
    return str(features), sample_count(times), names


def source_signature(path):
    info = path.stat()
    return str(path), info.st_size, info.st_mtime_ns


def discover(hdf5_dir, workers):
    files = sorted(hdf5_dir.glob('*/record_*/*_features.hdf5'))
    if not files:
        raise ValueError(f'No <strain>/record_*/*_features.hdf5 files in {hdf5_dir}')
    pairs = [(p.parent.parent.name, p.parent.name) for p in files]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Expected exactly one features HDF5 per strain/record')
    sources = list(tqdm(ordered_results(inspect_source, ((p,) for p in files), workers),
                        total=len(files), desc='Reading reference clocks', unit='record'))
    retained = []
    for filename, count, names in sources:
        if count <= MIN_SAMPLES:
            continue
        features = Path(filename)
        video = features.with_name(features.name.replace('_features.hdf5', '.hdf5'))
        if not video.is_file():
            raise ValueError(f'Missing original mask HDF5: {video}')
        manual = [n for n in names if n not in EIGEN]
        if len(manual) != 47 or len(names) != 53 or not set(EIGEN).issubset(names):
            raise ValueError(f'Expected 47 manual features and 6 eigen projections: {features}')
        retained.append(dict(record=[features.parent.parent.name, features.parent.name],
                             features=str(features), video=str(video), samples=count,
                             manual=manual, signatures=[source_signature(features), source_signature(video)]))
    if not retained:
        raise ValueError('No records longer than 890 seconds')
    if any(r['manual'] != retained[0]['manual'] for r in retained):
        raise ValueError('Manual feature names/order differ between records')
    return retained, min(r['samples'] for r in retained)



def discover_video_reference(hdf5_dir, reference):
    _, keys, sizes = inspect_dataset(reference, 48)
    if len(set(sizes.values())) != 1:
        raise ValueError('--reference-skeleton must be a trimmed archive with one common frame count')
    target = next(iter(sizes.values()))
    records = []
    for pair in keys:
        paths = list((hdf5_dir / pair[0] / pair[1]).glob('*_features.hdf5'))
        if len(paths) != 1:
            raise ValueError(f'Expected one features HDF5 for reference record {pair}')
        features = paths[0]
        video = features.with_name(features.name.replace('_features.hdf5', '.hdf5'))
        if not video.is_file():
            raise ValueError(f'Missing original mask HDF5: {video}')
        records.append(dict(record=list(pair), features=str(features), video=str(video), samples=None,
                            manual=[], signatures=[source_signature(features), source_signature(video)]))
    return records, target


def stage_input(record, arguments, directory, stop, callback):
    features, dataset, *rest = arguments
    features = Path(features)
    video = features.with_name(features.name.replace('_features.hdf5', '.hdf5'))
    staged_features = directory / features.name
    staged_video = directory / video.name
    copy_video(features, staged_features, stop, callback)
    if dataset in ('video', 'matlab_features_new'):
        copy_video(video, staged_video, stop, callback)
    else:
        # Feature extraction reads only the adjacent video's acquisition clock.
        # Preserve its full timestamp arrays (including their validity) exactly.
        with h5py.File(video, 'r') as source, h5py.File(staged_video, 'w') as output:
            for name in ('timestamp/raw', 'timestamp/time'):
                if name in source:
                    output.create_dataset(name, data=source[name][:])
    return (staged_features, dataset, *rest)


def pack_record(slot, arguments, directory, report):
    features, dataset, index, count, target, original_video, matlab, frame_workers, pool_type, signature = arguments
    features = Path(features)
    video = features.with_name(features.name.replace('_features.hdf5', '.hdf5'))
    part = Path(directory) / f'{index:06d}.npz.part'
    try:
        if dataset == 'video':
            if count is None:
                count = sample_count(reference_timestamps(features)[1])
            _, part, metric = pack_trim_hdf5_record(index, (video, ('', ''), features, count,
                                                          target, 'nearest', None, original_video), directory, report)
        else:
            if dataset == 'matlab_features_new':
                _, metadata, _, _ = hdf5_plan((video, ('', ''), features, count, target,
                                              'nearest', None, original_video))
                values = rebuild_record(video, metadata['target_time_s'], matlab, frame_workers, pool_type,
                                        lambda stage, done, total: report(done, total),
                                        temporary_dir=features.parent)
            else:
                times, skeletons = read_record(features)
                if dataset == 'skeleton_angles':
                    report(0, len(times))
                    _, values, _, _ = preprocess(times, skeletons)
                    report(len(times), len(times))
                else:
                    valid = np.isfinite(spatial_resample(skeletons)).all(axis=(1, 2))
                    with h5py.File(features, 'r') as source:
                        table = source['features_timeseries'][:]
                    names = (EIGEN if dataset == 'eigenworms' else
                             [n for n in feature_names(table) if n not in EIGEN])
                    columns = []
                    report(0, len(names))
                    for done, name in enumerate(names, 1):
                        _, column, _ = preprocess_feature(times, table[name], valid, name in ORIENTATIONS)
                        columns.append(column)
                        report(done, len(names))
                    values = np.column_stack(columns)
                if len(values) != count:
                    raise ValueError(f'Reference/extraction sample counts disagree: {features}')
                values = values[:target]
            if not np.isfinite(values).all() or len(values) != target:
                raise ValueError(f'Invalid trimmed {dataset} values: {features}')
            with ZipFile(part, 'w', compression=ZIP_DEFLATED, compresslevel=6) as output:
                write_array(output, f'data_{index:06d}', values)
            metric = 0.
        with ZipFile(part, 'a', compression=ZIP_DEFLATED) as output:
            write_array(output, 'pipeline_chunk_signature', signature)
            write_array(output, 'pipeline_chunk_index', index)
        return index, part, metric
    except BaseException:
        part.unlink(missing_ok=True)
        raise


class Checkpoint:
    """Append one trimmed record with a small rollback journal, never a full copy.

    Before ZIP append overwrites the old central directory, save that directory.
    A killed append can then roll back to the last closed, committed ZIP on resume.
    """
    def __init__(self, destination, state_dir, signature):
        self.destination, self.signature = destination, signature
        self.part = destination.with_suffix('.npz.part')
        self.journal = state_dir / (destination.stem + '.journal.json')
        self.tail = state_dir / (destination.stem + '.zip-tail')
        self.recover()
        self.validate_existing()

    def validate_existing(self):
        if not self.part.exists():
            return
        reason = None
        try:
            with ZipFile(self.part) as source:
                if 'pipeline_signature.npy' not in source.namelist():
                    reason = 'has no pipeline resume metadata'
                else:
                    signature = str(load_array(source, 'pipeline_signature').item())
        except (BadZipFile, EOFError, ValueError) as error:
            reason = f'has unreadable ZIP/resume metadata ({error})'
        if reason is None:
            if signature != self.signature:
                raise ValueError(f'Partial output belongs to different HDF5 inputs/settings: {self.part}')
            return
        # An older streaming ZIP, or a killed write without a recovery journal,
        # cannot be trusted as a resumable checkpoint. Preserve it unchanged.
        backup = self.part.with_name(self.part.name + '.preserved-' + uuid.uuid4().hex)
        if backup.parent.resolve() != self.part.parent.resolve():
            raise ValueError('Partial-file backup must stay in its original folder')
        self.part.rename(backup)
        print(f'Existing partial output {reason}. Preserved unchanged: {backup}', flush=True)
        print('Starting a fresh checkpoint; compatible completed worker chunks will still be reused.', flush=True)

    def recover(self):
        if not self.journal.exists():
            return
        info = json.loads(self.journal.read_text(encoding='utf-8'))
        if info['signature'] != self.signature:
            raise ValueError('Resume journal belongs to different inputs/settings')
        tail = self.tail.read_bytes()
        if len(tail) != info['tail_size'] or self.part.stat().st_size < info['offset']:
            raise ValueError('Incomplete ZIP recovery journal; preserve it and the partial output')
        with self.part.open('r+b') as output:
            output.seek(info['offset'])
            output.write(tail)
            output.truncate()
        self.journal.unlink()
        self.tail.unlink(missing_ok=True)

    def completed(self):
        if not self.part.exists():
            return set()
        with ZipFile(self.part) as source:
            if str(load_array(source, 'pipeline_signature').item()) != self.signature:
                raise ValueError(f'Partial output belongs to different HDF5 inputs/settings: {self.part}')
            prefix = 'pipeline_record_'
            return {int(n[len(prefix):-4]) for n in source.namelist()
                    if n.startswith(prefix) and n.endswith('.npy')}

    def commit(self, index, chunk):
        with ZipFile(self.part) as archive:
            offset = archive.start_dir
        with self.part.open('rb') as source:
            source.seek(offset)
            self.tail.write_bytes(source.read())
        atomic_json(self.journal, dict(signature=self.signature, offset=offset, tail_size=self.tail.stat().st_size))
        try:
            with ZipFile(self.part, 'a') as output, ZipFile(chunk) as source:
                from openworm.utils.openworm_zip import append_entry
                for name in source.namelist():
                    if not name.startswith('pipeline_chunk_'):
                        append_entry(output, source, name)
                write_array(output, f'pipeline_record_{index:06d}', True)
        except BaseException:
            self.recover()
            raise
        self.journal.unlink()
        self.tail.unlink(missing_ok=True)
        chunk.unlink(missing_ok=True)


def names_for(dataset, records):
    return dict(skeleton_angles=ANGLES, manual_features=records[0]['manual'],
                eigenworms=EIGEN, matlab_features_new=FEATURES, video=['mask'])[dataset]


def matching_output(path, dataset, records, target, signature):
    if not path.is_file():
        return False
    try:
        with ZipFile(path) as source:
            names = set(source.namelist())
            if 'pipeline_signature.npy' in names and str(load_array(source, 'pipeline_signature').item()) != signature:
                return False
            expected = [r['record'] for r in records]
            if list(zip(load_array(source, 'labels').tolist(), load_array(source, 'record_ids').tolist())) != list(map(tuple, expected)):
                return False
            keys = [f'data_{i:06d}' for i in range(len(records))]
            if load_array(source, 'record_keys').tolist() != keys or load_array(source, 'feature_names').tolist() != names_for(dataset, records):
                return False
            if dataset in ('video', 'matlab_features_new'):
                if load_array(source, 'alignment_reference').item() != 'skeleton_angles':
                    return False
                if not np.allclose(load_array(source, 'time_s'), np.arange(target)*.04, rtol=0, atol=1e-10):
                    return False
            if dataset == 'video' and load_array(source, 'gap_policy').item() != 'nearest_short_gap_black_long_gap':
                return False
            for key in keys:
                with source.open(key + '.npy') as image:
                    version = np.lib.format.read_magic(image)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        return False
                    shape, fortran, dtype = reader(image)
                if not shape or shape[0] != target or fortran or dtype.hasobject:
                    return False
                if dataset == 'video':
                    if len(shape) != 3 or any(n < 1 for n in shape) or source.getinfo(key+'.npy').compress_type not in (ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA):
                        return False
                elif shape != (target, len(names_for(dataset, records))) or dtype != np.dtype('float64'):
                    return False
        return True
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return False


def initialize(checkpoint, dataset, records, target, signature):
    if checkpoint.part.exists():
        return
    pairs = [r['record'] for r in records]
    temporary = checkpoint.part.with_suffix('.initializing')
    with ZipFile(temporary, 'w', compression=ZIP_DEFLATED) as output:
        if dataset == 'video':
            video_metadata(output, pairs)
            write_array(output, 'alignment_method', 'nearest')
            write_array(output, 'alignment_tolerance_s', np.inf)
            write_array(output, 'gap_policy', 'nearest_short_gap_black_long_gap')
            write_array(output, 'gap_threshold_s', .25)
        else:
            write_metadata(output, names_for(dataset, records), pairs)
        write_array(output, 'time_s', np.arange(target) * .04)
        write_array(output, 'alignment_reference', 'skeleton_angles')
        write_array(output, 'pipeline_signature', signature)
    os.replace(temporary, checkpoint.part)


def run_pipeline(hdf5_dir, data_dir, output_dir, staging_dir, workers=24,
                 matlab_workers=4, frame_workers=6, pool_type='threads', matlab=None, dataset='all', reference_skeleton=None):
    if dataset not in ('all', *DATASETS):
        raise ValueError(f'Unknown dataset: {dataset}')
    selected = DATASETS if dataset == 'all' else (dataset,)
    if min(workers, matlab_workers, frame_workers) < 1 or matlab_workers > 4:
        raise ValueError('Workers must be positive; --matlab-workers must be between 1 and 4')
    if pool_type not in ('threads', 'processes'):
        raise ValueError('Unknown MATLAB pool type')
    hdf5_dir, data_dir, output_dir, staging_dir = (Path(p).resolve() for p in (hdf5_dir, data_dir, output_dir, staging_dir))
    if reference_skeleton is not None and selected != ('video',):
        raise ValueError('--reference-skeleton requires --dataset video')
    reference = Path(reference_skeleton).resolve() if reference_skeleton is not None else None
    if selected == ('video',) and reference is None:
        reference = next((p for p in (
            output_dir / 'openworm_skeleton_angles_trimmed_890_dataset.npz',
            data_dir / 'openworm_skeleton_angles_trimmed_890_dataset.npz') if p.is_file()), None)
    if reference is not None:
        records, target = discover_video_reference(hdf5_dir, reference)
        print(f'Using existing skeleton reference: {reference}', flush=True)
    else:
        records, target = discover(hdf5_dir, workers)
    manifest = dict(version=VERSION, target=target, records=records)
    if reference is not None:
        manifest['skeleton_reference'] = source_signature(reference)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    state = data_dir / '.openworm-pipeline' / signature
    state.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(state / 'manifest.json', manifest)
    print(f'Keeping {len(records):,} records; {target:,} samples each at 25 Hz. Final trimmed datasets only.', flush=True)
    destinations = []
    for dataset in selected:
        destination = output_dir / f'openworm_{dataset}_trimmed_890_dataset.npz'
        destinations.append(destination)
        if matching_output(destination, dataset, records, target, signature):
            print(f'Skipping completed {dataset}: {destination}', flush=True)
            continue
        checkpoint = Checkpoint(destination, state, signature)
        initialize(checkpoint, dataset, records, target, signature)
        done = checkpoint.completed()
        if done - set(range(len(records))):
            raise ValueError('Invalid record indices in partial output')
        parts = staging_dir / '.openworm-pipeline' / signature / dataset
        parts.mkdir(parents=True, exist_ok=True)
        # Reuse worker results that finished just before an interruption.
        for i in range(len(records)):
            chunk = parts / f'{i:06d}.npz.part'
            if i in done:
                chunk.unlink(missing_ok=True)
            elif chunk.is_file():
                try:
                    with ZipFile(chunk) as source:
                        valid = (load_array(source, 'pipeline_chunk_signature').item() == signature
                                 and load_array(source, 'pipeline_chunk_index').item() == i)
                        if valid and source.testzip() is not None:
                            valid = False
                except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                    valid = False
                if valid:
                    checkpoint.commit(i, chunk)
                    done.add(i)
                else:
                    chunk.unlink(missing_ok=True)
        concurrent = matlab_workers if dataset == 'matlab_features_new' else workers
        jobs = [(tuple(r['record']), (Path(r['features']), dataset, i, r['samples'], target,
                                      Path(r['video']), matlab, frame_workers, pool_type, signature))
                for i, r in enumerate(records) if i not in done]
        with tqdm(total=len(records), initial=len(done), desc=f'Trimmed {dataset}', unit='record') as progress:
            if jobs:
                with StagedJobs(jobs, staging_dir, concurrent, progress, stage_input) as staged:
                    completed = packed_records(pack_record, staged, concurrent, parts, progress, staged.release, staged.poll)
                    try:
                        for i, chunk, metric in completed:
                            checkpoint.commit(i, chunk)
                            done.add(i)
                            if dataset == 'video':
                                progress.set_postfix_str(f'last real-frame offset {metric*1000:.3f} ms')
                            progress.update(1)
                    finally:
                        completed.close()
        if len(done) != len(records):
            raise ValueError(f'Incomplete {dataset}; rerun to resume')
        os.replace(checkpoint.part, destination)
        print(f'Saved {destination}', flush=True)
        if not any(parts.iterdir()):
            parts.rmdir()
    return destinations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('all', *DATASETS), default='all',
                        help='Build/resume all five datasets (default) or only the selected trimmed dataset')
    parser.add_argument('--reference-skeleton', type=Path,
                        help='Trimmed skeleton archive for video-only exact record/frame matching (auto-detected in output/data folders)')
    parser.add_argument('--hdf5-dir', type=Path, default=Path('E:/nematode classification/openworm'))
    parser.add_argument('--data-dir', type=Path, default=Path('D:/data'), help='Small resume manifests/journals only; no untrimmed datasets')
    parser.add_argument('--output-dir', type=Path, default=Path('E:/nematode classification'))
    parser.add_argument('--staging-dir', type=Path, default=Path('C:/openworm-staging'))
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--matlab-workers', type=int, default=4, help='Concurrent MATLAB sessions, maximum 4')
    parser.add_argument('--frame-workers', type=int, default=6)
    parser.add_argument('--pool-type', choices=('threads', 'processes'), default='threads')
    parser.add_argument('--matlab', type=Path, help='MATLAB executable override')
    args = parser.parse_args()
    run_pipeline(**vars(args))


if __name__ == '__main__':
    main()
