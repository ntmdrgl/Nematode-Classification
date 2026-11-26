"""Stream grayscale video NPZ records through ImageNet ResNet-34 (512 features).

python openworm/openworm_resnet_features.py --workers 2
python openworm/openworm_resnet_features.py --estimate-only

Output is float32, pooled before the classifier. Checkpoints live beside the
output in a .work directory. Rerun the same command to resume checkpoints. Compressed
input must be decompressed up to the saved frame when resuming. Final NPZ packing
restarts after interruption; extracted features are retained. Stop all workers
before restarting. Do not run two instances against the same output.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import struct
import time
import tempfile
import threading
import sys
from zipfile import ZipFile, ZIP_DEFLATED

# Support direct script execution and package imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openworm.utils.openworm_zip import append_entry

DEFAULT_INPUT = Path(r'E:\nematode classification\openworm_video_trimmed_890_dataset.npz')


def header(stream):
    """Read NPY shape without importing NumPy or reading pixel payloads."""
    if stream.read(6) != b'\x93NUMPY':
        raise ValueError('Invalid NPY magic')
    version = stream.read(2)
    width = 2 if version == b'\x01\x00' else 4
    size = struct.unpack('<H' if width == 2 else '<I', stream.read(width))[0]
    return ast.literal_eval(stream.read(size).decode('utf-8' if version == b'\x03\x00' else 'latin1'))


def read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if not block:
            raise ValueError('Truncated video payload')
        result.extend(block)
    return result


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare_frames(pixels, reference=False):
    """Keep pretrained PIL geometry; vectorize tensor conversion/normalization."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F
    if reference:
        return torch.stack([_transform(Image.fromarray(frame).convert('RGB')) for frame in pixels])
    def resize_frame(frame):
        image = Image.fromarray(frame).convert('RGB')
        image = F.resize(image, _transform.resize_size, interpolation=_transform.interpolation,
                         antialias=_transform.antialias)
        image = F.center_crop(image, _transform.crop_size)
        return np.asarray(image)
    images = list(_image_pool.map(resize_frame, pixels)) if _image_pool else [resize_frame(frame) for frame in pixels]
    batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().float().div_(255)
    mean = batch.new_tensor(_transform.mean)[None, :, None, None]
    std = batch.new_tensor(_transform.std)[None, :, None, None]
    batch = batch.sub_(mean).div_(std)
    global _preprocess_checked
    if not _preprocess_checked:
        expected = prepare_frames(pixels, reference=True)
        if not torch.equal(expected, batch):
            raise ValueError('Batched preprocessing differs from pretrained reference')
        _preprocess_checked = True
    return batch


def profile(source, device, threads, batch_size, batches, preprocess_workers):
    """Bounded, non-writing stage benchmark; does not touch extraction checkpoints."""
    import numpy as np
    import torch
    initialize(device, threads, None, None, preprocess_workers)
    with ZipFile(source) as archive:
        name = next(n for n in archive.namelist() if n.startswith('data_') and n.endswith('.npy'))
        with archive.open(name) as stream:
            info = header(stream)
            shape = info['shape']
            if info['descr'] != '|u1' or info['fortran_order'] or len(shape) != 3:
                raise ValueError('Profiling requires C-order uint8 grayscale frames')
            count = min(batch_size, shape[0])
            pixels = np.frombuffer(read_exact(stream, count * shape[1] * shape[2]), dtype=np.uint8).reshape(count, *shape[1:])
            start = time.perf_counter()
            reference = prepare_frames(pixels, reference=True)
            legacy_seconds = time.perf_counter() - start
            prepare_frames(pixels)  # Validate equivalence outside the timed section.
            start = time.perf_counter()
            batch = prepare_frames(pixels)
            batch_seconds = time.perf_counter() - start
            error = (reference - batch).abs().max().item()
            if not torch.equal(reference, batch):
                raise ValueError(f'Batched preprocessing differs from reference: max error {error}')
            # Exclude model/cuDNN warmup from steady-state measurements.
            with torch.inference_mode():
                for _ in range(3):
                    _model(batch.to(device)).cpu()
            times = dict(read_decompress=0., preprocess=0., host_to_device=0., inference=0., device_to_host=0.)
            measured = 0
            remaining = shape[0] - count
            for _ in range(batches):
                count = min(batch_size, remaining)
                if count <= 0:
                    break
                start = time.perf_counter()
                pixels = np.frombuffer(read_exact(stream, count * shape[1] * shape[2]), dtype=np.uint8).reshape(count, *shape[1:])
                times['read_decompress'] += time.perf_counter() - start
                start = time.perf_counter()
                batch = prepare_frames(pixels)
                if device == 'cuda':
                    batch = batch.pin_memory()
                times['preprocess'] += time.perf_counter() - start
                start = time.perf_counter()
                tensor = batch.to(device, non_blocking=True)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times['host_to_device'] += time.perf_counter() - start
                start = time.perf_counter()
                with torch.inference_mode():
                    result = _model(tensor)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times['inference'] += time.perf_counter() - start
                start = time.perf_counter()
                result.cpu().numpy()
                times['device_to_host'] += time.perf_counter() - start
                measured += count
                remaining -= count
            total = sum(times.values())
            report = dict(device=device, batch_size=batch_size, preprocess_workers=preprocess_workers, measured_frames=measured,
                          stage_seconds=times, stage_percent={k: round(v / total * 100, 1) for k, v in times.items()},
                          sequential_frames_per_second=measured / total,
                          legacy_preprocess_seconds=legacy_seconds, batched_preprocess_seconds=batch_seconds,
                          preprocess_max_error=error,
                          note='Single worker, sequential stages; excludes writes and startup; production overlaps loading and inference.')
            print(json.dumps(report, indent=2), flush=True)


def extract(job):
    import numpy as np
    import torch
    from PIL import Image

    source, key, shape, dtype_string, work, batch_size, index, checkpoint_seconds = job
    work = Path(work)
    array_path = work / (key + '.npy')
    state_path = work / (key + '.json')
    done = json.loads(state_path.read_text())['frames'] if state_path.exists() else 0
    if not 0 <= done <= shape[0]:
        raise ValueError(f'Invalid checkpoint: {key}')
    features = np.lib.format.open_memmap(array_path, mode='r+' if done else 'w+',
                                       dtype=np.float32, shape=(shape[0], 512))
    if features.shape != (shape[0], 512) or features.dtype != np.float32:
        raise ValueError(f'Invalid feature checkpoint: {key}')
    if done == shape[0]:
        return key
    dtype = np.dtype(dtype_string)
    if dtype != np.dtype('uint8'):
        raise ValueError(f'Expected uint8 mask intensities, got {dtype}: {key}')
    frame_bytes = shape[1] * shape[2]
    with ZipFile(source) as archive, archive.open(key + '.npy') as stream:
        header(stream)
        # ZIP compressed streams have no random pixel access. Discard in chunks.
        remaining = done * frame_bytes
        while remaining:
            size = min(remaining, 8 * 1024**2)
            read_exact(stream, size)
            remaining -= size
        def prepare(count):
            pixels = np.frombuffer(read_exact(stream, count * frame_bytes), dtype=np.uint8)
            pixels = pixels.reshape(count, *shape[1:])
            batch = prepare_frames(pixels)
            return batch.pin_memory() if _device == 'cuda' else batch

        last_checkpoint = time.monotonic()
        # One producer overlaps ZIP reads and CPU preprocessing with GPU inference.
        # Only one future batch is buffered, keeping memory bounded.
        with ThreadPoolExecutor(max_workers=1) as loader:
            count = min(batch_size, shape[0] - done)
            pending_batch = loader.submit(prepare, count)
            while done < shape[0]:
                if _stop.is_set():
                    raise InterruptedError('Stopped; rerun to resume saved batches')
                batch = pending_batch.result()
                next_count = min(batch_size, shape[0] - done - count)
                if next_count:
                    pending_batch = loader.submit(prepare, next_count)
                with torch.inference_mode():
                    values = _model(batch.to(_device, non_blocking=True)).cpu().numpy()
                if values.shape != (count, 512) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid model features: {key}')
                features[done:done + count] = values
                done += count
                _counts[index] = done
                if done == shape[0] or time.monotonic() - last_checkpoint >= checkpoint_seconds:
                    features.flush()
                    # Persist features before publishing the completion marker.
                    with array_path.open('r+b') as feature_file:
                        os.fsync(feature_file.fileno())
                    save_json(state_path, {'frames': done})
                    last_checkpoint = time.monotonic()
                count = next_count
        if stream.read(1):
            raise ValueError(f'Trailing pixels: {key}')
    return key


def initialize(device, threads, stop, counts, preprocess_workers=4):
    global _model, _transform, _device, _stop, _counts, _preprocess_checked, _image_pool
    import atexit
    _image_pool = ThreadPoolExecutor(max_workers=preprocess_workers) if preprocess_workers > 1 else None
    if _image_pool:
        atexit.register(_image_pool.shutdown)
    _preprocess_checked = False
    _stop = stop
    _counts = counts
    import torch
    from torchvision.models import resnet34, ResNet34_Weights
    torch.set_num_threads(threads)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    _device = device
    weights = ResNet34_Weights.IMAGENET1K_V1
    _transform = weights.transforms()
    _model = resnet34(weights=weights)
    _model.fc = torch.nn.Identity()
    _model = _model.to(device).eval()


def inspect(source, progress=False):
    records = []
    with ZipFile(source) as archive:
        names = archive.namelist()
        if progress:
            from tqdm import tqdm
            names = tqdm(names, desc='Scanning archive', unit='entry')
        for name in names:
            if name.startswith('data_') and name.endswith('.npy'):
                with archive.open(name) as stream:
                    info = header(stream)
                shape = info['shape']
                if len(shape) != 3 or min(shape) < 1 or info['fortran_order'] or info['descr'] != '|u1':
                    raise ValueError(f'Expected C-order uint8 grayscale frames: {name}, {info}')
                records.append((name[:-4], shape, info['descr']))
    if not records:
        raise ValueError('No data_<index> frame stacks found')
    return records


def checkpoint_records(source, work, progress=False):
    """Use ZIP directory names and feature headers; never open video entries."""
    with ZipFile(source) as archive:
        keys = [name[:-4] for name in archive.namelist()
                if name.startswith('data_') and name.endswith('.npy')]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('Missing or duplicate video record keys in archive directory')
    if progress:
        from tqdm import tqdm
        keys = tqdm(keys, desc='Reading checkpoint headers', unit='record')
    records = []
    for key in keys:
        path = work / (key + '.npy')
        if not path.is_file():
            raise ValueError(f'Missing checkpoint for {key}; resume extraction first')
        with path.open('rb') as stream:
            info = header(stream)
        shape = info['shape']
        if len(shape) != 2 or shape[0] < 1 or shape[1] != 512:
            raise ValueError(f'Invalid feature checkpoint shape: {path}')
        records.append((key, (shape[0],), info['descr']))
    return records


def validate_checkpoints(work, records, progress=False):
    """Reject incomplete or malformed features before starting final packing."""
    if progress:
        from tqdm import tqdm
        records = tqdm(records, desc='Validating checkpoints', unit='record')
    for key, shape, _ in records:
        state = work / (key + '.json')
        path = work / (key + '.npy')
        if not state.is_file() or not path.is_file():
            raise ValueError(f'Missing checkpoint for {key}; resume extraction first')
        done = json.loads(state.read_text())['frames']
        if type(done) is not int or done != shape[0]:
            raise ValueError(f'Incomplete checkpoint for {key}: {done}/{shape[0]} frames')
        with path.open('rb') as stream:
            info = header(stream)
            offset = stream.tell()
        if (info['shape'] != (shape[0], 512) or info['descr'] not in ('<f4', '=f4')
                or info['fortran_order'] or path.stat().st_size != offset + shape[0] * 512 * 4):
            raise ValueError(f'Invalid feature checkpoint: {path}')


def compress_checkpoint(path, part, key, index, counts, stop):
    """Compress one checkpoint with bounded buffers; zlib releases the GIL."""
    with ZipFile(part, 'w', compression=ZIP_DEFLATED, compresslevel=1) as archive:
        with path.open('rb') as src, archive.open(key + '.npy', 'w', force_zip64=True) as dst:
            while block := src.read(1024**2):
                if stop.is_set():
                    raise InterruptedError('Packing stopped; feature checkpoints are unchanged')
                dst.write(block)
                counts[index] += len(block)
    return part, key


def pack_feature_records(target, work, records, workers, progress):
    """Bound compression jobs/temp parts to worker count; merge without recompression."""
    counts = [0] * len(records)
    stop = threading.Event()
    jobs = iter(enumerate(records))
    with tempfile.TemporaryDirectory(prefix='.pack-', dir=work) as directory:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = set()

            def submit_next():
                item = next(jobs, None)
                if item is None:
                    return
                index, (key, _, _) = item
                part = Path(directory) / f'{index:06d}.npz'
                pending.add(pool.submit(compress_checkpoint, work / (key + '.npy'), part,
                                        key, index, counts, stop))

            for _ in range(min(workers, len(records))):
                submit_next()
            try:
                while pending:
                    completed, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in completed:
                        part, key = future.result()
                        with ZipFile(part) as compressed:
                            append_entry(target, compressed, key + '.npy')
                        part.unlink()
                        submit_next()
                    progress.update(sum(counts) - progress.n)
            except BaseException:
                stop.set()
                for future in pending:
                    future.cancel()
                raise


def pack_records(source, output, work, records, workers=4):
    import numpy as np
    from tqdm import tqdm
    validate_checkpoints(work, records, progress=True)
    # Atomic publication. Never expose a partially written NPZ as final output.
    partial = output.with_suffix('.npz.part')
    print('Packing final NPZ (checkpoints retained for resume)...', flush=True)
    with ZipFile(source) as original, ZipFile(partial, 'w', compression=ZIP_DEFLATED, compresslevel=1) as target:
        for entry in tqdm(original.infolist(), desc='Copying metadata', unit='entry'):
            if entry.filename.startswith(('data_', 'compression_')) or entry.filename in ('feature_names.npy', 'array_axes.npy', 'source_dataset.npy'):
                continue
            with original.open(entry) as src, target.open(entry.filename, 'w', force_zip64=True) as dst:
                shutil.copyfileobj(src, dst, 8 * 1024**2)
        for name, value in [('feature_names', [f'resnet34_{i:03d}' for i in range(512)]),
                            ('array_axes', ['frame', 'feature']), ('resnet_model', 'resnet34'),
                            ('resnet_weights', 'IMAGENET1K_V1'), ('source_dataset', 'resnet34_features')]:
            with target.open(name + '.npy', 'w', force_zip64=True) as dst:
                np.lib.format.write_array(dst, np.asarray(value), allow_pickle=False)
        packing_bytes = sum((work / (key + '.npy')).stat().st_size for key, _, _ in records)
        with tqdm(total=packing_bytes, desc=f'Packing NPZ ({workers} workers)', unit='B', unit_scale=True, smoothing=0) as progress:
            pack_feature_records(target, work, records, workers, progress)
    os.replace(partial, output)
    print(f'Saved {output}. Checkpoints retained in {work}; remove that directory when no longer needed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=2, help='Parallel record/model processes; use 1 if GPU memory is limited')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--checkpoint-seconds', type=float, default=30, help='Persist progress every N seconds per worker; uncommitted batches repeat on resume')
    parser.add_argument('--preprocess-workers', type=int, default=4, help='PIL resize/crop threads per record worker; 1 disables image threading')
    parser.add_argument('--threads', type=int, default=2, help='CPU inference threads per worker')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--profile-batches', type=int, default=0, help='Benchmark N batches of the first record without writing features')
    parser.add_argument('--pack-workers', type=int, default=4, help='Parallel CPU compression threads for final packing (default: 4)')
    parser.add_argument('--pack-only', action='store_true', help='Validate and pack completed checkpoints without importing PyTorch or loading ResNet')
    parser.add_argument('--estimate-only', action='store_true', help='Read array headers only; no ML dependencies needed')
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.threads, args.preprocess_workers, args.pack_workers) < 1:
        parser.error('workers, batch-size, threads, preprocess-workers and pack-workers must be positive')
    if not 0 < args.checkpoint_seconds < float('inf'):
        parser.error('checkpoint-seconds must be finite and positive')
    source = args.input.resolve()
    output = (args.output or source.with_name('openworm_resnet_features_dataset.npz')).resolve()
    if source == output:
        parser.error('Output must differ from input')
    if args.pack_only and (args.profile_batches or args.estimate_only):
        parser.error('pack-only cannot be combined with profile-batches or estimate-only')
    if args.profile_batches:
        if args.profile_batches < 1:
            parser.error('profile-batches must be positive')
        import torch
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        profile(source, device, args.threads, args.batch_size, args.profile_batches, args.preprocess_workers)
        return
    records = None
    if not args.pack_only:
        print('Reading input archive headers (no video inference)...', flush=True)
        records = inspect(source, progress=not args.estimate_only)
        frames = sum(shape[0] for _, shape, _ in records)
        size = frames * 512 * 4
        print(f'{len(records):,} records; {frames:,} frames; float32 features: '
              f'{size / 10**9:.3f} GB ({size / 1024**3:.3f} GiB), before ZIP compression.', flush=True)
        if args.estimate_only:
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.with_suffix('.work')
    work.mkdir(exist_ok=True)
    # An OS lock releases even after a crash; the small lock file may remain.
    lock = (work / 'run.lock').open('a+b')
    lock.seek(0)
    lock.write(b'0')
    lock.flush()
    lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    stat = source.stat()
    signature = dict(source=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                     model='resnet34', weights='IMAGENET1K_V1', dtype='float32', preprocessing='weights.transforms RGB', version=1)
    manifest = work / 'manifest.json'
    if args.pack_only and not manifest.is_file():
        raise ValueError('Missing checkpoint manifest; resume extraction first')
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Input or extraction settings changed; use a different output path')
    if args.pack_only:
        print('Using checkpoint headers; skipping original video array scan.', flush=True)
        records = checkpoint_records(source, work, progress=True)
        pack_records(source, output, work, records, args.pack_workers)
        return
    save_json(manifest, signature)
    import numpy as np
    import torch
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    # Download once in parent to avoid competing worker downloads.
    from torchvision.models import ResNet34_Weights
    ResNet34_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    jobs = [(str(source), key, shape, dtype, str(work), args.batch_size, index, args.checkpoint_seconds)
            for index, (key, shape, dtype) in enumerate(records)]
    print(f'Using {args.workers} parallel {device} workers. Checkpoints: {work}', flush=True)
    context = multiprocessing.get_context('spawn')
    stop = context.Event()
    counts = context.Array('q', len(records), lock=False)
    for index, (key, shape, _) in enumerate(records):
        checkpoint = work / (key + '.json')
        count = json.loads(checkpoint.read_text())['frames'] if checkpoint.exists() else 0
        if not isinstance(count, int) or not 0 <= count <= shape[0]:
            raise ValueError(f'Invalid checkpoint: {key}')
        counts[index] = count
    from tqdm import tqdm
    with tqdm(total=frames, initial=sum(counts), desc='ResNet-34 frames', unit='frame',
              smoothing=0, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
              '[{elapsed} elapsed, ETA {remaining}, {rate_fmt}]') as progress:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                 initializer=initialize, initargs=(device, args.threads, stop, counts, args.preprocess_workers)) as pool:
            pending = {pool.submit(extract, job) for job in jobs}
            try:
                while pending:
                    completed, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    for future in completed:
                        future.result()
                    progress.update(sum(counts) - progress.n)
                    progress.refresh()
            except BaseException:
                stop.set()
                for future in pending:
                    future.cancel()
                raise
    pack_records(source, output, work, records, args.pack_workers)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
