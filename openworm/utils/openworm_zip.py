"""Copy validated ZIP entries without a second decompression/compression pass.

ZIP metadata bookkeeping is isolated here because stdlib ZipFile has no public
raw-entry copy API. Callers validate input pixels/CRC in workers before merging;
compressed payload bytes, codec, CRC and sizes are copied unchanged. Local and
central ZIP64 headers are regenerated for the destination offsets and names.
"""
from copy import copy
import struct
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZIP_STORED, ZipFile

BUFFER_BYTES = 8 * 1024 ** 2


def _extra_without_offsets(extra, renamed):
    result = bytearray()
    offset = 0
    while offset < len(extra):
        if offset + 4 > len(extra):
            raise ValueError("Truncated ZIP extra-field header")
        kind, size = struct.unpack_from("<HH", extra, offset)
        end = offset + 4 + size
        if end > len(extra):
            raise ValueError("Truncated ZIP extra field")
        # ZIP64 includes old sizes/offsets, and Unicode-path fields refer to the
        # original name. ZipFile generates fresh ZIP64 and UTF-8 name fields.
        if kind != 1 and not (renamed and kind == 0x7075):
            result.extend(extra[offset:end])
        offset = end
    return bytes(result)


def append_entry(output, source, source_name, name=None, callback=None):
    """Append one already-compressed entry. No pixel arrays are materialized."""
    original = source.getinfo(source_name)
    name = source_name if name is None else name
    if (original.flag_bits & (1 | 0x20 | 0x40) or original.is_dir()
            or original.compress_type not in (ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA)):
        raise ValueError("Unsupported encrypted, patched or non-array ZIP entry")
    info = copy(original)
    info.filename = info.orig_filename = name
    info.extra = _extra_without_offsets(info.extra, name != source_name)
    info.flag_bits &= ~8  # Fresh headers contain known CRC/sizes; omit descriptors.
    info._end_offset = None
    with source._lock, output._lock:
        if source._writing or output._writing or output.fp is None:
            raise ValueError("Cannot merge ZIP entries while another entry is open for writing")
        if name in output.NameToInfo:
            raise ValueError(f"Duplicate output ZIP entry: {name}")
        source.fp.seek(original.header_offset)
        header = source.fp.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid source ZIP local header")
        method = struct.unpack_from("<H", header, 8)[0]
        if method != original.compress_type:
            raise ValueError("ZIP local/central compression methods disagree")
        filename_size, extra_size = struct.unpack_from("<HH", header, 26)
        source.fp.seek(filename_size + extra_size, 1)
        output.fp.seek(output.start_dir)
        info.header_offset = output.fp.tell()
        output._writecheck(info)
        output._didModify = True
        output.fp.write(info.FileHeader(zip64=True))
        remaining = original.compress_size
        if callback is not None:
            callback(0, original.compress_size)
        while remaining:
            block = source.fp.read(min(remaining, BUFFER_BYTES))
            if not block:
                raise ValueError("Truncated compressed ZIP payload")
            output.fp.write(block)
            remaining -= len(block)
            if callback is not None:
                callback(original.compress_size - remaining, original.compress_size)
        output.start_dir = output.fp.tell()
        output.filelist.append(info)
        output.NameToInfo[name] = info


def append_archive(output, path):
    with ZipFile(path) as source:
        for entry in source.infolist():
            append_entry(output, source, entry.filename)
