"""Bounded-memory access to the uncompressed NPZ files used for OpenWorm splits."""

import struct
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np


class CompressedArray:
    """Bounded-memory compatibility reader for older compressed NPZ datasets.

    Random access requires decompressing the prefix of a ZIP member; the large
    OpenWorm archives use the faster memory-mapped path instead.
    """

    def __init__(self, path, name):
        self.path, self.name = str(path), f"{name}.npy"
        self._archive = self._source = None
        with ZipFile(path) as archive, archive.open(self.name) as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                self.shape, order, self.dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                self.shape, order, self.dtype = np.lib.format.read_array_header_2_0(member)
            else:
                raise ValueError(f"Unsupported NPY version: {version}")
            self.offset = member.tell()
        self.ndim = len(self.shape)
        if order or self.dtype.kind not in "biuf" or self.dtype.hasobject or not self.shape:
            raise ValueError("Expected a real numeric C-order array")
        self.record_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._source is None:
            self._archive = ZipFile(self.path)
            self._source = self._archive.open(self.name)
        self._source.seek(self.offset + int(index) * self.record_bytes)
        content = self._source.read(self.record_bytes)
        if len(content) != self.record_bytes:
            raise ValueError("Truncated feature array")
        return np.frombuffer(content, dtype=self.dtype).reshape(self.shape[1:])

    def __del__(self):
        if self._source is not None:
            self._source.close()
        if self._archive is not None:
            self._archive.close()


def feature_array(path, name):
    """Map uncompressed features; stream legacy compressed features per record."""
    with ZipFile(path) as archive:
        stored = archive.getinfo(f"{name}.npy").compress_type == ZIP_STORED
    return map_array(path, name) if stored else CompressedArray(path, name)


def map_array(path, name):
    """Map a numeric C-order NPY member directly, without loading its NPZ array."""
    path = Path(path)
    with ZipFile(path) as archive:
        info = archive.getinfo(f"{name}.npy")
        if info.compress_type != ZIP_STORED or info.flag_bits & 1:
            raise ValueError(f"{path}: {name} must be uncompressed and unencrypted")
        with archive.open(info) as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, order, dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                shape, order, dtype = np.lib.format.read_array_header_2_0(member)
            else:
                raise ValueError(f"Unsupported NPY version: {version}")
            header_size = member.tell()
        if order or dtype.hasobject or dtype.kind not in "biuf":
            raise ValueError("Expected a real numeric C-order array without objects")
        if header_size + int(np.prod(shape)) * dtype.itemsize != info.file_size:
            raise ValueError(f"Invalid array size: {path}/{name}")
    with path.open("rb") as source:
        source.seek(info.header_offset)
        header = struct.unpack("<4s5H3I2H", source.read(30))
        if header[0] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
    offset = info.header_offset + 30 + header[-2] + header[-1] + header_size
    return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)
