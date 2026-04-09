import os
import struct
import unittest

import pytest

from torch_tensorrt.executorch._serialization import (
    HEADER_FORMAT,
    HEADER_SIZE,
    TENSORRT_MAGIC,
    TensorRTBlobMetadata,
    TensorRTIOBinding,
    deserialize_engine,
    serialize_engine,
)

assertions = unittest.TestCase()


def _make_metadata(**kwargs):
    defaults = dict(
        io_bindings=[
            TensorRTIOBinding("input0", "float32", [1, 3, 224, 224], True),
            TensorRTIOBinding("output0", "float32", [1, 1000], False),
        ],
        hardware_compatible=False,
        device_id=0,
    )
    defaults.update(kwargs)
    return TensorRTBlobMetadata(**defaults)


@pytest.mark.unit
@pytest.mark.critical
def test_roundtrip_small():
    engine = os.urandom(100)
    meta = _make_metadata()
    blob = serialize_engine(engine, meta)
    engine_out, meta_out = deserialize_engine(blob)
    assertions.assertEqual(engine_out, engine)
    assertions.assertEqual(len(meta_out.io_bindings), 2)
    assertions.assertEqual(meta_out.io_bindings[0].name, "input0")
    assertions.assertEqual(meta_out.io_bindings[0].dtype, "float32")
    assertions.assertEqual(meta_out.io_bindings[0].shape, [1, 3, 224, 224])
    assertions.assertTrue(meta_out.io_bindings[0].is_input)
    assertions.assertEqual(meta_out.io_bindings[1].name, "output0")
    assertions.assertEqual(meta_out.io_bindings[1].dtype, "float32")
    assertions.assertEqual(meta_out.io_bindings[1].shape, [1, 1000])
    assertions.assertFalse(meta_out.io_bindings[1].is_input)


@pytest.mark.unit
def test_roundtrip_large():
    engine = os.urandom(10 * 1024 * 1024)
    meta = _make_metadata()
    blob = serialize_engine(engine, meta)
    engine_out, meta_out = deserialize_engine(blob)
    assertions.assertEqual(engine_out, engine)


@pytest.mark.unit
def test_empty_engine():
    meta = _make_metadata()
    blob = serialize_engine(b"", meta)
    assertions.assertTrue(len(blob) >= HEADER_SIZE)
    engine_out, meta_out = deserialize_engine(blob)
    assertions.assertEqual(engine_out, b"")


@pytest.mark.unit
def test_metadata_fields():
    meta = _make_metadata(
        hardware_compatible=True,
        device_id=3,
        serialized_metadata="test_meta",
        target_platform="x86_64",
    )
    blob = serialize_engine(b"eng", meta)
    _, meta_out = deserialize_engine(blob)
    assertions.assertTrue(meta_out.hardware_compatible)
    assertions.assertEqual(meta_out.device_id, 3)
    assertions.assertEqual(meta_out.serialized_metadata, "test_meta")
    assertions.assertEqual(meta_out.target_platform, "x86_64")


@pytest.mark.unit
def test_wrong_magic_raises():
    engine = b"test_engine"
    meta = _make_metadata()
    blob = serialize_engine(engine, meta)
    bad_blob = b"TRT2" + blob[4:]
    with pytest.raises(ValueError, match="Invalid magic"):
        deserialize_engine(bad_blob)


@pytest.mark.unit
def test_truncated_blob_raises():
    with pytest.raises((ValueError, struct.error)):
        deserialize_engine(b"TR01" + b"\x00" * 10)


@pytest.mark.unit
def test_engine_size_mismatch_raises():
    engine = b"test_engine"
    meta = _make_metadata()
    blob = serialize_engine(engine, meta)
    truncated = blob[: len(blob) - 5]
    with pytest.raises(ValueError, match="Engine extends past blob"):
        deserialize_engine(truncated)


@pytest.mark.unit
def test_engine_16_byte_aligned():
    meta = _make_metadata()
    for size in [0, 1, 15, 16, 17, 100, 255]:
        engine = os.urandom(size)
        blob = serialize_engine(engine, meta)
        _, _, _, eng_off, _, _ = struct.unpack(HEADER_FORMAT, blob[:HEADER_SIZE])
        assertions.assertEqual(
            eng_off % 16, 0, f"Engine offset {eng_off} not 16-byte aligned for size {size}"
        )


@pytest.mark.unit
def test_unaligned_engine_offset_raises():
    header = struct.pack(
        HEADER_FORMAT, TENSORRT_MAGIC, HEADER_SIZE, 0, 33, 0, b"\x00" * 8
    )
    with pytest.raises(ValueError, match="not 16-byte aligned"):
        deserialize_engine(header)


@pytest.mark.unit
def test_metadata_offset_inside_header_raises():
    header = struct.pack(
        HEADER_FORMAT, TENSORRT_MAGIC, 10, 0, 32, 0, b"\x00" * 8
    )
    with pytest.raises(ValueError, match="inside the header"):
        deserialize_engine(header)


@pytest.mark.unit
def test_metadata_extends_past_blob_raises():
    header = struct.pack(
        HEADER_FORMAT, TENSORRT_MAGIC, HEADER_SIZE, 100, 48, 0, b"\x00" * 8
    )
    with pytest.raises(ValueError, match="Metadata extends past blob"):
        deserialize_engine(header)


@pytest.mark.unit
def test_metadata_overlaps_engine_raises():
    header = struct.pack(
        HEADER_FORMAT, TENSORRT_MAGIC, HEADER_SIZE, 20, 48, 0, b"\x00" * 8
    )
    blob = header + b"\x00" * 20
    with pytest.raises(ValueError, match="Metadata region overlaps engine"):
        deserialize_engine(blob)
