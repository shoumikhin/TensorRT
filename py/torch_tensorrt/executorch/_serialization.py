"""TR01 wire-format serialisation for TensorRT engine blobs.

Layout (all multi-byte integers are little-endian):

  Offset  Size  Field
  ------  ----  -----
   0       4    Magic bytes: b"TR01"
   4       4    Metadata offset (bytes from start of blob)
   8       4    Metadata size (bytes)
  12       4    Engine offset (bytes from start of blob, 16-byte aligned)
  16       8    Engine size (bytes)
  24       8    Reserved (schema-version tag + padding)

The metadata region contains a UTF-8 JSON object.  The engine region
follows, padded so that its start offset is 16-byte aligned.
"""

import dataclasses
import json
import struct
from dataclasses import dataclass, field
from typing import List, Tuple

TENSORRT_MAGIC = b"TR01"
HEADER_FORMAT = "<4sIIIQ8s"  # little-endian
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def _align_to_16(offset: int) -> int:
    """Round *offset* up to the nearest multiple of 16."""
    return (offset + 15) & ~15


@dataclass
class TensorRTIOBinding:
    name: str
    dtype: str
    shape: List[int]
    is_input: bool


@dataclass
class TensorRTBlobMetadata:
    io_bindings: List[TensorRTIOBinding] = field(default_factory=list)
    hardware_compatible: bool = False
    device_id: int = 0
    serialized_metadata: str = ""
    target_platform: str = ""

    def to_json(self) -> bytes:
        # Field ordering matters: TensorRTBlobHeader.h parses io_bindings first,
        # then searches forward for hardware_compatible and device_id.
        data = {
            "io_bindings": [
                {
                    "name": b.name,
                    "dtype": b.dtype,
                    "shape": b.shape,
                    "is_input": b.is_input,
                }
                for b in self.io_bindings
            ],
            "hardware_compatible": self.hardware_compatible,
            "device_id": self.device_id,
            "serialized_metadata": self.serialized_metadata,
            "target_platform": self.target_platform,
        }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "TensorRTBlobMetadata":
        parsed = json.loads(data.decode("utf-8"))
        _io_binding_fields = {f.name for f in dataclasses.fields(TensorRTIOBinding)}
        io_bindings = [
            TensorRTIOBinding(**{k: v for k, v in b.items() if k in _io_binding_fields})
            for b in parsed.get("io_bindings", [])
        ]
        return cls(
            io_bindings=io_bindings,
            hardware_compatible=parsed.get("hardware_compatible", False),
            device_id=parsed.get("device_id", 0),
            serialized_metadata=parsed.get("serialized_metadata", ""),
            target_platform=parsed.get("target_platform", ""),
        )


def serialize_engine(
    engine_bytes: bytes,
    metadata: TensorRTBlobMetadata,
) -> bytes:
    metadata_json = metadata.to_json()
    metadata_offset = HEADER_SIZE
    engine_offset = _align_to_16(metadata_offset + len(metadata_json))

    reserved = b"\x01" + b"\x00" * 7  # schema version 1
    header = struct.pack(
        HEADER_FORMAT,
        TENSORRT_MAGIC,
        metadata_offset,
        len(metadata_json),
        engine_offset,
        len(engine_bytes),
        reserved,
    )
    padding = b"\x00" * (engine_offset - metadata_offset - len(metadata_json))
    return header + metadata_json + padding + engine_bytes


def deserialize_engine(blob: bytes) -> Tuple[bytes, TensorRTBlobMetadata]:
    if len(blob) < HEADER_SIZE:
        raise ValueError(f"Blob too small: {len(blob)} bytes")
    magic, meta_off, meta_size, eng_off, eng_size, _ = struct.unpack(
        HEADER_FORMAT, blob[:HEADER_SIZE]
    )
    if magic != TENSORRT_MAGIC:
        raise ValueError(f"Invalid magic: {magic}")
    if eng_off % 16 != 0:
        raise ValueError(
            f"Engine offset not 16-byte aligned: {eng_off}"
        )
    if meta_off < HEADER_SIZE:
        raise ValueError(
            f"Metadata offset {meta_off} is inside the header (size={HEADER_SIZE})"
        )
    if meta_off + meta_size > len(blob):
        raise ValueError(
            f"Metadata extends past blob: offset={meta_off}, size={meta_size}, blob={len(blob)}"
        )
    if eng_off + eng_size > len(blob):
        raise ValueError(
            f"Engine extends past blob: offset={eng_off}, size={eng_size}, blob={len(blob)}"
        )
    if meta_off + meta_size > eng_off:
        raise ValueError(
            f"Metadata region overlaps engine: meta=[{meta_off}, {meta_off + meta_size}), engine=[{eng_off}, {eng_off + eng_size})"
        )
    metadata = TensorRTBlobMetadata.from_json(blob[meta_off : meta_off + meta_size])
    engine_bytes = blob[eng_off : eng_off + eng_size]
    return engine_bytes, metadata
