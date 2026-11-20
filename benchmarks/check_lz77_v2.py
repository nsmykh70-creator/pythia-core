#!/usr/bin/env python3
"""Independent parser/round-trip checks for the PYTHIA PYLZ v1/v2 token wire format."""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

WINDOW = 65535

def put_offset(value: int) -> bytes:
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)

def get_offset(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for index, shift in enumerate((0, 7, 14)):
        if pos >= len(data):
            raise ValueError("truncated offset")
        b = data[pos]
        pos += 1
        value |= (b & 127) << shift
        if not b & 128:
            if (index and value < (1 << (7 * index))) or not 0 < value <= WINDOW:
                raise ValueError("invalid/noncanonical offset")
            return value, pos
    raise ValueError("overlong offset")

def decode(data: bytes) -> bytes:
    if len(data) < 9 or data[:4] != b"PYLZ" or data[4] not in (1, 2):
        raise ValueError("bad header")
    version = data[4]
    expected = int.from_bytes(data[5:9], "little")
    out = bytearray()
    pos = 9
    while len(out) < expected:
        if pos >= len(data):
            raise ValueError("truncated token")
        tag = data[pos]
        pos += 1
        if tag < 128:
            length = tag + 1
            if pos + length > len(data) or len(out) + length > expected:
                raise ValueError("bad literal")
            out.extend(data[pos:pos + length])
            pos += length
        else:
            length = (tag & 127) + 3
            if length < 4:
                raise ValueError("bad match length")
            if version == 1:
                if pos + 2 > len(data):
                    raise ValueError("truncated v1 offset")
                offset = int.from_bytes(data[pos:pos + 2], "little")
                pos += 2
            else:
                offset, pos = get_offset(data, pos)
            if not 0 < offset <= len(out) or len(out) + length > expected:
                raise ValueError("invalid match")
            for _ in range(length):
                out.append(out[-offset])
    if pos != len(data):
        raise ValueError("trailing bytes")
    return bytes(out)

def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_lz77_v2.py INPUT TOKEN_STREAM")
    source = Path(sys.argv[1]).read_bytes()
    token = Path(sys.argv[2]).read_bytes()
    restored = decode(token)
    assert restored == source
    assert token[4] == 2
    # Legacy fixed-width-offset frame: 'abcd' + match(length=4, offset=4).
    legacy = b"PYLZ\x01\x08\x00\x00\x00\x03abcd\x81\x04\x00"
    assert decode(legacy) == b"abcdabcd"
    # All ULEB128 boundary values have canonical encodings.
    for value in (1, 127, 128, 16383, 16384, 65535):
        encoded = put_offset(value)
        assert get_offset(encoded, 0) == (value, len(encoded))
    try:
        get_offset(b"\x84\x00", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("noncanonical offset was accepted")
    print(f"wire_version={token[4]} token_bytes={len(token)} source_bytes={len(source)}")
    print(f"source_sha256={hashlib.sha256(source).hexdigest()}")
    print(f"restored_sha256={hashlib.sha256(restored).hexdigest()} roundtrip=OK")
    print("legacy_v1=OK offset_boundaries=OK noncanonical_rejected=OK")

if __name__ == "__main__":
    main()
