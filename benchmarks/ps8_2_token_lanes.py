#!/usr/bin/env python3
"""Independent PS8.2 PYLZ v2 four/six-lane wire probe (stdlib only).

This checks the token grammar and exact lane reconstruction independently of
Rust. gzip/bzip2/xz figures are external lane-structure probes, not PYTHIA
archive results.
"""
from __future__ import annotations
import argparse
import bz2
import gzip
import hashlib
import lzma
import struct
from pathlib import Path


def parse(data: bytes):
    if len(data) < 9 or data[:4] != b"PYLZ" or data[4] not in (1, 2):
        raise ValueError("unsupported PYLZ header")
    version = data[4]
    raw_len = struct.unpack_from("<I", data, 5)[0]
    pos = 9
    produced = 0
    kinds, literal_lengths, match_lengths = bytearray(), bytearray(), bytearray()
    offset_classes, offset_residuals, literals = bytearray(), bytearray(), bytearray()

    def read_uleb():
        nonlocal pos
        value = 0
        for i, shift in enumerate((0, 7, 14)):
            if pos >= len(data):
                raise ValueError("truncated offset")
            b = data[pos]
            pos += 1
            value |= (b & 0x7F) << shift
            if not (b & 0x80):
                if value == 0 or value > 65535 or (i and value < (1 << (7 * i))):
                    raise ValueError("invalid/noncanonical offset")
                return value
        raise ValueError("overlong offset")

    while produced < raw_len:
        if pos >= len(data):
            raise ValueError("truncated token")
        tag = data[pos]
        pos += 1
        if tag < 128:
            count = tag + 1
            if pos + count > len(data):
                raise ValueError("truncated literal")
            kinds.append(0)
            literal_lengths.append(tag)
            literals.extend(data[pos:pos + count])
            pos += count
            produced += count
        else:
            match_len = (tag & 0x7F) + 3
            if match_len < 4:
                raise ValueError("invalid match length")
            if version == 1:
                if pos + 2 > len(data):
                    raise ValueError("truncated legacy offset")
                distance = struct.unpack_from("<H", data, pos)[0]
                pos += 2
            else:
                distance = read_uleb()
            if distance == 0 or distance > produced:
                raise ValueError("invalid back-reference")
            cls = distance.bit_length() - 1
            kinds.append(1)
            match_lengths.append(match_len - 3)
            offset_classes.append(cls)
            offset_residuals.extend(struct.pack("<H", distance - (1 << cls)))
            produced += match_len
        if produced > raw_len:
            raise ValueError("token stream exceeds declared output")
    if pos != len(data):
        raise ValueError("trailing bytes")
    return version, raw_len, [bytes(x) for x in (
        kinds, literal_lengths, match_lengths, offset_classes, offset_residuals, literals
    )]


def rebuild(version: int, raw_len: int, lanes: list[bytes]) -> bytes:
    kinds, literal_lengths, match_lengths, offset_classes, offset_residuals, literals = lanes
    out = bytearray(b"PYLZ" + bytes([version]) + struct.pack("<I", raw_len))
    il = im = ic = ir = ib = 0
    produced = 0
    for kind in kinds:
        if kind == 0:
            if il >= len(literal_lengths):
                raise ValueError("literal length lane exhausted")
            length = literal_lengths[il]
            il += 1
            count = length + 1
            if ib + count > len(literals):
                raise ValueError("literal lane exhausted")
            out.append(length)
            out.extend(literals[ib:ib + count])
            ib += count
            produced += count
        elif kind == 1:
            if im >= len(match_lengths) or ic >= len(offset_classes) or ir + 2 > len(offset_residuals):
                raise ValueError("match metadata lane exhausted")
            length = match_lengths[im]
            im += 1
            cls = offset_classes[ic]
            ic += 1
            residual = struct.unpack_from("<H", offset_residuals, ir)[0]
            ir += 2
            if cls > 15 or residual >= (1 << cls):
                raise ValueError("noncanonical offset class/residual")
            distance = (1 << cls) + residual
            if distance > produced:
                raise ValueError("back-reference exceeds produced bytes")
            out.append(0x80 | length)
            if version == 1:
                out.extend(struct.pack("<H", distance))
            else:
                while distance >= 0x80:
                    out.append((distance & 0x7F) | 0x80)
                    distance >>= 7
                out.append(distance)
            produced += length + 3
        else:
            raise ValueError("unknown token kind")
    if (il, im, ic, ir, ib) != (len(literal_lengths), len(match_lengths), len(offset_classes), len(offset_residuals), len(literals)):
        raise ValueError("unused lane bytes")
    if produced != raw_len:
        raise ValueError("reconstructed length mismatch")
    return bytes(out)


def decode_raw(version: int, raw_len: int, lanes: list[bytes]) -> bytes:
    kinds, literal_lengths, match_lengths, offset_classes, offset_residuals, literals = lanes
    out = bytearray()
    il = im = ic = ir = ib = 0
    for kind in kinds:
        if kind == 0:
            length = literal_lengths[il]; il += 1
            count = length + 1
            out.extend(literals[ib:ib + count]); ib += count
        else:
            length = match_lengths[im]; im += 1
            cls = offset_classes[ic]; ic += 1
            residual = struct.unpack_from("<H", offset_residuals, ir)[0]; ir += 2
            distance = (1 << cls) + residual
            for _ in range(length + 3):
                out.append(out[-distance])
    if len(out) != raw_len:
        raise ValueError("decoded raw length mismatch")
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="PYLZ v1/v2 token stream")
    ap.add_argument("--original", type=Path, help="optional original raw file for SHA-256 comparison")
    ap.add_argument("--skip-external", action="store_true", help="skip gzip/bzip2/xz lane probes")
    args = ap.parse_args()
    data = args.input.read_bytes()
    version, raw_len, lanes = parse(data)
    restored = rebuild(version, raw_len, lanes)
    if restored != data:
        raise SystemExit("FAIL: rebuilt stream differs")
    raw = decode_raw(version, raw_len, lanes)
    if args.original:
        original = args.original.read_bytes()
        if raw != original:
            raise SystemExit("FAIL: decoded original differs from supplied raw file")
        print(f"original_sha256={hashlib.sha256(raw).hexdigest()} original_match=YES")
    print(f"input_bytes={len(data)} raw_bytes={raw_len} version={version}")
    print(f"sha256={hashlib.sha256(data).hexdigest()}")
    print(f"tokens={len(lanes[0])} matches={lanes[0].count(1)} literal_runs={lanes[0].count(0)}")
    print("six_lane_bytes=" + ",".join(str(len(x)) for x in lanes))
    # Compact profile: retain token kinds, one merged length lane, u16 offsets,
    # and literal bytes. This competes with the six-lane context profile.
    kinds, literal_lengths, match_lengths, offset_classes, offset_residuals, literals = lanes
    lengths = bytearray()
    offsets = bytearray()
    il = im = ic = ir = 0
    for kind in kinds:
        if kind == 0:
            lengths.append(literal_lengths[il]); il += 1
        else:
            lengths.append(match_lengths[im]); im += 1
            cls = offset_classes[ic]; ic += 1
            residual = struct.unpack_from("<H", offset_residuals, ir)[0]; ir += 2
            offsets.extend(struct.pack("<H", (1 << cls) + residual))
    compact_lanes = [kinds, bytes(lengths), bytes(offsets), literals]
    print("four_lane_bytes=" + ",".join(str(len(x)) for x in compact_lanes))
    print("roundtrip=OK")
    if not args.skip_external:
        compressors = {
            "gzip9": lambda x: gzip.compress(x, compresslevel=9, mtime=0),
            "bzip2_9": lambda x: bz2.compress(x, compresslevel=9),
            "xz6": lambda x: lzma.compress(x, preset=6),
        }
        for name, compress in compressors.items():
            flat = len(compress(data))
            # Header constants already include the 4/6 u32 lane-size fields.
            six = sum(len(compress(x)) for x in lanes) + 39
            four = sum(len(compress(x)) for x in compact_lanes) + 31
            best_name, best = ("four", four) if four <= six else ("six", six)
            print(f"{name}_flat={flat} {name}_four_lane={four} {name}_six_lane={six} best={best_name}:{best} delta={best-flat}")


if __name__ == "__main__":
    main()
