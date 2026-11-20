#!/usr/bin/env python3
"""Independent reference and diagnostic tests for PYTHIA PS8.8 predictor."""
import random
import zlib

MASK = (1 << 64) - 1

def predictions(p1, p2, p3):
    return (0, p1, (2*p1-p2) & 255, (3*p1-3*p2+p3) & 255, p1 ^ p2)

def distance(a, b):
    d = (a-b) & 255
    return min(d, (-d) & 255)

def forward(data):
    losses = [0] * 5
    p1 = p2 = p3 = 0
    out = bytearray()
    for value in data:
        ps = predictions(p1, p2, p3)
        expert = min(range(5), key=lambda i: losses[i])
        out.append((value-ps[expert]) & 255)
        for i, pred in enumerate(ps):
            losses[i] = min(MASK, losses[i] + distance(value, pred))
        p3, p2, p1 = p2, p1, value
    return bytes(out)

def inverse(data):
    losses = [0] * 5
    p1 = p2 = p3 = 0
    out = bytearray()
    for residual in data:
        ps = predictions(p1, p2, p3)
        expert = min(range(5), key=lambda i: losses[i])
        value = (residual+ps[expert]) & 255
        out.append(value)
        for i, pred in enumerate(ps):
            losses[i] = min(MASK, losses[i] + distance(value, pred))
        p3, p2, p1 = p2, p1, value
    return bytes(out)

def make_cases():
    cases = [b"", b"\0", b"\xff", bytes(range(256))]
    for n in (2, 3, 4, 7, 8, 9, 255, 256, 257, 4096, 65537):
        cases.append(bytes((i*31+i//7+91) & 255 for i in range(n)))
        cases.append(bytes((i*i+13*i+5) & 255 for i in range(n)))
    cases.append(b"PYTHIA adaptive expert predictor " * 4096)
    rng = random.Random(42)
    cases.append(bytes(rng.randrange(256) for _ in range(1 << 20)))
    return cases

def main():
    cases = make_cases()
    for index, original in enumerate(cases):
        assert inverse(forward(original)) == original, (index, len(original))
    print(f"roundtrip: PASS ({len(cases)} cases)")

    rng = random.Random(12345)
    datasets = {
        "linear-1MiB": bytes((i*17) & 255 for i in range(1 << 20)),
        "quadratic-1MiB": bytes((i*i+3*i+7) & 255 for i in range(1 << 20)),
        "periodic-text-1MiB": (b"PYTHIA adaptive predictor; reversible compression. " * 22000)[:1 << 20],
        "random-1MiB": bytes(rng.randrange(256) for _ in range(1 << 20)),
    }
    print("dataset, original_zlib, residual_zlib, residual_zero_bytes")
    for name, original in datasets.items():
        residual = forward(original)
        assert inverse(residual) == original
        print(f"{name}, {len(zlib.compress(original, 6))}, {len(zlib.compress(residual, 6))}, {residual.count(0)}")

if __name__ == "__main__":
    main()
