#!/usr/bin/env python3
"""The digest the executor prints, computed over a slice's `.npy` dump.

A traced slice writes `float32` files; the executor's dump digests the `bfloat16` bytes a kernel
wrote. Both sides digest FNV-1a over the same rows of one hidden-state tensor, and that is what makes
the numbers comparable at all: this converts a dump to the bytes the device holds, then digests them
exactly as `platform::fnv1a` does. The conversion is exact rather than approximate because a
`float32` read back from a `bfloat16` device buffer has zero low bits, so dropping them restores the
device's byte.

    python3 tools/diagnosis/npy_digest.py --rows 24 hidden_00.npy hidden_00_attended.npy

Prints one line per file: the digest, the rows kept, the width, and the byte count, in the same shape
the executor's `p<k>=<digest>@<bytes>B` entries use.
"""

import argparse
import pathlib
import sys

import numpy as np

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a(payload: bytes) -> int:
    """The 64-bit FNV-1a `platform::fnv1a` computes, one byte at a time."""
    digest = FNV_OFFSET
    for byte in payload:
        digest ^= byte
        digest = (digest * FNV_PRIME) & MASK
    return digest


def as_device_bytes(values: np.ndarray) -> bytes:
    """The `bfloat16` bytes a device buffer of these `float32` values holds."""
    widened = np.ascontiguousarray(values.astype(np.float32))
    bits = widened.view(np.uint32) >> 16
    return bits.astype("<u2").tobytes()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--rows", type=int, default=24, help="leading rows to digest (default 24)")
    arguments = parser.parse_args(argv)
    for path in arguments.paths:
        values = np.load(path)
        flat = values.reshape(-1, values.shape[-1]) if values.ndim > 1 else values.reshape(-1, 1)
        kept = flat[: arguments.rows]
        if kept.shape[0] != arguments.rows:
            print(f"{path}: only {kept.shape[0]} row(s), asked for {arguments.rows}", file=sys.stderr)
            return 1
        payload = as_device_bytes(kept)
        print(f"{path.name} digest={fnv1a(payload):016x} rows={kept.shape[0]} "
              f"width={kept.shape[1]} bytes={len(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
