#!/usr/bin/env python3
"""Check the transpose program's tiles and edges on the CPU.

    TRITON_INTERPRET=1 python interp_check.py

Triton's interpreter runs the kernel's own Python, so this proves the tiling, the edge
masks and the address arithmetic on any host with no device. A transpose moves bits: no
arithmetic happens, so every element is compared exactly, and the shapes are chosen for
what can go wrong rather than for size — a matrix smaller than one tile, one wide and one
tall with a partial last tile in one axis only, and a square one whose grid is exact.
"""

import os
import pathlib
import sys

if os.environ.get("TRITON_INTERPRET") != "1":
    print(
        "this check runs the kernel through Triton's interpreter: "
        "TRITON_INTERPRET=1 python interp_check.py",
        file=sys.stderr,
    )
    raise SystemExit(2)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transpose as kernels  # noqa: E402

BLOCK = 32


def run(rows, columns, seed):
    """Transpose one matrix and compare it with numpy's, element for element."""
    rng = np.random.default_rng(seed)
    source = rng.integers(
        -(2**31), 2**31 - 1, size=(rows, columns), dtype=np.int64
    )
    source = source.astype(np.int32)
    out = np.full((columns, rows), -7, dtype=np.int32)
    kernels._transpose_kernel[(-(-columns // BLOCK), -(-rows // BLOCK))](
        torch.from_numpy(source),
        torch.from_numpy(out),
        rows,
        columns,
        BLOCK,
        0,
        BLOCK=BLOCK,
        PDL=False,
    )
    expected = np.ascontiguousarray(source.T)
    if np.array_equal(out, expected):
        print(f"exact: {rows} x {columns} written as {columns} x {rows}")
        return 0
    differing = np.argwhere(out != expected)[0]
    print(
        f"DIFFERS: {rows} x {columns} at {tuple(int(v) for v in differing)}: "
        f"got {int(out[tuple(differing)])}, expected "
        f"{int(expected[tuple(differing)])}",
        file=sys.stderr,
    )
    return 1


def main():
    scenarios = [
        (64, 64, 1),  # a grid of whole tiles, no edge
        (64, 96, 2),  # wider: a partial tile in the destination's rows
        (96, 64, 3),  # taller: a partial tile in the destination's columns
        (5, 7, 4),  # smaller than one tile on both sides
        (1, 1, 5),  # the degenerate tile
    ]
    failures = sum(run(*scenario) for scenario in scenarios)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
