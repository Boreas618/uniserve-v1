#!/usr/bin/env python3
"""Run the transpose program on a GPU and check it against torch's own transpose.

    python transpose_check.py
    python transpose_check.py --dtype bf16 --rows 1000 --columns 3

The package's Triton source is launched through Triton's own launcher over device
tensors, one shape per case, and compared with `source.T.contiguous()` **bit for bit**:
a transpose performs no arithmetic, so the operator's tolerance is exact and any
difference is a wrong element rather than a rounding. Half precision is compared as its
raw bit patterns, so a value that is not a number can never compare equal to itself.

The shapes are the ones a tiled transpose breaks on: an exact grid of tiles, a partial
tile at the right edge, a partial tile at the bottom edge, both at once, a matrix
smaller than a single tile, and a sliver whose destination is one long row per source
column — the worst case for the coalescing the tiling exists for.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: Triton's element type for each element type the operator accepts.
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}

#: The shapes one run checks, as (rows, columns) of the source.
SHAPES = [
    (64, 64),  # an exact grid of tiles
    (64, 96),  # a partial tile in the destination's rows
    (96, 64),  # a partial tile in the destination's columns
    (70, 77),  # both edges partial
    (5, 7),  # smaller than one tile
    (1000, 3),  # a sliver: three columns become three rows of a thousand
]


def bits(tensor):
    """The tensor's raw bit patterns, so equality is bit equality."""
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    return tensor.view(torch.int16)


def check(rows, columns, block, dtype, pdl, seed):
    """Transpose one shape and compare it with torch, bit for bit."""
    import transpose as kernels

    torch.manual_seed(seed)
    source = torch.randn(rows, columns, device="cuda", dtype=torch.float32)
    source = source.to(dtype)
    out = torch.full((columns, rows), float("nan"), device="cuda", dtype=dtype)
    kernels._transpose_kernel[(-(-columns // block), -(-rows // block))](
        source,
        out,
        rows,
        columns,
        block,
        1 if pdl else 0,
        BLOCK=block,
        PDL=pdl,
    )
    torch.cuda.synchronize()

    expected = source.T.contiguous()
    differing = (bits(out) != bits(expected)).nonzero()
    if differing.numel() == 0:
        print(
            f"exact: {rows} x {columns} written as {columns} x {rows}, "
            f"{dtype}, tile {block}, pdl {str(pdl).lower()}"
        )
        return 0
    at = tuple(int(value) for value in differing[0])
    print(
        f"DIFFERS: {rows} x {columns} at {at}: got {out[at].item()}, "
        f"expected {expected[at].item()} ({differing.size(0)} of {out.numel()})",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="f32")
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--pdl", choices=["true", "false"], default="false")
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--columns", type=int, default=0)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        print("no GPU on this host: this check needs one", file=sys.stderr)
        return 1

    pdl = arguments.pdl == "true"
    shapes = SHAPES
    if arguments.rows or arguments.columns:
        shapes = [(arguments.rows, arguments.columns)]
    failures = 0
    for at, (rows, columns) in enumerate(shapes):
        failures += check(
            rows, columns, arguments.block, arguments.dtype, pdl, at + 1
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
