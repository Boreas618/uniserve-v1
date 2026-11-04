#!/usr/bin/env python3
"""Execute the patch kernel's index arithmetic on the CPU and check both directions.

The kernel is a permutation, so it can be run without a GPU: Triton's interpreter executes the
kernel body with numpy-backed tensors, one program at a time, and the result is compared against
the operator's own definition from ``crates/ops/src/patch.rs``, written out in torch:

    patchify    x [channels, frames, height, width] -> r [rows, channels · 4]
                row = ((f · height/2) + i) · (width/2) + j,  column = ((c · 2) + a) · 2 + b
                r[row, column] = x[c, f, 2·i + a, 2·j + b]

Run it where Triton is installed (the Program Server's own environment):

    TRITON_INTERPRET=1 python patch_arithmetic.py [--channels 24] [--frames 3]

The interpreter is slow — one program at a time — so the check covers a few geometries rather
than a production one. It is the arithmetic that is under test: the grid, the two decodings and
the two directions, which is what a GPU run could only confirm again.
"""

import argparse
import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import patch as kernels  # noqa: E402


def launch(x, out, frames, rows, channels, height, width, block, patchify):
    """Run the kernel over its whole grid, one interpreter program at a time."""
    grid = (rows, (channels * 4 + block - 1) // block, 1)
    kernels._patch_kernel[grid](
        x,
        out,
        frames,
        rows,
        channels,
        height,
        width,
        block,
        0,
        PATCHIFY=patchify,
        BLOCK=block,
    )


def reference(latent, rows, patchify, channels, frames, height, width):
    """The operator's mapping, in torch, from the formulas in `crates/ops/src/patch.rs`."""
    stacked = torch.zeros(rows, channels * 4, dtype=latent.dtype)
    for row in range(rows):
        per_frame = (height // 2) * (width // 2)
        frame, within = divmod(row, per_frame)
        patch_row, patch_column = divmod(within, width // 2)
        for column in range(channels * 4):
            channel, inner = divmod(column, 4)
            inner_row, inner_column = divmod(inner, 2)
            stacked[row, column] = latent[
                channel, frame, 2 * patch_row + inner_row, 2 * patch_column + inner_column
            ]
    return stacked


def one(channels, frames, height, width, block):
    """Check both directions at one geometry."""
    rows = frames * (height // 2) * (width // 2)
    latent = torch.arange(
        channels * frames * height * width, dtype=torch.float32
    ).reshape(channels, frames, height, width)
    packed = reference(latent, rows, True, channels, frames, height, width)

    packed_out = torch.full((rows, channels * 4), -1.0)
    launch(latent, packed_out, frames, rows, channels, height, width, block, True)
    if not torch.equal(packed_out, packed):
        print(f"patchify differs at {channels}×{frames}×{height}×{width}", file=sys.stderr)
        return False

    latent_out = torch.full_like(latent, -1.0)
    launch(packed, latent_out, frames, rows, channels, height, width, block, False)
    if not torch.equal(latent_out, latent):
        print(f"unpatchify differs at {channels}×{frames}×{height}×{width}", file=sys.stderr)
        return False
    print(f"ok channels={channels} frames={frames} {height}×{width} rows={rows} block={block}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=3, help="small, so the run stays quick")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--block", type=int, default=4, help="narrower than a row, so the column slices run")
    arguments = parser.parse_args()
    if not torch.cuda.is_available() and os.environ["TRITON_INTERPRET"] != "1":
        print("run this with TRITON_INTERPRET=1 on a host without a GPU", file=sys.stderr)
        return 1
    geometries = [(args_frames, height, width) for args_frames in (1, arguments.frames)
                  for height, width in ((2, 2), (4, 6))]
    return 0 if all(
        one(arguments.channels, frames, height, width, arguments.block)
        for frames, height, width in geometries
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
