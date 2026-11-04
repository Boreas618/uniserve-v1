#!/usr/bin/env python3
"""Check the adaptive_pool kernel's arithmetic on this host, with no GPU.

    TRITON_INTERPRET=1 python interp_check.py

Triton's interpreter runs the kernel's own Python on the CPU, so this checks
the arithmetic and the loop bounds of exactly the program the store's module is
compiled from — not a transcription of it. It is the CPU half of the check: the
GPU script beside this one (`pool_check.py`) runs the compiled program on a
device, and this one can run anywhere, which is what lets a geometry be tried
before a device is booked.

The geometries below are chosen for what they can break rather than for what
the checkpoint uses:

* the checkpoint's own (8 heads of 256 pooled to 32) and a width the output
divides evenly (256 to 64), where every window is the same length; * the
degenerate pool (8 pooled to 16), where every window is one position and
positions repeat; * **widths the output does not divide** (24 to 7, 17 to 5),
where `ceil((i+1)·width/output) - floor(i·width/output)` reaches one position
past `ceil(width/output)` — the case that makes the loop bound
`ceil(width/output) + 1` and that a width which divides never shows.

The reference is the model's two steps in FP64: the mean over heads, then
`torch.nn.functional.adaptive_avg_pool1d` along the width.
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

import adaptive_pool as kernels
import numpy as np
import torch
import torch.nn.functional as functional

#: `DType::unit_roundoff` for FP32, the accumulate type.
FP32_ROUNDOFF = 5.960464477539063e-08

# : (heads, width, output, rows, block): the shapes above, in the order they
# break : things.
GEOMETRIES = [
    (8, 256, 32, 2, 64),
    (8, 256, 64, 3, 64),
    (8, 8, 16, 2, 64),
    (16, 24, 7, 2, 16),
    (8, 24, 7, 1, 64),
    (4, 17, 5, 3, 8),
]


def window_of(width, output, position):
    """The window of one output position: `floor` to `ceil`, never empty."""
    start = (position * width) // output
    end = -((-(position + 1) * width) // output)
    return start, max(end, start + 1)


def bound(heads, width, output):
    """The operator's relative bound, with the longest window the pool can give."""
    longest = max(
        window_of(width, output, position)[1]
        - window_of(width, output, position)[0]
        for position in range(output)
    )
    terms = heads * longest
    gamma = (terms * FP32_ROUNDOFF) / (1.0 - terms * FP32_ROUNDOFF)
    return 2.0 * gamma + FP32_ROUNDOFF


def main() -> int:
    failures = 0
    for heads, width, output, rows, block in GEOMETRIES:
        generator = np.random.default_rng(heads * 1000 + width)
        x = torch.from_numpy(
            generator.normal(0.0, 1.0, (rows, heads, width)).astype(np.float32)
        )
        out = torch.zeros((rows, output), dtype=torch.float32)
        longest = -(-width // output) + 1
        kernels._adaptive_pool_kernel[((rows * output + block - 1) // block,)](
            x,
            out,
            rows,
            heads,
            width,
            output,
            0,
            HEADS=heads,
            WIDTH=width,
            OUTPUT=output,
            WINDOW=longest,
            BLOCK=block,
            PDL=False,
        )
        expected = functional.adaptive_avg_pool1d(
            x.double().mean(dim=1), output
        )
        difference = (out.double() - expected).abs().max().item()
        largest = expected.abs().max().item()
        allowed = bound(heads, width, output) * largest + 64.0 * FP32_ROUNDOFF
        windows = sorted(
            {
                window_of(width, output, position)[1]
                - window_of(width, output, position)[0]
                for position in range(output)
            }
        )
        status = "ok" if difference <= allowed else "FAILED"
        print(
            f"{status}: {heads} heads of {width} to {output}, {rows} rows, "
            f"windows {windows}, max |out - exact| = {difference:.3e} "
            f"allowed {allowed:.3e}"
        )
        if difference > allowed:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
