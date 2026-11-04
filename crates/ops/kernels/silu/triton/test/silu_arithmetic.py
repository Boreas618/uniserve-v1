#!/usr/bin/env python3
"""Execute the silu kernel on the CPU and check it against the operator's formula.

Triton's interpreter runs the kernel body with numpy-backed tensors, one program at a time, so
the activation can be checked here — the arithmetic is one scalar formula per element, not a
tensor-core product:

    out = x / (1 + exp(−x))

The reference is computed in FP32 with torch, and the comparison is the operator's: one rounding
from FP32 into the output's type plus an absolute part covering the exponential's own error.
The check uses FP32 buffers so the comparison isolates the formula; a narrowed output's rounding
is the store, which the operator's tolerance states and the caller's GPU run measures.

    TRITON_INTERPRET=1 python silu_arithmetic.py
"""

import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import silu as kernels  # noqa: E402

#: The operator's value envelope and its absolute tolerance, from `crates/ops/src/activation.rs`.
VALUE_ENVELOPE = 8.0
ATOL = 4.0 * VALUE_ENVELOPE * torch.finfo(torch.float32).eps


def run(rows, columns, block):
    """Activate one `[rows, columns]` buffer over its whole grid."""
    x = (torch.rand(rows, columns, dtype=torch.float32) * 2.0 - 1.0) * VALUE_ENVELOPE
    out = torch.full_like(x, float("nan"))
    grid = ((x.numel() + block - 1) // block, 1, 1)
    kernels._silu_kernel[grid](x, out, x.numel(), columns, 0, BLOCK=block)
    expected = x / (1.0 + torch.exp(-x))
    error = (out - expected).abs().max().item()
    tolerance = torch.finfo(torch.float32).eps * expected.abs().max().item() + ATOL
    print(f"rows={rows} columns={columns} block={block}: max |error| = {error:.3e}")
    return error <= tolerance


def main() -> int:
    cases = [(1, 64, 64), (3, 100, 128), (4, 5376, 1024)]
    return 0 if all(run(*case) for case in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
