#!/usr/bin/env python3
"""Execute the scaled_add kernel on the CPU and check the operator's two identities.

Triton's interpreter runs the kernel body with numpy-backed tensors, one program at a time. The
step is scalar arithmetic per element, so it can be checked here:

    target ← β · target + α · source

Two cases carry the operator's contract: with β = 0 and α = 1 it is a copy (bit-exact when the
element types agree), and with β = 1 and α = 0 it is the identity. A general pair of coefficients
is compared in FP32 with the operator's tolerance, which is `2u(target) + u(source)` relative —
the tolerance a narrowing store costs.

    TRITON_INTERPRET=1 python scaled_add_arithmetic.py
"""

import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import scaled_add as kernels  # noqa: E402


def run(rows, columns, block, alpha, beta, exact):
    """Apply one step over the whole grid and compare."""
    target = torch.rand(rows, columns, dtype=torch.float32)
    source = torch.rand(rows, columns, dtype=torch.float32)
    original = target.clone()
    expected = beta * original + alpha * source
    grid = ((target.numel() + block - 1) // block, 1, 1)
    kernels._scaled_add_kernel[grid](target, source, alpha, beta, target.numel(), BLOCK=block)
    if exact:
        ok = torch.equal(target, expected)
        detail = "bit-exact" if ok else "differs"
    else:
        error = (target - expected).abs().max().item()
        tolerance = torch.finfo(torch.float32).eps * 3.0 * expected.abs().max().item()
        ok = error <= tolerance
        detail = f"max |error| = {error:.3e}"
    print(f"rows={rows} columns={columns} block={block} α={alpha} β={beta}: {detail}")
    return ok


def main() -> int:
    cases = [
        # β = 0, α = 1 is a copy; β = 1, α = 0 is the identity: both exact in FP32.
        (2, 64, 64, 1.0, 0.0, True),
        (2, 64, 64, 0.0, 1.0, True),
        (3, 100, 128, 0.25, 0.75, False),
        (4, 5376, 1024, 0.5, 1.0, False),
    ]
    return 0 if all(run(*case) for case in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
