#!/usr/bin/env python3
"""Execute the sequence_append kernel on the CPU and check it against the operator's contract.

Triton's interpreter runs the kernel body with numpy-backed buffers, one program at a time. The
operator writes the last `registers + tail` rows of a buffer the caller has already filled:

    out[rows + r, :]              = registers_weight[r, :]     for r < registers
    out[rows + registers + t, :]  = 0                          for t < tail

Nothing else may change: the caller's own rows and anything past the tail are other producers'
bytes, so the check fills them with a sentinel and asserts every one of them is untouched.

    TRITON_INTERPRET=1 python append_arithmetic.py
"""

import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import sequence_append as kernels  # noqa: E402

#: What the caller's rows and the spare rows hold: a value the operator must never write.
SENTINEL = -12345.0


def one(width, registers, tail, rows, block):
    """Check one geometry, including the rows this launch must leave alone."""
    total_rows = rows + registers + tail + 2
    out = torch.full((total_rows, width), SENTINEL, dtype=torch.float32)
    weights = torch.rand(registers, width, dtype=torch.float32)
    grid = (((registers + tail) * width + block - 1) // block, 1, 1)
    kernels._sequence_append_kernel[grid](
        out,
        weights,
        rows,
        width,
        registers,
        tail,
        block,
        0,
        BLOCK=block,
    )

    expected = torch.full_like(out, SENTINEL)
    expected[rows : rows + registers] = weights
    expected[rows + registers : rows + registers + tail] = 0.0
    ok = torch.equal(out, expected)
    print(
        f"width={width} registers={registers} tail={tail} rows={rows} block={block}: "
        f"{'exact, nothing else written' if ok else 'DIFFERS'}"
    )
    return ok


def main() -> int:
    cases = [(8, 4, 12, 3, 16), (5376, 4, 12, 0, 1024), (5, 1, 1, 2, 4), (8, 4, 12, 2, 1024)]
    return 0 if all(one(*case) for case in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
