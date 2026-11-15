"""Row-wise RMS normalization: the Triton source of the ``rms_norm`` operator.

One program normalizes one contiguous last-axis row with FP32 statistics and
rounds once when storing to the output dtype. The descriptor bakes the row tile
(the program's block) and whether the launch is programmatic-dependent; the row
count comes from the grid and `columns` from the launch, so one module serves any
row count whose columns fit the tile.

The kernel body imports nothing but Triton: the launch-time wrapper, its operand
checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch


@triton.jit(do_not_specialize=["columns"])
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    columns,
    eps,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
    CENTRED: tl.constexpr,  # noqa: N803
    GROUPS: tl.constexpr,  # noqa: N803
):
    """Normalize one row per program.

    ``x``, ``w`` and ``y`` are the launch's packed operands (row tile, weight
    row, output row); ``columns`` is the row width and ``eps`` the variance
    floor. Every program loads its whole row before storing, so the output
    may alias the input.
    ``CENTRED`` reads a zero-centred weight: the row is scaled by ``1 + w``,
    the one added in FP32. ``GROUPS`` is the weight's row count: row ``i``
    reads weight row ``i mod GROUPS``.
    """
    dependent_launch.prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < columns
    row_base = row * columns
    x = tl.load(x_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + (row % GROUPS) * columns + offsets, mask=mask, other=0.0).to(tl.float32)
    if CENTRED:
        w = w + 1.0

    variance = tl.sum(x * x, axis=0) / columns
    y = x * tl.rsqrt(variance + eps) * w
    tl.store(y_ptr + row_base + offsets, y, mask=mask)
