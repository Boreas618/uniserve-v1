"""Row-wise RMS normalization: the *rebind* fixture of the ``rms_norm`` operator.

This module is a second implementation of the serving program's descriptor: the
same row, the same weight, the same epsilon, one kernel per row and the same
grid, but the mean square is scaled by the reciprocal of the row width instead
of divided by it. The result therefore differs from the serving program's in
its last rounding step and stays inside the operator's tolerance, which is what
makes installing it a pure rebind: the graph's node is re-pointed at a new
module and nothing about the plan, the grid or the descriptor changes.

The body below is the serving program's with that one change.

One program normalizes one contiguous last-axis row with FP32 statistics and
rounds once when storing to the output dtype. The descriptor bakes the row tile
(the program's block) and whether the launch is programmatic-dependent; the row
count comes from the grid and `columns` from the launch, so one module serves any
row count whose columns fit the tile.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/rms.py``), unchanged except that it
imports nothing but Triton: the launch-time wrapper, its operand checks and its
torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["columns"])
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    columns,
    eps,
    pdl,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Normalize one row per program.

    ``x``, ``w`` and ``y`` are the launch's packed operands (row tile, weight
    row, output row); ``columns`` is the row width, ``eps`` the variance floor
    and ``pdl`` the launch's copy of the descriptor's flag. Every program loads
    its whole row before storing, so the output may alias the input.
    """
    _pdl_prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < columns
    row_base = row * columns
    x = tl.load(x_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # One rounding step away from the serving program: multiply by the
    # reciprocal instead of dividing. Inside the operator's tolerance.
    variance = tl.sum(x * x, axis=0) * (1.0 / columns)
    y = x * tl.rsqrt(variance + eps) * w
    tl.store(y_ptr + row_base + offsets, y, mask=mask)
