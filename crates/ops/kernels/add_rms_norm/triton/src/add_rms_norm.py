"""Residual-add RMS normalization: the Triton source of the ``add_rms_norm``
operator.

One program adds one residual row, keeps the FP32 sum as the residual stream,
normalizes that sum and rounds once into each output. As in ``rms_norm`` the
descriptor bakes the row tile and the dependent-launch flag, the row count comes
from the grid and `columns` from the launch.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/rms.py``), unchanged except that it
imports nothing but Triton: the operand checks, output allocation and torch types
stay with the operator's Rust side.
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
def _add_rms_norm_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    c_ptr,
    columns,
    eps,
    pdl,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Add a residual, preserve the sum, and normalize it in one program.

    ``x``, ``r``, ``w``, ``y`` and ``c`` are the launch's packed operands;
    ``c`` receives the residual sum and ``y`` its normalized projection.
    """
    _pdl_prologue(PDL)

    # Rows are contiguous; int64 row bases keep offsets past 2**31 valid.
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < columns
    row_base = row * columns
    x = tl.load(x_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
    c = x + r

    # Both outputs derive from the same fp32 sum: ``c_ptr`` carries the residual
    # stream and ``y_ptr`` carries its normalized projection.
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(c * c, axis=0) / columns
    y = c * tl.rsqrt(variance + eps) * w
    tl.store(c_ptr + row_base + offsets, c, mask=mask)
    tl.store(y_ptr + row_base + offsets, y, mask=mask)
