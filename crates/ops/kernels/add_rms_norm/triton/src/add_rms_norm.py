"""Residual-add RMS normalization: the Triton source of the ``add_rms_norm``
operator.

One program adds one residual row, keeps the FP32 sum as the residual stream,
normalizes that sum and rounds once into each output. The kernel body is the
one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/rms.py``), unchanged except that
it imports nothing but Triton: the operand checks, output allocation and torch
types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _add_rms_norm_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    c_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Add a residual, preserve the sum, and normalize it in one program."""
    _pdl_prologue(PDL)

    # Rows are contiguous; int64 row bases keep offsets past 2**31 valid.
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, block)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    c = x + r

    # Both outputs derive from the same fp32 sum: ``c_ptr`` carries the
    # residual stream and ``y_ptr`` carries its normalized projection.
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(c * c, axis=0) / n_cols
    y = c * tl.rsqrt(var + eps) * w
    tl.store(c_ptr + row * n_cols + offs, c, mask=mask)
    tl.store(y_ptr + row * n_cols + offs, y, mask=mask)
