"""Row-wise RMS normalization: the Triton source of the ``rms_norm`` operator.

One program normalizes one last-axis row with FP32 statistics and rounds once
when storing to the output dtype. The row's leading-axis geometry is a
compile-time tuple, so one module covers one (geometry, width, dtypes,
programmatic-dependent-launch) descriptor and no other.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/rms.py``), unchanged except that
it imports nothing but Triton: the launch-time wrapper, its operand checks and
its torch types stay with the operator's Rust side.
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
def _row_offset(row, shape: tl.constexpr, strides: tl.constexpr):
    """Return the element offset of flattened row ``row``.

    ``shape`` holds the extents of every leading axis after the first and
    ``strides`` the element strides of all leading axes, both as compile-time
    tuples; the first extent only bounds the launch.
    """
    offset = tl.full((), 0, tl.int64)
    remaining = row.to(tl.int64)
    for axis in tl.static_range(len(shape) - 1, -1, -1):
        offset += (remaining % shape[axis]) * strides[axis + 1]
        remaining //= shape[axis]
    return offset + remaining * strides[0]


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    x_shape: tl.constexpr,
    x_strides: tl.constexpr,
    y_shape: tl.constexpr,
    y_strides: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Normalize one hidden-state row per Triton program.

    Input and output rows are addressed through their own leading-axis strides
    with unit channel stride. Every program loads its whole row before storing,
    so the output may alias the input.
    """
    _pdl_prologue(PDL)

    row = tl.program_id(0)
    offs = tl.arange(0, block)
    mask = offs < n_cols
    x_row = _row_offset(row, x_shape, x_strides)
    y_row = _row_offset(row, y_shape, y_strides)
    x = tl.load(x_ptr + x_row + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / n_cols
    y = x * tl.rsqrt(var + eps) * w
    tl.store(y_ptr + y_row + offs, y, mask=mask)
