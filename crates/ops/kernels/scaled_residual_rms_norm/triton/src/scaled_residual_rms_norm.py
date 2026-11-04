"""Scaled residual RMS normalization: the Triton source of the ``scaled_residual_rms_norm``
operator.

One program folds a channel-scaled, optionally biased update into the residual
row, stores the sum rounded to the residual's dtype, and stores the RMS-normalized
*unrounded* sum, as the eager expression does. With ``absmax`` the program also
stores the largest magnitude of its stored normalized row.

The kernel bodies are the ones UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/residual.py``), unchanged except that
they import nothing but Triton: the launch-time wrapper, its operand checks and
its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["eps"])
def _scaled_residual_rms_norm_kernel(
    hidden_ptr,
    update_ptr,
    update_bias_ptr,
    scale_ptr,
    weight_ptr,
    output_ptr,
    eps,
    HAS_UPDATE_BIAS: tl.constexpr,  # noqa: N803
    WIDTH: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Update a residual row in place and emit its RMS-normalized value."""
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    offsets = row * WIDTH + columns

    hidden = tl.load(
        hidden_ptr + offsets, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)
    update = tl.load(
        update_ptr + offsets, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)
    if HAS_UPDATE_BIAS:
        update += tl.load(
            update_bias_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
    scale = tl.load(
        scale_ptr + columns, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)

    residual = hidden + update * scale
    inverse_rms = tl.rsqrt(tl.sum(residual * residual, axis=0) / WIDTH + eps)
    weight = tl.load(
        weight_ptr + columns, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)

    # ``hidden`` receives the sum rounded to its dtype; the normalized output
    # derives from the unrounded FP32 sum.
    tl.store(hidden_ptr + offsets, residual, mask=columns < WIDTH)
    tl.store(
        output_ptr + offsets,
        residual * inverse_rms * weight,
        mask=columns < WIDTH,
    )


@triton.jit(do_not_specialize=["eps"])
def _scaled_residual_rms_norm_absmax_kernel(
    hidden_ptr,
    update_ptr,
    update_bias_ptr,
    scale_ptr,
    weight_ptr,
    output_ptr,
    partials_ptr,
    eps,
    HAS_UPDATE_BIAS: tl.constexpr,  # noqa: N803
    WIDTH: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Fuse residual update, RMS normalization, and magnitude capture."""
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    offsets = row * WIDTH + columns

    hidden = tl.load(
        hidden_ptr + offsets, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)
    update = tl.load(
        update_ptr + offsets, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)
    if HAS_UPDATE_BIAS:
        update += tl.load(
            update_bias_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
    scale = tl.load(
        scale_ptr + columns, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)

    residual = hidden + update * scale
    inverse_rms = tl.rsqrt(tl.sum(residual * residual, axis=0) / WIDTH + eps)
    weight = tl.load(
        weight_ptr + columns, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)
    output = (residual * inverse_rms * weight).to(output_ptr.dtype.element_ty)

    tl.store(hidden_ptr + offsets, residual, mask=columns < WIDTH)
    tl.store(output_ptr + offsets, output, mask=columns < WIDTH)
    tl.store(
        partials_ptr + row,
        tl.max(
            tl.where(columns < WIDTH, tl.abs(output.to(tl.float32)), 0.0),
            axis=0,
        ),
    )
