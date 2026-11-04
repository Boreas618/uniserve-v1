"""Scaled residual layer normalization: the Triton source of the ``scaled_residual_layer_norm``
operator.

One program folds a channel-scaled, optionally biased update into the residual row
and stores the layer-normalized, affine-transformed result; the residual itself is
unchanged. With ``absmax`` the program also stores the largest magnitude of its
stored row.

The kernel bodies are the ones UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/residual.py``), unchanged except that
they import nothing but Triton: the launch-time wrapper, its operand checks and
its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["eps"])
def _scaled_residual_layer_norm_kernel(
    hidden_ptr,
    update_ptr,
    update_bias_ptr,
    scale_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    eps,
    HAS_UPDATE_BIAS: tl.constexpr,  # noqa: N803
    WIDTH: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Fuse a scaled residual update with layer normalization."""
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
    mean = tl.sum(residual, axis=0) / WIDTH
    # Masked lanes center to zero so they contribute nothing to the variance.
    centered = tl.where(columns < WIDTH, residual - mean, 0.0)
    inverse_std = tl.rsqrt(tl.sum(centered * centered, axis=0) / WIDTH + eps)
    weight = tl.load(
        weight_ptr + columns, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)
    bias = tl.load(bias_ptr + columns, mask=columns < WIDTH, other=0.0).to(
        tl.float32
    )

    tl.store(
        output_ptr + offsets,
        centered * inverse_std * weight + bias,
        mask=columns < WIDTH,
    )


@triton.jit(do_not_specialize=["eps"])
def _scaled_residual_layer_norm_absmax_kernel(
    hidden_ptr,
    update_ptr,
    update_bias_ptr,
    scale_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    partials_ptr,
    eps,
    HAS_UPDATE_BIAS: tl.constexpr,  # noqa: N803
    WIDTH: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Compute fused residual layer norm and output magnitude."""
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
    mean = tl.sum(residual, axis=0) / WIDTH
    # Masked lanes center to zero so they contribute nothing to the variance.
    centered = tl.where(columns < WIDTH, residual - mean, 0.0)
    inverse_std = tl.rsqrt(tl.sum(centered * centered, axis=0) / WIDTH + eps)
    weight = tl.load(
        weight_ptr + columns, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)
    bias = tl.load(bias_ptr + columns, mask=columns < WIDTH, other=0.0).to(
        tl.float32
    )
    output = (centered * inverse_std * weight + bias).to(
        output_ptr.dtype.element_ty
    )

    tl.store(output_ptr + offsets, output, mask=columns < WIDTH)
    tl.store(
        partials_ptr + row,
        tl.max(
            tl.where(columns < WIDTH, tl.abs(output.to(tl.float32)), 0.0),
            axis=0,
        ),
    )
