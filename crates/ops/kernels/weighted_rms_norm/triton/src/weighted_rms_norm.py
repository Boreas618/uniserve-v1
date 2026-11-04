"""Weighted RMS normalization: the Triton source of the ``weighted_rms_norm`` operator.

One program normalizes one contiguous activation row: the mean square reduces in
FP32, the normalized value rounds once into the output type. With ``absmax`` the
program also stores the largest magnitude of its stored row, which the caller
reduces.

The kernel bodies are the ones UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/residual.py``), unchanged except that
they import nothing but Triton: the launch-time wrapper, its operand checks and
its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["eps"])
def _weighted_rms_norm_kernel(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    eps,
    WIDTH: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Normalize one activation row and apply its learned scale."""
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)

    hidden = tl.load(
        hidden_ptr + row * WIDTH + columns, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)
    weight = tl.load(
        weight_ptr + columns, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)

    # Reduction and scaling stay in fp32; the store casts to the output dtype.
    inverse_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / WIDTH + eps)
    tl.store(
        output_ptr + row * WIDTH + columns,
        hidden * inverse_rms * weight,
        mask=columns < WIDTH,
    )


@triton.jit(do_not_specialize=["eps"])
def _weighted_rms_norm_absmax_kernel(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    partials_ptr,
    eps,
    WIDTH: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Normalize one row and publish its output magnitude."""
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)

    hidden = tl.load(
        hidden_ptr + row * WIDTH + columns, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)
    weight = tl.load(
        weight_ptr + columns, mask=columns < WIDTH, other=0.0
    ).to(tl.float32)

    inverse_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / WIDTH + eps)
    # Round before measuring so the magnitude describes the stored values.
    output = (hidden * inverse_rms * weight).to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + row * WIDTH + columns, output, mask=columns < WIDTH)

    # ``partials_ptr`` collects one FP32 magnitude per row; the caller reduces
    # them to one scalar with `uniserve_kernels.reduction.absmax` in a separate
    # launch.
    tl.store(
        partials_ptr + row,
        tl.max(
            tl.where(columns < WIDTH, tl.abs(output.to(tl.float32)), 0.0),
            axis=0,
        ),
    )
