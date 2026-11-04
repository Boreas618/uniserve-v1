"""Scaled residual: the Triton source of the ``scaled_residual`` operator.

Programs tile the flattened contiguous activation: each element adds one channel's
scale times the update's element into the residual, in place, with the biased
update rounding before the product and the sum.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/residual.py``), unchanged except that
it imports nothing but Triton: the launch-time wrapper, its operand checks and its
torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["elements"])
def _scaled_residual_kernel(
    hidden_ptr,
    update_ptr,
    update_bias_ptr,
    scale_ptr,
    elements,
    HAS_UPDATE_BIAS: tl.constexpr,  # noqa: N803
    WIDTH: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Apply a channel-scaled update to the residual stream."""
    # Flat element offsets address ``hidden``/``update``; the channel index
    # selects the per-channel scale and optional bias.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    columns = offsets % WIDTH

    hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_UPDATE_BIAS:
        update += tl.load(
            update_bias_ptr + columns, mask=mask, other=0.0
        ).to(tl.float32)
    scale = tl.load(scale_ptr + columns, mask=mask, other=0.0).to(tl.float32)

    tl.store(hidden_ptr + offsets, hidden + update * scale, mask=mask)
