"""The gated residual: the Triton source of the ``gated_residual`` operator.

Elementwise over the flat ``[rows, width]`` activation: each element recovers its row to select the
gate row, so the row width is not bounded by one program's register budget, and the result
overwrites ``update``. The descriptor bakes the element count one launch walks (``rows × width``,
like the kernel's own tile mask), the row width and the tile block; the gate row stride is a launch
argument, because one parameter table serves many activation rows.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/modulation.py``), unchanged except that it imports
nothing but Triton: the launch-time wrapper, its operand checks and its torch types stay with the
operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["gate_row_stride"])
def _gated_residual_kernel(
    hidden_ptr,
    update_ptr,
    gate_ptr,
    row_indices_ptr,
    gate_row_stride,
    ELEMENTS: tl.constexpr,  # noqa: N803
    WIDTH: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Add one gated update into the residual stream, tile by tile.

    ``offsets`` is the flat element index of one lane; ``row`` and ``columns`` recover the
    activation row and the channel that select the gate. The gated product rounds before the
    residual sum, and ``update`` receives the sum.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    row = offsets // WIDTH
    columns = offsets % WIDTH
    modulation_row = tl.load(row_indices_ptr + row, mask=mask, other=0)
    hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(
        gate_ptr + modulation_row * gate_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # The gated update rounds before the residual sum.
    activation = hidden_ptr.dtype.element_ty
    gated = (gate * update).to(activation).to(tl.float32)
    tl.store(update_ptr + offsets, hidden + gated, mask=mask)
