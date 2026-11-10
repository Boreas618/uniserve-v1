"""Sigmoid-gated add: the Triton source of the ``sigmoid_gated_add`` operator.

``target[r] += sigmoid(gate[r]) · update[r]``: a row's update scaled by the
sigmoid of its own scalar logit, as a shared expert joins the routed ones. The
sigmoid, the product and the sum are FP32 and the target rounds once.

The kernel body imports nothing but Triton: the operand checks and the launch
stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["columns"])
def _sigmoid_gated_add_kernel(
    target_ptr,
    update_ptr,
    gate_ptr,
    columns,
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Add one block of one row's gated update into the target.

    Grid: ``(ceil(columns / BLOCK), rows)``. ``target`` and ``update`` are
    contiguous ``[rows, columns]`` and ``gate`` is ``[rows, 1]``.
    """
    block = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < columns
    base = row * columns

    logit = tl.load(gate_ptr + row).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-logit))
    target = tl.load(target_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(target_ptr + base + offsets, target + gate * update, mask=mask)
