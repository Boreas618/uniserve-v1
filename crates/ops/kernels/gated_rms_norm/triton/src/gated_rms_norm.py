"""Gated RMS normalization: the Triton source of the ``gated_rms_norm`` operator.

One program normalizes one contiguous row with FP32 statistics, scales it by the
per-channel weight and by the SiLU of the same row of the gate, and rounds once
when storing: the output norm of a gated delta network, normalized before it is
gated (FLA's ``RMSNormGated`` with ``norm_before_gate``). The row tile is the
descriptor's; the row count comes from the grid and ``columns`` from the launch.

The kernel body imports nothing but Triton: the operand checks and the launch
stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["columns"])
def _gated_rms_norm_kernel(
    x_ptr,
    gate_ptr,
    w_ptr,
    y_ptr,
    columns,
    eps,
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Normalize and gate one row per program.

    ``x``, ``gate`` and ``y`` are contiguous ``[rows, columns]`` views and ``w``
    the ``[columns]`` weight. Every operand of the row is loaded before the
    store, so the output may alias either input.
    """
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < columns
    row_base = row * columns
    x = tl.load(x_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / columns
    normalized = x * tl.rsqrt(variance + eps) * w
    y = normalized * (gate / (1.0 + tl.exp(-gate)))
    tl.store(y_ptr + row_base + offsets, y, mask=mask)
