"""A learned position table resampled on the device: ``position_interpolate``'s Triton source.

A vision tower is trained at one image resolution and runs at several, so its learned
``[in_side · in_side, channels]`` position table is resampled to the grid the image has. The **host
computes the taps** — four indices into the table's points and four FP32 weights per output point —
and the program gathers and weights:

```text
out[j, c] = Σ_{t < 4} weights[j, t] · table[indices[j, t], c]
```

Which resampling rule the taps follow is the caller's: corner alignment or half-pixel, clamped or
extended, and each tower has its own. A shared operator must not bake one caller's convention in, so
there is no side, no ratio and no division here — only the four taps the call hands over.

``indices`` is **i32**, which is the caller's own depth for a table of points; the addresses are
formed in i64 so a wide table cannot wrap. The products and their sum are FP32 and the store rounds
once into the output's dtype, which is the operator's tolerance: ``Tolerance::dot`` over four terms
with the output's rounding.

A program is one (output point, block of channels), so the four taps of a point are read and summed
by one program — the sum's order is fixed and two programs never write the same element.
"""

import triton
import triton.language as tl

#: Taps one output point reads: a bilinear resample of a two-dimensional grid.
TAPS = tl.constexpr(4)


@triton.jit
def _position_interpolate_kernel(
    table_ptr,
    indices_ptr,
    weights_ptr,
    out_ptr,
    points,
    channels,
    block,
    pdl,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather four table rows and write one block of one output point.

    ``CHANNELS`` is the table's feature width and ``BLOCK`` the lanes one program covers, both the
    descriptor's. ``points``, ``channels``, ``block`` and ``pdl`` are the launch's words: the grid
    names the output point and the channel slice.
    """
    point = tl.program_id(0).to(tl.int64)
    lanes = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    live = lanes < CHANNELS

    out_dtype = out_ptr.dtype.element_ty
    taps = tl.arange(0, TAPS)

    # The caller's four taps and four weights of this point. The indices stay i32 as the operator
    # declares them; the addresses below are i64 so a wide table cannot wrap.
    indices = tl.load(indices_ptr + point * TAPS + taps)
    weights = tl.load(weights_ptr + point * TAPS + taps)

    rows = indices.to(tl.int64)[:, None] * CHANNELS + lanes[None, :]
    values = tl.load(
        table_ptr + rows, mask=live[None, :], other=0.0
    ).to(tl.float32)
    weighted = tl.sum(values * weights[:, None], axis=0)

    tl.store(
        out_ptr + point * CHANNELS + lanes,
        weighted.to(out_dtype, fp_downcast_rounding="rtne"),
        mask=live,
    )
