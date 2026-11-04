"""The sequence's register prefix and zero tail: the ``sequence_append`` operator's Triton source.

The decoder's sequence is assembled in the runner's buffer: the caller writes its own ``rows``
rows, and this operator puts a learned register prefix after them and zeroes the tail a bucket
reserves.

```text
out[rows + r, :]              = registers_weight[r, :]     for r < registers
out[rows + registers + t, :]  = 0                          for t < tail
```

The grid covers only the rows this launch owns, so a thread cannot write the caller's rows even by
accident, and nothing past the tail is touched: a buffer shared with another producer is safe to
pass.
"""

import triton
import triton.language as tl


@triton.jit
def _sequence_append_kernel(
    out_ptr,
    registers_ptr,
    rows,
    width,
    registers,
    tail,
    block,
    pdl,
    BLOCK: tl.constexpr,
):
    """Write one block of the register rows and the zero tail.

    The flattened index runs over the owned rows only — ``registers + tail`` of them — so the row
    it names is either a register row (copied from the learned weights) or a tail row (zeroed).
    ``block`` and ``pdl`` are the launch's words: the block width is ``BLOCK`` and programmatic
    dependent launch is a launching decision.
    """
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    owned = (registers.to(tl.int64) + tail.to(tl.int64)) * width.to(tl.int64)
    live = offsets < owned

    row = offsets // width.to(tl.int64)
    column = offsets % width.to(tl.int64)
    # A register row is copied; a tail row is the output's element type's zero.
    value = tl.where(
        row < registers.to(tl.int64),
        tl.load(registers_ptr + row * width.to(tl.int64) + column, mask=live, other=0.0),
        0.0,
    )
    tl.store(
        out_ptr + (rows.to(tl.int64) + row) * width.to(tl.int64) + column,
        value.to(out_ptr.dtype.element_ty),
        mask=live,
    )
