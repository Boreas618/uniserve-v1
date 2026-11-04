"""The ``cache_copy_region`` operator's Triton source: an FP32 span into the destination's units.

```text
values[column, unit, (offset + token)·row + i] = round_to_stored(source[token, column, i])
initialized[column, unit]                      = 1
```

where ``row = heads · head_dim`` and ``unit = units[slot]``. The source is the contiguous FP32 block
the conversion step decoded -- ``[count, columns, heads, head_dim]`` -- and the destination is the
pool's own unit field: ``[columns, units, plane_bytes]`` of raw bytes, ``plane_bytes`` covering one
page of ``block_size · heads · head_dim`` values of the stored element type.

**This program is the dense half.** A dense pool stores the values themselves, so the write is one
rounding into the stored type and nothing to coordinate; an FP8 pool also grows the unit's scale to
cover the span and rescales the unit's resident bytes first, which is a reduction over the *whole*
span and therefore cannot be raced by one program per token. The grid today is
``(columns, units_per_page, count)``, and the scaled half waits for a grid that has one program per
(column, unit) to do that reduction in.

**The flag is raised once per unit, by the token-0 program of each (column, slot).** Every program of
the axis would store the same constant, which is harmless but pointless traffic on the device's
slowest path.

Round-to-nearest-even is what the reference's copy does into a BF16 or F16 tensor, and it is stated
here rather than left to the store: a BF16 store's conversion is the hardware's round-to-nearest on a
device, but Triton's interpreter *truncates* it, so a kernel that relied on the store would be
bit-exact on one instrument and one ulp off on the other. The bit trick below (bias by half the
discarded field plus the surviving low bit, then truncate) is round-to-nearest-even on both, which is
also why ``rgb_bytes`` writes its tie-to-even the same way instead of calling a library.
"""

import triton
import triton.language as tl


@triton.jit
def _round_bf16(value):
    """Round one FP32 to BF16's own precision, nearest even, in the bit pattern.

    The low sixteen bits of an FP32 are what a BF16 discards. Adding ``0x7FFF`` rounds that field up
    whenever it is past its midpoint, and adding the surviving low bit as well makes an exact midpoint
    go to the even neighbour; masking the field off then leaves the rounded value, exactly representable
    in BF16, so the store's own conversion no longer decides the answer.
    """
    bits = value.to(tl.int32, bitcast=True)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    # `-65536` is `0xFFFF0000` as a signed 32-bit mask; the literal form does not fit the type.
    return (rounded & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _cache_copy_region_kernel(
    values_ptr,
    scale_ptr,
    initialized_ptr,
    source_ptr,
    units_ptr,
    offset,
    count,
    units_total,
    ROW_SIZE: tl.constexpr,
    ROW_POW2: tl.constexpr,
    PLANE_ELEMENTS: tl.constexpr,
    COLUMNS: tl.constexpr,
    BF16: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    """One token's row of one unit of one column.

    ``scale_ptr`` is unused here: a dense pool passes no scale view, and the operator's flag slot then
    carries the initialized view's address, which the ABI declares as the same ``*u8``. ``values_ptr``
    is typed with the stored element type rather than as bytes, because ``plane_bytes`` is a whole
    number of stored elements and every offset below is one.
    """
    column = tl.program_id(0)
    slot = tl.program_id(1)
    token = tl.program_id(2)

    unit = tl.load(units_ptr + slot)
    # The unit ids are i32 and the address arithmetic is a product of extents: one signedness, or
    # Triton refuses the sum rather than guessing which was meant.
    unit = unit.to(tl.int64)
    offset = offset.to(tl.int64)
    units_total = units_total.to(tl.int64)

    row = tl.arange(0, ROW_POW2)
    live = row < ROW_SIZE

    source_index = (token.to(tl.int64) * COLUMNS + column) * ROW_SIZE + row
    value = tl.load(source_ptr + source_index, mask=live, other=0.0)

    destination = (
        (column.to(tl.int64) * units_total + unit) * PLANE_ELEMENTS
        + (offset + token.to(tl.int64)) * ROW_SIZE
        + row
    )
    if BF16:
        value = _round_bf16(value)
    tl.store(values_ptr + destination, value.to(values_ptr.dtype.element_ty), mask=live)

    # One program per (column, unit) raises the flag; the others add nothing by storing it again.
    if token == 0:
        tl.store(initialized_ptr + column * units_total + unit, 1)
