"""The ``cache_copy_region`` operator's Triton source: an FP32 span into the destination's units.

One program per (column, unit slot), because the FP8 destination's scale is a property of the whole
written span and not of a token:

```text
dense  (scaled = false)
    values[column, unit, (offset + token)·row + i] = round_to_stored(source[token, column, i])
    initialized[column, unit]                      = 1

FP8    (scaled = true)
    maximum  = max(|source[:, column]|)                     # one amax over the span
    proposed = max(maximum, 1e-12) / 448.0
    updated  = initialized ? max(scale, proposed) : proposed        # growth only, once written
    every resident value of the plane *= old / updated
    scale[column, unit] = updated
    values[column, unit, (offset + token)·row + i] = encode(source[token, column, i] / updated)
    initialized[column, unit] = 1                           # last
```

The three constants are the reference's (`uniserve/prefix_cache/mha.py:257-304`): `448.0` is the
largest finite E4M3 value, `1e-12` keeps an all-zero span from producing a zero scale, and the growth
is `max` so a unit that already holds large values is never rescaled downward. The order matters:
rescale the residents *before* the new scale is stored, and raise the flag last, because a reader that
sees the flag set is entitled to read every row of the unit.

**A per-token grid would be wrong, not merely slower**: each token's program would compute its own
partial amax and the cell would hold whichever program wrote last. That is a different tensor rather
than a different rounding, which is why the operator's grid is `(columns, units_per_page, 1)`.

Two conversions are stated here rather than left to a store, because Triton's interpreter and a device
disagree on them:

* **BF16** — round to nearest even in the bit pattern (bias by half the discarded field plus the
  surviving low bit, then truncate). The interpreter truncates a BF16 store; a device rounds.
* **E4M3** — clamp to `±448.0` *before* the cast. In range the interpreter's conversion is nearest even,
  as the hardware's is; out of range it wraps where the hardware saturates, and the reference clamps
  anyway (`(values / scale).clamp(-448.0, 448.0)`).
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
def _encode_fp8(value):
    """One FP32 into an E4M3 byte: round to nearest even, saturating at the format's 448.

    Written in integers rather than as ``value.to(tl.float8e4nv)`` because an instrument that converts
    FP8 in software can convert it *wrongly*: Triton's interpreter turns ``7.943263`` into ``4.0`` and
    ``-449`` into ``-448`` only by luck of the clamp, while a device's ``cvt.rn.satfinite.e4m3x2.f32``
    is correct. Stating the rule here makes the kernel's answer the same on both.

    The format is one sign bit, four exponent bits (bias 7) and three mantissa bits, with subnormals
    below ``2**-6`` whose quantum is ``2**-9`` and whose largest pattern (``0x08``) is the first normal
    value: rounding the subnormal mantissa up to eight therefore lands exactly where the carry belongs.
    """
    sign = value.to(tl.int32, bitcast=True) & -2147483648
    magnitude = tl.minimum(tl.abs(value), 448.0)
    bits = magnitude.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF

    # Normal path: the value keeps a normal exponent in the target format. The three kept mantissa bits
    # round to nearest even: past the midpoint, or exactly on it when the kept value is odd.
    target = exponent - 120
    kept = mantissa >> 20
    round_bit = (mantissa >> 19) & 1
    sticky = mantissa & 0x7FFFF
    carry = (round_bit == 1) & ((sticky != 0) | ((kept & 1) == 1))
    kept = kept + carry.to(tl.int32)
    carry_up = kept == 8
    kept = tl.where(carry_up, 0, kept)
    target = target + carry_up.to(tl.int32)
    normal = (target << 3) | kept
    # Subnormal path: the value is under `2**-6`, so its representation is a count of `2**-9` quanta,
    # rounded to nearest even in the same way. `2**-9` is 512 in the value's own units.
    quanta = magnitude * 512.0
    whole = quanta.to(tl.int32)
    fraction = quanta - whole.to(tl.float32)
    up = (fraction > 0.5) | ((fraction == 0.5) & ((whole & 1) == 1))
    subnormal = whole + up.to(tl.int32)
    result = tl.where(target >= 1, normal, subnormal)
    # A zero magnitude encodes as a zero byte, not as the sign bit over a zero exponent.
    result = tl.where(magnitude == 0.0, 0, result)
    return (result | (sign >> 24 & 0x80)).to(tl.uint8)


@triton.jit
def _decode_fp8(byte):
    """One E4M3 byte back to FP32, exactly: every E4M3 value is representable in FP32."""
    return byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)


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
    SCALED: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    """One (column, unit slot): the span's scale, the residents' rescale, the span's write, the flag.

    ``values_ptr`` is typed with the stored element type: BF16 or F16 for a dense pool, ``*u8`` for an
    FP8 one (Triton's signature table has no pointer to a float8, so the byte is bitcast to
    ``tl.float8e4nv`` here). ``scale_ptr`` is the per-unit F32 scale for an FP8 destination; a dense pool
    passes no scale view and the operator's flag slot then carries the initialized view's address, which
    this kernel never reads as a scale because ``SCALED`` is false.

    The launch's scalars arrive as ordinary numbers, so they are widened to int64 once, before any
    address arithmetic: a plane's index is a product of extents and a 32-bit product would wrap into
    another unit's bytes with no shape to complain about.
    """
    column = tl.program_id(0)
    slot = tl.program_id(1)
    unit = tl.load(units_ptr + slot)
    unit = unit.to(tl.int64)
    flag_index = column * units_total + unit
    plane_base = (column * units_total + unit) * PLANE_ELEMENTS
    row = tl.arange(0, ROW_POW2)
    row = row.to(tl.int64)
    live = row < ROW_SIZE

    if SCALED:
        initialized = tl.load(initialized_ptr + flag_index) != 0
        # Pass one: the span's largest magnitude, which is the whole reason for one program per unit.
        maximum = 0.0
        for token in tl.range(0, count):
            index = (token * COLUMNS + column) * ROW_SIZE + row
            value = tl.load(source_ptr + index, mask=live, other=0.0)
            maximum = tl.maximum(maximum, tl.max(tl.abs(value)))
        proposed = tl.maximum(maximum, 1e-12) / 448.0
        previous = tl.load(scale_ptr + flag_index)
        updated = tl.where(initialized, tl.maximum(previous, proposed), proposed)
        # The residents were encoded with `previous`; a reader multiplies them by `updated`, so each one
        # is decoded, scaled by the ratio and encoded again. A fresh cell has nothing resident.
        if initialized:
            ratio = previous / updated
            stride = tl.arange(0, BLOCK)
            stride = stride.to(tl.int64)
            for start in tl.range(0, PLANE_ELEMENTS, BLOCK):
                index = start + stride
                inside = index < PLANE_ELEMENTS
                byte = tl.load(values_ptr + plane_base + index, mask=inside, other=0)
                rescaled = _decode_fp8(byte) * ratio
                tl.store(
                    values_ptr + plane_base + index,
                    _encode_fp8(rescaled),
                    mask=inside,
                )
        tl.store(scale_ptr + flag_index, updated)
        # Pass two: the span itself, divided by the scale every reader will multiply back.
        for token in tl.range(0, count):
            index = (token * COLUMNS + column) * ROW_SIZE + row
            value = tl.load(source_ptr + index, mask=live, other=0.0)
            destination = plane_base + (offset + token) * ROW_SIZE + row
            tl.store(values_ptr + destination, _encode_fp8(value / updated), mask=live)
    else:
        for token in tl.range(0, count):
            index = (token * COLUMNS + column) * ROW_SIZE + row
            value = tl.load(source_ptr + index, mask=live, other=0.0)
            if BF16:
                value = _round_bf16(value)
            destination = plane_base + (offset + token) * ROW_SIZE + row
            tl.store(
                values_ptr + destination,
                value.to(values_ptr.dtype.element_ty),
                mask=live,
            )

    # Last, and once per unit: a reader that sees the flag set may read every row of the unit.
    tl.store(initialized_ptr + flag_index, 1)
