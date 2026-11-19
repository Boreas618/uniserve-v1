"""The Triton source of the ``record_write`` operator.

One program writes one token's record into its paged cache plane
(``ops::record::RecordWrite``). It reads the token's row, normalizes it by its
RMS and a learned gain when the compilation normalizes, turns the row's
trailing ``ROTATED`` channels in adjacent pairs by the angles of position
``entry · RATIO``, and encodes the result into the record's format:

- ``bf16`` stores the row as BF16 elements of a BF16 plane;
- ``mxfp8`` stores E4M3 values and one UE8M0 scale byte per 32;
- ``nvfp4`` stores E2M1 values two per byte and one E4M3 scale byte per 16;
- ``mxfp4`` stores E2M1 values two per byte and one UE8M0 scale byte per 32.

The encoding is the operator's (``crates/ops/src/record.rs``): a block's
power-of-two scale has the exponent ``ceil(log2(amax · f32(1/max)))`` read off
the FP32 bits of the product, never a ``log2`` evaluation, and every code is
rounded to nearest even and saturated. The E2M1 codes are chosen by their
midpoints in arithmetic, which Triton's interpreter runs as the device does.

The normalization and the rotation each round into the rows' element type, as
the reference rounds its normalized row and writes its rotation back into it;
the launch table compiles without contracting a product into a sum
(``enable_fp_fusion = false``), so each product and each sum rounds once. The
cosines and sines are libdevice's full-range FP32 functions, the ones
``rotary_factors`` evaluates.

The record lands at slot ``entry mod ENTRIES`` of page ``table[row, (entry /
ENTRIES) mod max_pages]``; a page holds each record's scales after its own
values, or (``PAGED``) every record's values and then every record's scales.
A token whose entry is negative writes nothing.

The kernel body imports nothing but Triton: the shape checks and the plane's
allocation stay with the operator's Rust side.
"""

import struct

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

#: The record formats, as the launch table's ``FORMAT`` numbers them.
BF16 = tl.constexpr(0)
MXFP8 = tl.constexpr(1)
NVFP4 = tl.constexpr(2)
MXFP4 = tl.constexpr(3)

#: The largest E4M3 and E2M1 magnitudes.
E4M3_MAX = tl.constexpr(448.0)
E2M1_MAX = tl.constexpr(6.0)


def _f32(value):
    """``value`` rounded to the nearest FP32 value, as a Python float."""
    return struct.unpack("f", struct.pack("f", value))[0]


#: ``1 / 448`` and ``1 / 6`` in FP32, the factors the operator's rule
#: multiplies a block's peak by. Their binary expansions repeat (``1/7`` and
#: ``1/3`` times a power of two) and so hold no rounding tie, which makes the
#: FP64 quotient rounded to FP32 the FP32 quotient itself.
INV_E4M3_MAX = tl.constexpr(_f32(1.0 / 448.0))
INV_E2M1_MAX = tl.constexpr(_f32(1.0 / 6.0))

#: The floors of a block's largest magnitude: MXFP8's, NVFP4's and MXFP4's.
MXFP8_FLOOR = tl.constexpr(1e-4)
NVFP4_FLOOR = tl.constexpr(6.0 * 2.0**-9)
MXFP4_FLOOR = tl.constexpr(6.0 * 2.0**-126)


@triton.jit
def _log2_ceil(x):
    """``ceil(log2(x))`` of positive FP32 values, read off their bits: the
    unbiased exponent, plus one when the mantissa is not zero.
    """  # noqa: D205
    bits = x.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF).to(tl.int32) - 127
    return exponent + ((bits & 0x7FFFFF) != 0).to(tl.int32)


@triton.jit
def _pow2(exponent):
    """``2^exponent`` for exponents of normal FP32 values."""
    return ((exponent + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _e2m1_codes(quotient):
    """The four-bit E2M1 codes of values already clamped to ``±6``.

    The boundary between neighbouring magnitudes is their midpoint, and a
    midpoint goes to the even code, so the midpoints that end a code's range
    are inclusive for the even codes and exclusive for the odd ones. The sign
    bit is set for a negative value.
    """
    magnitude = tl.abs(quotient)
    code = tl.where(magnitude > 0.25, 1, 0)
    code = tl.where(magnitude >= 0.75, 2, code)
    code = tl.where(magnitude > 1.25, 3, code)
    code = tl.where(magnitude >= 1.75, 4, code)
    code = tl.where(magnitude > 2.5, 5, code)
    code = tl.where(magnitude >= 3.5, 6, code)
    code = tl.where(magnitude > 5.0, 7, code)
    return code | tl.where(quotient < 0.0, 8, 0)


@triton.jit
def _packed_nibbles(codes, WIDTH: tl.constexpr):
    """Four-bit codes ``[WIDTH]`` two per byte, the even element in the low
    nibble: ``[WIDTH / 2]`` bytes.
    """  # noqa: D205
    low, high = tl.split(tl.reshape(codes, (WIDTH // 2, 2)))
    return (low | (high << 4)).to(tl.uint8)


@triton.jit
def _rotated(
    values, inv_freq, position, WIDTH: tl.constexpr, ROTATED: tl.constexpr, ROW
):
    """``values`` with their trailing ``ROTATED`` channels turned in adjacent
    pairs by the angles of ``position``, each turned value rounded to the row
    element type ``ROW``.

    Pair ``j`` of the turned channels (the pair of row channels ``WIDTH −
    ROTATED + 2j`` and the one after it) turns by ``f32(position) ·
    inv_freq[j]``; the pairs before them keep a unit factor, which moves no
    value.
    """  # noqa: D205
    even, odd = tl.split(tl.reshape(values, (WIDTH // 2, 2)))
    pair = tl.arange(0, WIDTH // 2) - (WIDTH - ROTATED) // 2
    turning = pair >= 0
    frequency = tl.load(inv_freq + tl.maximum(pair, 0), mask=turning, other=0.0)
    angle = position.to(tl.float32) * frequency
    cosine = tl.where(turning, libdevice.cos(angle), 1.0)
    sine = tl.where(turning, libdevice.sin(angle), 0.0)

    turned_even = (even * cosine - odd * sine).to(ROW).to(tl.float32)
    turned_odd = (odd * cosine + even * sine).to(ROW).to(tl.float32)
    return tl.interleave(turned_even, turned_odd)


@triton.jit(do_not_specialize=["tokens", "x_stride", "max_pages"])
def _record_write_kernel(
    x,
    weight,
    inv_freq,
    entries,
    rows,
    table,
    plane,
    tokens,
    x_stride,
    max_pages,
    eps,
    WIDTH: tl.constexpr,
    NORM: tl.constexpr,
    ROTATED: tl.constexpr,
    RATIO: tl.constexpr,
    FORMAT: tl.constexpr,
    PAGED: tl.constexpr,
    ENTRIES: tl.constexpr,
    BLOCK: tl.constexpr,
    VALUE_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
):
    """Write token ``program_id(0)``'s record. Grid: ``(tokens,)``.

    ``x`` is ``[tokens, WIDTH]`` rows ``x_stride`` elements apart, ``weight``
    the ``[WIDTH]`` gain (read when ``NORM``), ``inv_freq`` the ``[ROTATED /
    2]`` FP32 inverse frequencies (read when ``ROTATED`` is not zero),
    ``entries`` the I32 entry and ``rows`` the table row of each token,
    ``table`` the ``[rows, max_pages]`` page table and ``plane`` the ``[pages,
    ENTRIES, record]`` plane. ``BLOCK`` is the values one scale covers and
    ``VALUE_BYTES``/``SCALE_BYTES`` a record's value and scale bytes.
    """
    token = tl.program_id(0).to(tl.int64)
    entry = tl.load(entries + token)
    if entry < 0:
        return

    # The record's page and slot: the table is read modulo its width, so a
    # table narrower than the request is a ring of pages.
    row = tl.load(rows + token).to(tl.int64)
    column = ((entry // ENTRIES) % max_pages).to(tl.int64)
    page = tl.load(table + row * max_pages + column).to(tl.int64)
    slot = (entry % ENTRIES).to(tl.int64)

    channels = tl.arange(0, WIDTH)
    values = tl.load(x + token * x_stride + channels).to(tl.float32)
    if NORM:
        gain = tl.load(weight + channels).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / WIDTH
        normalized = values * tl.rsqrt(variance + eps) * gain
        values = normalized.to(x.dtype.element_ty).to(tl.float32)
    if ROTATED > 0:
        values = _rotated(
            values, inv_freq, entry * RATIO, WIDTH, ROTATED, x.dtype.element_ty
        )

    if FORMAT == BF16:
        tl.store(
            plane + (page * ENTRIES + slot) * WIDTH + channels,
            values.to(tl.bfloat16),
        )
    else:
        # A packed page is `ENTRIES` records of `VALUE_BYTES + SCALE_BYTES` bytes.
        record = VALUE_BYTES + SCALE_BYTES
        base = plane + page * (ENTRIES * record)
        if PAGED:
            value_at = base + slot * VALUE_BYTES
            scale_at = base + ENTRIES * VALUE_BYTES + slot * SCALE_BYTES
        else:
            value_at = base + slot * record
            scale_at = value_at + VALUE_BYTES

        blocks = tl.reshape(values, (WIDTH // BLOCK, BLOCK))
        peak = tl.max(tl.abs(blocks), axis=1)
        scale_lanes = tl.arange(0, SCALE_BYTES)
        value_lanes = tl.arange(0, VALUE_BYTES)
        if FORMAT == NVFP4:
            # The scale is E4M3 of the block's peak over E2M1's largest, saturated.
            ratio = tl.math.div_rn(tl.maximum(peak, NVFP4_FLOOR), E2M1_MAX)
            scale_codes = tl.minimum(ratio, E4M3_MAX).to(tl.float8e4nv)
            scale = scale_codes.to(tl.float32)
            quotient = tl.math.div_rn(blocks, scale[:, None])
            codes = _e2m1_codes(tl.clamp(quotient, -E2M1_MAX, E2M1_MAX))
            tl.store(
                value_at + value_lanes,
                _packed_nibbles(tl.reshape(codes, (WIDTH,)), WIDTH),
            )
            tl.store(
                scale_at + scale_lanes, scale_codes.to(tl.uint8, bitcast=True)
            )
        else:
            if FORMAT == MXFP8:
                exponent = _log2_ceil(
                    tl.maximum(peak, MXFP8_FLOOR) * INV_E4M3_MAX
                )
            else:
                exponent = _log2_ceil(
                    tl.maximum(peak, MXFP4_FLOOR) * INV_E2M1_MAX
                )
            # A power-of-two scale divides exactly, so its reciprocal multiplies
            # to the same quotient.
            quotient = blocks * _pow2(-exponent)[:, None]
            if FORMAT == MXFP8:
                codes = tl.clamp(quotient, -E4M3_MAX, E4M3_MAX).to(
                    tl.float8e4nv
                )
                tl.store(
                    value_at + value_lanes,
                    tl.reshape(codes.to(tl.uint8, bitcast=True), (WIDTH,)),
                )
            else:
                codes = _e2m1_codes(tl.clamp(quotient, -E2M1_MAX, E2M1_MAX))
                tl.store(
                    value_at + value_lanes,
                    _packed_nibbles(tl.reshape(codes, (WIDTH,)), WIDTH),
                )
            tl.store(scale_at + scale_lanes, (exponent + 127).to(tl.uint8))


def derive(fields, parameters, target):
    """A record's scale block and its value and scale bytes.

    Raises:
        SystemExit: When the rotation is wider than the row, or the row is no
            whole number of the format's scale blocks.
    """
    width, rotated, fmt = fields["width"], fields["rotated"], fields["format"]
    if rotated > width:
        raise SystemExit(
            f"a rotation of {rotated} channels exceeds a row of {width}"
        )

    block = {"bf16": 1, "mxfp8": 32, "nvfp4": 16, "mxfp4": 32}[fmt]
    if width % block:
        raise SystemExit(
            f"{fmt} records of {width} values: a multiple of {block}"
        )
    value_bytes = {"bf16": 2 * width, "mxfp8": width}.get(fmt, width // 2)
    return {
        "block": block,
        "value_bytes": value_bytes,
        "scale_bytes": 0 if fmt == "bf16" else width // block,
    }
