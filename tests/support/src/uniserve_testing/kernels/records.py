"""The cache record formats in torch, as ``crates/ops/src/record.rs`` states
them: the reference the checks of the operators that write and read records
hold a kernel to.

A record is ``width`` values in one format:

- ``bf16``: BF16 values, no scale;
- ``mxfp8``: E4M3 values, one UE8M0 scale byte per 32;
- ``nvfp4``: E2M1 values two per byte (the even element in the low nibble),
  one E4M3 scale byte per 16;
- ``mxfp4``: E2M1 values two per byte, one UE8M0 scale byte per 32.

The encoding is DeepSeek's reference fake quantization: a block's scale is
``2^ceil(log2(amax · f32(1/max)))`` for a UE8M0 scale (``ceil(log2)`` read off
the FP32 bits) and ``e4m3(amax / 6)`` for NVFP4's, with ``amax`` floored at
``1e-4``, ``6 · 2^-126`` and ``6 · 2^-9`` respectively, and each value is
``code(clamp(x / scale, ±max))`` rounded to nearest even. A page of
``entries`` records keeps each record's scales after its own values
(``entry``) or every record's values and then every record's scales
(``page``).
"""  # noqa: D205

import torch

#: The values one scale covers, per packed format.
BLOCK = {"mxfp8": 32, "nvfp4": 16, "mxfp4": 32}

#: The largest E4M3 and E2M1 magnitudes.
E4M3_MAX = 448.0
E2M1_MAX = 6.0

#: The E2M1 magnitudes by the code's low three bits.
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def value_bytes(format, width):
    """The bytes of ``width`` encoded values, without their scales."""
    return {"bf16": 2 * width, "mxfp8": width}.get(format, width // 2)


def scale_bytes(format, width):
    """The bytes of the scales of ``width`` values."""
    return width // BLOCK[format] if format in BLOCK else 0


def record_bytes(format, width):
    """The bytes of one record of ``width`` values."""
    return value_bytes(format, width) + scale_bytes(format, width)


def f32_reciprocal(value):
    """``1 / value`` divided in FP32, as the Rust rule computes it."""
    one = torch.tensor(1.0, dtype=torch.float32)
    return one / torch.tensor(value, dtype=torch.float32)


def log2_ceil(x):
    """``ceil(log2(x))`` of positive FP32 values, read off their bits: the
    unbiased exponent, plus one for a non-zero mantissa.
    """  # noqa: D205
    bits = x.to(torch.float32).contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    return exponent + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def pow2(exponent):
    """``2^exponent`` for the exponents of normal FP32 values."""
    return ((exponent.to(torch.int32) + 127) << 23).view(torch.float32)


def e4m3_codes(x):
    """The E4M3 codes of FP32 ``x``, saturated at ±448, nearest even."""
    clamped = x.to(torch.float32).clamp(-E4M3_MAX, E4M3_MAX)
    return clamped.to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_values(codes):
    """The FP32 values of E4M3 codes."""
    return codes.contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def e2m1_codes(x):
    """The four-bit E2M1 codes of FP32 ``x``: nearest even on the E2M1 grid,
    saturated at ±6, the sign bit set for a negative ``x``.
    """  # noqa: D205
    magnitude = x.to(torch.float32).abs()
    # The boundaries between neighbouring codes are their midpoints, and a
    # midpoint goes to the even code.
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for boundary, inclusive, value in (
        (0.25, False, 1),
        (0.75, True, 2),
        (1.25, False, 3),
        (1.75, True, 4),
        (2.5, False, 5),
        (3.5, True, 6),
        (5.0, False, 7),
    ):
        above = magnitude >= boundary if inclusive else magnitude > boundary
        code = torch.where(above, torch.full_like(code, value), code)
    sign = (x < 0).to(torch.uint8) << 3
    return code | sign


def e2m1_values(codes):
    """The FP32 values of four-bit E2M1 codes."""
    magnitude = E2M1_VALUES.to(codes.device)[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -magnitude, magnitude)


def pack_nibbles(codes):
    """E2M1 codes two per byte along the last axis, even element low."""
    return (codes[..., 0::2] & 0x0F) | (codes[..., 1::2] << 4)


def unpack_nibbles(packed):
    """The E2M1 codes of bytes packed two per byte, even element low."""
    low, high = packed & 0x0F, packed >> 4
    return torch.stack((low, high), dim=-1).flatten(-2)


def encode(format, x):
    """Encode FP32 rows ``x``, ``[..., width]``, as records: their value
    bytes ``[..., value_bytes]`` and scale bytes ``[..., scale_bytes]``.
    """  # noqa: D205
    x = x.to(torch.float32)
    width = x.shape[-1]
    if format == "bf16":
        values = x.to(torch.bfloat16).contiguous().view(torch.uint8)
        return values, torch.zeros(*x.shape[:-1], 0, dtype=torch.uint8)

    block = BLOCK[format]
    blocks = x.reshape(*x.shape[:-1], width // block, block)
    amax = blocks.abs().amax(dim=-1)
    if format == "nvfp4":
        amax = amax.clamp_min(6.0 * 2.0**-9)
        scale_codes = e4m3_codes(amax / torch.tensor(E2M1_MAX))
        scale = e4m3_values(scale_codes)
        quotient = (blocks / scale.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
        codes = e2m1_codes(quotient).reshape(x.shape)
        return pack_nibbles(codes), scale_codes

    largest = E4M3_MAX if format == "mxfp8" else E2M1_MAX
    floor = 1e-4 if format == "mxfp8" else 6.0 * 2.0**-126
    exponent = log2_ceil(amax.clamp_min(floor) * f32_reciprocal(largest))
    quotient = (blocks / pow2(exponent).unsqueeze(-1)).clamp(-largest, largest)
    scales = (exponent + 127).to(torch.uint8)
    if format == "mxfp8":
        return e4m3_codes(quotient).reshape(x.shape), scales
    return pack_nibbles(e2m1_codes(quotient).reshape(x.shape)), scales


def decode(format, values, scales, width):
    """The FP32 values of records' value and scale bytes."""
    if format == "bf16":
        return values.contiguous().view(torch.bfloat16).to(torch.float32)

    block = BLOCK[format]
    if format == "mxfp8":
        decoded = e4m3_values(values)
    else:
        decoded = e2m1_values(unpack_nibbles(values))
    if format == "nvfp4":
        scale = e4m3_values(scales)
    else:
        scale = pow2(scales.to(torch.int32) - 127)
    shape = decoded.shape
    decoded = decoded.reshape(*shape[:-1], width // block, block)
    return (decoded * scale.unsqueeze(-1)).reshape(shape)


def fake_quantize(format, x):
    """``x`` encoded and decoded: the values a record of it holds."""
    values, scales = encode(format, x)
    return decode(format, values, scales, x.shape[-1])


def plane(format, layout, values, scales, entries):
    """The pages of records ``[pages · entries, …]`` in ``layout``:
    ``[pages, entries, record]`` bytes, or ``[pages, entries, width]`` BF16.
    """  # noqa: D205
    if format == "bf16":
        width = values.shape[-1] // 2
        return (
            values.contiguous().view(torch.bfloat16).reshape(-1, entries, width)
        )
    pages = values.shape[0] // entries
    values = values.reshape(pages, entries, -1)
    scales = scales.reshape(pages, entries, -1)
    if layout == "entry":
        return torch.cat((values, scales), dim=-1)
    page = torch.cat(
        (values.reshape(pages, -1), scales.reshape(pages, -1)), dim=-1
    )
    return page.reshape(pages, entries, -1)


def unplane(format, layout, pages, width):
    """The value and scale bytes of every record of a plane, ``[pages ·
    entries, …]`` each: the inverse of :func:`plane`.
    """  # noqa: D205
    if format == "bf16":
        values = pages.contiguous().view(torch.uint8)
        return values.reshape(-1, 2 * width), None
    count, entries, _ = pages.shape
    value, scale = value_bytes(format, width), scale_bytes(format, width)
    if layout == "entry":
        flat = pages.reshape(count * entries, -1)
        return flat[:, :value], flat[:, value:]
    flat = pages.reshape(count, -1)
    values = flat[:, : entries * value].reshape(count * entries, value)
    scales = flat[:, entries * value :].reshape(count * entries, scale)
    return values, scales


def decode_plane(format, layout, pages, width):
    """The FP32 values of every record of a plane, ``[pages, entries,
    width]``.
    """  # noqa: D205
    values, scales = unplane(format, layout, pages, width)
    decoded = decode(format, values, scales, width)
    return decoded.reshape(pages.shape[0], pages.shape[1], width)
