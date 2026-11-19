"""The MX operands' encoding and scale layouts, as the checks read them.

An MX operand (``crates/ops/src/mx.rs``) holds blocks of 32 elements along the
contraction, each with one UE8M0 scale byte ``b`` standing for ``2^(b - 127)``,
and its elements as E4M3 codes (MXFP8) or E2M1 codes packed two a byte, the
lower column in the low nibble (MXFP4). The scales are laid out as ``mx::
ScaleLayout`` defines it for a matrix of ``rows`` rows and ``blocks = columns /
32`` scale columns:

- ``linear``: ``[rows, blocks]`` bytes, row major;
- ``swizzled_128x4``: rows padded to 128 and blocks to 4, the 128x4 atom of
  block-scaled tensor-core operands (:func:`swizzled_offsets`);
- ``packed_mn_major``: four consecutive scales of a row in one ``u32`` word,
  block ``4g + i`` in byte ``i``, word ``(g, row)`` at ``g * padded + row``
  with the rows padded to four.

:func:`encode_mxfp8` is the operator's encoding formula, which the encoder's
check holds bytes to and the products' checks encode their operands with.
"""

import ctypes

import torch

from uniserve_testing.kernels.tensor.operands import UNIT, gamma

#: Elements one scale covers.
BLOCK = 32

#: Scales one packed word holds, and the rows a packed column is padded to.
WORD_SCALES = 4
WORD_ROWS = 4

#: The largest finite E4M3 magnitude.
E4M3_MAX = 448.0

#: The smallest block magnitude the encoding takes.
AMAX_FLOOR = 1e-4

#: E2M1's values by code, the sign in bit 3, in FP32 (they are exact there).
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_VALUES = torch.tensor(
    [*E2M1, *(-value for value in E2M1)], dtype=torch.float32
)


def scale_values(scale_bytes):
    """UE8M0 bytes as the powers of two they stand for, in FP32."""
    exponent = scale_bytes.to(torch.int32) - 127
    return torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)


def ocp_exponents(values):
    """The scale byte of each 32-element block of ``values`` ``[rows, columns]``
    (FP32) by the OCP rule (``ops::mx::ScaleRule::Ocp``): the exponent field of
    the block's largest magnitude less E4M3's largest exponent, 8, at least 0.
    """  # noqa: D205
    rows, columns = values.shape
    blocks = values.float().reshape(rows, columns // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1)
    field = (amax.view(torch.int32) >> 23) & 0xFF
    return (field - 8).clamp_min(0).to(torch.uint8)


def block_exponents(values):
    """The scale byte of each 32-element block of ``values`` ``[rows, columns]``
    (FP32) by the covering rule (``ops::mx::ScaleRule::Covering``), the
    activation encoding's: the least power of two at or
    above ``fl(max(amax, 1e-4) · fl(1/448))``, read from that product's bits.
    """  # noqa: D205
    rows, columns = values.shape
    blocks = values.reshape(rows, columns // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp_min(AMAX_FLOOR)
    inverse = torch.tensor(1.0 / E4M3_MAX, dtype=torch.float32)
    ratio = (amax * inverse).view(torch.int32)
    exponent = ((ratio >> 23) & 0xFF) - 127 + ((ratio & 0x7FFFFF) != 0).int()
    return (exponent + 127).to(torch.uint8)


def encode_mxfp8(values):
    """``values`` ``[rows, columns]`` (any float type) as their MXFP8 encoding:
    ``(codes, scales)``, the E4M3 codes ``[rows, columns]`` and the linear
    scale bytes ``[rows, columns / 32]``, as the operator's formula states them.
    """  # noqa: D205
    values = values.float()
    rows, columns = values.shape
    scales = block_exponents(values)
    scale = scale_values(scales).repeat_interleave(BLOCK, dim=1)
    codes = (values / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return codes, scales


def decode_mxfp8(codes, scales):
    """The values E4M3 ``codes`` stand for under linear ``scales``, in FP64."""
    scale = scale_values(scales).double().repeat_interleave(BLOCK, dim=1)
    return codes.double() * scale


def decode_mxfp4(packed, scales, columns):
    """The values packed E2M1 bytes ``[rows, columns / 2]`` stand for under
    linear ``scales``, in FP64.
    """  # noqa: D205
    rows = packed.shape[0]
    narrow = packed.view(torch.uint8).to(torch.int64).reshape(rows, -1)
    codes = torch.zeros(rows, columns, dtype=torch.int64, device=packed.device)
    codes[:, 0::2] = narrow & 0xF
    codes[:, 1::2] = (narrow >> 4) & 0xF
    values = E2M1_VALUES.to(packed.device)[codes].double()
    scale = scale_values(scales).double().repeat_interleave(BLOCK, dim=1)
    return values * scale


def packed_shape(rows, columns):
    """The ``[groups, padded rows]`` words of a packed scale table."""
    groups = columns // (BLOCK * WORD_SCALES)
    return groups, -(-rows // WORD_ROWS) * WORD_ROWS


def packed_scales(scales):
    """Linear scale bytes ``[rows, blocks]`` as the packed MN-major ``u32``
    words ``[blocks / 4, padded rows]``; the padding rows are zero.
    """  # noqa: D205
    rows, blocks = scales.shape
    groups, padded = packed_shape(rows, blocks * BLOCK)
    # A word's bytes in memory order are its scales in block order (the device
    # is little-endian), so the table is the bytes `[groups, padded, 4]` viewed
    # as words.
    words = torch.zeros(groups, padded, WORD_SCALES, dtype=torch.uint8)
    grouped = scales.cpu().reshape(rows, groups, WORD_SCALES)
    words[:, :rows, :] = grouped.permute(1, 0, 2)
    return words.view(torch.int32).reshape(groups, padded).contiguous()


def unpacked_scales(words, rows):
    """Packed MN-major words ``[groups, padded rows]`` as the linear scale
    bytes ``[rows, groups · 4]`` they hold.
    """  # noqa: D205
    groups, padded = words.shape
    bytes_ = words.cpu().contiguous().view(torch.uint8)
    bytes_ = bytes_.reshape(groups, padded, WORD_SCALES)[:, :rows, :]
    return bytes_.permute(1, 0, 2).reshape(rows, groups * WORD_SCALES)


def swizzled_offsets(rows, blocks):
    """Flat offsets of the 128x4 atom of every ``[row, block]`` of a
    ``[rows, blocks]`` scale table, the blocks padded to four.
    """  # noqa: D205
    padded_blocks = -(-blocks // 4) * 4
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    block = torch.arange(blocks, dtype=torch.int64)[None, :]
    return (
        (row // 128) * (padded_blocks * 128)
        + (block // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (block % 4)
    )


class GemmCall(ctypes.Structure):
    """``uniserve_mxfp8_gemm_call`` (``ops::mx``'s ``Mxfp8Gemm`` call), field
    for field.
    """  # noqa: D205

    _fields_ = [
        ("input", ctypes.c_void_p),
        ("input_scales", ctypes.c_void_p),
        ("weight", ctypes.c_void_p),
        ("weight_scales", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("m", ctypes.c_int64),
        ("n", ctypes.c_int64),
        ("k", ctypes.c_int64),
        ("input_scale_stride", ctypes.c_int64),
        ("weight_scale_stride", ctypes.c_int64),
        ("multiprocessors", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


#: The output types' subnormal spacing: the absolute part of a bound that
#: ends in a rounding into them.
UNDERFLOW = {"bf16": 2.0**-133, "f32": 2.0**-149}


def dot_tolerance(k, output):
    """``Mxfp8Gemm::tolerance``: ``Tolerance::dot(k, f32, output)``.

    The codes' products under power-of-two scales are exact in FP32, so a
    product element is within ``atol + rtol · peak`` of the FP64 product of the
    decoded operands, ``peak`` the output's largest magnitude (a dot product's
    element can cancel below its own terms). Answers ``(atol, rtol)``.
    """
    return (
        UNDERFLOW[output],
        2.0 * gamma(k, UNIT["f32"]) * k**0.5 + UNIT[output],
    )
