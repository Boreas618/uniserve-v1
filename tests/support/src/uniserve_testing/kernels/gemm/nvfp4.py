"""The NVFP4 operands' encodings and scale layouts, as the checks read them.

An NVFP4 operand is packed E2M1 codes, two a byte with the lower column in the
low nibble, and one E4M3 scale per sixteen elements of the contraction. The
scales are laid out as ``ScaleLayout`` in ``crates/ops/src/fp4.rs`` defines it
for a matrix of ``rows`` rows and ``scales = columns / 16`` scale columns:

- ``linear``: ``[rows, scales]`` row major, offset ``row · scales + scale``;
- ``swizzled_128x4``: rows padded to 128 and scale columns to 4, in the atom
  layout the block-scaled ``tcgen05`` MMA reads: the padded matrix, cut into
  blocks of 128 rows by 4 scale columns, the 128 rows split into four groups
  of 32, and within a block ``lane · 16 + group · 4 + column``.

``test_nvfp4_gemm.py`` proves the swizzle is the atom layout ``cute`` builds,
and ``test_nvfp4_quantize.py`` that both layouts read one logical matrix
alike. Both call the ``nvfp4_gemm`` module's entry with :class:`GemmCall`,
which ``test_nvfp4_gemm.py`` holds to the generated header.
"""

import ctypes

import torch


class GemmCall(ctypes.Structure):
    """``uniserve_nvfp4_gemm_call`` (``ops::fp4``'s ``Nvfp4Gemm`` call), field
    for field.
    """  # noqa: D205

    _fields_ = [
        ("input", ctypes.c_void_p),
        ("weight", ctypes.c_void_p),
        ("input_scales", ctypes.c_void_p),
        ("weight_scales", ctypes.c_void_p),
        ("alpha", ctypes.c_void_p),
        ("bias", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("m", ctypes.c_int64),
        ("n", ctypes.c_int64),
        ("k", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


#: E2M1's eight magnitudes, by the code's low three bits; bit 3 is the sign.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: Elements one E4M3 block scale covers (``fp4.rs``'s ``BLOCK_ELEMENTS``).
BLOCK_ELEMENTS = 16

#: The swizzled layout's row block, its four groups of rows, and its scale
#: columns.
SWIZZLE_ROWS = 128
SWIZZLE_GROUPS = 4
SWIZZLE_ROWS_PER_GROUP = SWIZZLE_ROWS // SWIZZLE_GROUPS
SWIZZLE_COLUMNS = 4


def padded_shape(rows, scales):
    """The shape a swizzled scale matrix occupies: rows to 128, scale columns
    to 4.
    """  # noqa: D205
    return (
        -(-rows // SWIZZLE_ROWS) * SWIZZLE_ROWS,
        -(-scales // SWIZZLE_COLUMNS) * SWIZZLE_COLUMNS,
    )


def linear_offsets(rows, scales):
    """Flat offsets of the ``linear`` layout: ``[rows, scales]`` row major."""
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    scale = torch.arange(scales, dtype=torch.int64)[None, :]
    return row * scales + scale


def swizzled_offsets(rows, scales):
    """Flat offsets of the 128x4 swizzle of every ``[row, scale]`` of a
    ``[rows, scales]`` scale matrix.
    """  # noqa: D205
    _, padded_scales = padded_shape(rows, scales)
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    scale = torch.arange(scales, dtype=torch.int64)[None, :]
    within = row % SWIZZLE_ROWS
    group = within // SWIZZLE_ROWS_PER_GROUP
    lane = within % SWIZZLE_ROWS_PER_GROUP
    return (
        (row // SWIZZLE_ROWS) * (padded_scales * SWIZZLE_ROWS)
        + (scale // SWIZZLE_COLUMNS) * (SWIZZLE_ROWS * SWIZZLE_COLUMNS)
        + lane * (SWIZZLE_GROUPS * SWIZZLE_COLUMNS)
        + group * SWIZZLE_COLUMNS
        + (scale % SWIZZLE_COLUMNS)
    )


def unpacked(packed, rows, columns):
    """A ``[rows, columns // 2]`` matrix of packed bytes as its
    ``[rows, columns]`` E2M1 codes, the low nibble first.
    """  # noqa: D205
    narrow = packed.view(torch.uint8).to(torch.int16).reshape(rows, -1)
    codes = torch.zeros(rows, columns, dtype=torch.int64, device=packed.device)
    codes[:, 0::2] = (narrow & 0xF).long()
    codes[:, 1::2] = ((narrow >> 4) & 0xF).long()
    return codes
