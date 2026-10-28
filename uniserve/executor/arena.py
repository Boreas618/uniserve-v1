"""Byte layout rules of the device storage the executors allocate.

An executor lays its columns out in one arena per runner, each column at a
:data:`COLUMN_ALIGNMENT` boundary, and allocates request state as separate
tensors, which torch's caching allocator rounds up to
:data:`ALLOCATION_ROUNDING` bytes. Capacity planning reads the executors'
byte counts, so both rules are stated once here.
"""

from __future__ import annotations

__all__ = ["ALLOCATION_ROUNDING", "COLUMN_ALIGNMENT", "aligned"]

# Byte alignment of each column in an arena: every column's element view
# starts on a boundary vectorized kernels read whole.
COLUMN_ALIGNMENT = 256

# Bytes torch's caching allocator rounds each separately allocated tensor
# up to.
ALLOCATION_ROUNDING = 512


def aligned(size: int, alignment: int = COLUMN_ALIGNMENT) -> int:
    """``size`` bytes rounded up to a multiple of ``alignment``."""
    return -(-size // alignment) * alignment
