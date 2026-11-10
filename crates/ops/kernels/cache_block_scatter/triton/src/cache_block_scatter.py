"""Writing a contiguous snapshot into selected blocks of a state bank.

The second half of the prefix cache's block copy: ``field[blocks[row], :] =
snapshot[row, :]``. The address arithmetic is the reference kernel's
(``prefix_cache/_copy.py::_scatter_kernel``), so a field of any rank and any
strides is one program shape and the snapshot is read contiguously.

A negative index — the inactive pair's placeholder — and an index past the bank
write nothing, which is what keeps an unvalidated device vector from reaching
outside the allocation.
"""

import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(
    field,  # *dtype: the block-major storage of the bank, overwritten
    blocks,  # *i64: the block each row of the snapshot goes to
    snapshot,  # *dtype: [count, width], contiguous, holds the blocks
    index_stride: tl.constexpr,  # elements between two entries of `blocks`
    shape: tl.constexpr,  # the bank's block-major shape, one block per axis 0
    strides: tl.constexpr,  # the bank's element strides
    width: tl.constexpr,  # elements in one flattened block
    BLOCK: tl.constexpr,  # columns one program writes
):
    row = tl.program_id(0)
    block = tl.load(blocks + row * index_stride)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)

    # The flattened column's offsets along the block's trailing axes, weighted
    # by the field's strides. `block` is i64, so the sum is a byte-offset-sized
    # address and a bank of any size stays addressable.
    remaining = columns
    address = block * strides[0] + tl.zeros((BLOCK,), tl.int64)
    for axis in tl.static_range(len(shape) - 1, 0, -1):
        address += (remaining % shape[axis]) * strides[axis]
        remaining //= shape[axis]

    valid = (block >= 0) & (block < shape[0]) & (columns < width)
    value = tl.load(snapshot + row * width + columns, valid, 0)
    tl.store(field + address, value, valid)
