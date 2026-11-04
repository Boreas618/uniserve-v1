"""Reading selected blocks of a state bank into a contiguous snapshot.

A state bank stores each field block-major: axis 0 selects a block and the
trailing axes are the block's own rows, at any strides. This program is the
prefix cache's block copy, first half: ``snapshot[row, :] = field[blocks[row],
:]``, so that a later `cache_block_scatter` can write the same blocks into
another bank without a host read between the two launches.

The address arithmetic is the reference's (``prefix_cache/_copy.py``): the
trailing axes are unrolled at compile time, and one program walks one
``BLOCK``-element tile of the flattened block, so a field of any rank and any
strides is one program shape. A negative index — the inactive pair's
placeholder — and an index past the bank write nothing, which is what keeps an
unvalidated device vector from reaching outside the allocation.
"""

import triton
import triton.language as tl


@triton.jit
def _snapshot_kernel(
    field,  # *dtype: the block-major storage of the bank
    blocks,  # *i64: one block index per row of the snapshot
    snapshot,  # *dtype: [count, width], contiguous, receives the blocks
    index_stride: tl.constexpr,  # elements between two entries of `blocks`
    shape: tl.constexpr,  # the bank's block-major shape, one block per axis 0
    strides: tl.constexpr,  # the bank's element strides
    width: tl.constexpr,  # elements in one flattened block
    BLOCK: tl.constexpr,  # columns one program reads
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

    # `shape[0]` is the bank's block count: an index outside it is as inert as a
    # negative one, and the snapshot row then keeps what it held.
    valid = (block >= 0) & (block < shape[0]) & (columns < width)
    value = tl.load(field + address, valid, 0)
    tl.store(snapshot + row * width + columns, value, valid)
