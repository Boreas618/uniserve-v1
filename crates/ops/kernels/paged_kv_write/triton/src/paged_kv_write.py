"""Paged K/V scatter: the Triton source of the ``paged_kv_write`` operator.

Caches store ``[blocks, tokens, heads, dim]`` rows; writes address physical
token slots ``block * block_size + token``, and ``-1`` skips a token. One
program scatters a tile-wide slice of one token's flattened ``[heads, dim]``
row and, when the caller tracks initialization, the first column program of a
row marks the block it landed in.

The kernel body is the one UniServe scattered paged K/V with
(``uniserve_kernels/src/uniserve_kernels/cache.py``), unchanged except that it
imports nothing but Triton: the operand checks, the device and torch types
stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit
def _paged_kv_write_kernel(
    k_cache_ptr,
    v_cache_ptr,
    slots_ptr,
    k_src_ptr,
    v_src_ptr,
    k_initialized_ptr,
    v_initialized_ptr,
    num_pages: tl.constexpr,
    block_size: tl.constexpr,
    row_width: tl.constexpr,
    head_dim: tl.constexpr,
    k_strides: tl.constexpr,
    v_strides: tl.constexpr,
    tile: tl.constexpr,
):
    """Scatter borrowed key and value rows into physical token slots.

    ``num_pages`` and ``block_size`` are the cache's block count and tokens per
    block; ``tile`` is this kernel's column width. Shapes and source strides
    are constexpr, so each distinct cache shape or source stride tuple compiles
    its own specialization.

    Grid: (token rows, ceil(row_width / tile)). Each program scatters a
    ``tile``-wide slice of one token's flattened [heads, head_dim] row. The
    caches are contiguous, so a slot indexes a [num_pages * block_size,
    row_width] view directly.
    """
    row = tl.program_id(0)
    columns = tl.program_id(1) * tile + tl.arange(0, tile)
    location = tl.load(slots_ptr + row)
    # -1 excludes the token from both payload and initialization state.
    # Physical block zero is an ordinary, writable numerical block.
    cache_row = location
    persists = location >= 0
    valid_address = location < num_pages * block_size
    # Triton compiles device asserts only in debug mode, so this package builds
    # every module with `debug=True`.
    tl.device_assert(location >= -1, "paged KV write index below -1")
    tl.device_assert(
        (~persists) | valid_address, "paged KV write index out of bounds"
    )

    # Decompose flat row columns into (head, dim) for strided source reads.
    heads = columns // head_dim
    dimensions = columns % head_dim
    k_source_offsets = (
        row * k_strides[0] + heads * k_strides[1] + dimensions * k_strides[2]
    )
    v_source_offsets = (
        row * v_strides[0] + heads * v_strides[1] + dimensions * v_strides[2]
    )
    cache_offsets = cache_row * row_width + columns
    mask = persists & valid_address & (columns < row_width)
    k = tl.load(k_src_ptr + k_source_offsets, mask=mask, other=0.0)
    v = tl.load(v_src_ptr + v_source_offsets, mask=mask, other=0.0)
    tl.store(k_cache_ptr + cache_offsets, k, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, v, mask=mask)
    # A ``None`` flag pointer is a compile-time constant, so this branch
    # disappears when the caller tracks no initialization state.
    if k_initialized_ptr is not None:
        # Only the first column program of each token row stores flags, so a
        # row commits its block's flag once regardless of how many programs
        # span the head width. Rows sharing a block all store the same true
        # value. Payload and flags are written by one launch, so later work on
        # the same stream observes both.
        if tl.program_id(1) == 0:
            block = cache_row // block_size
            tl.store(
                k_initialized_ptr + block, 1, mask=persists & valid_address
            )
            tl.store(
                v_initialized_ptr + block, 1, mask=persists & valid_address
            )
