"""FP8 paged K/V scatter: the Triton source of the ``paged_kv_write`` operator's
``fp8`` backend.

An FP8 KV page holds E4M3 codes under one scale per page: ``value = code *
scale``. One program scatters a tile-wide slice of one token's flattened
``[heads, dim]`` row into its physical slot, encoding each value against its
block's scale and clamping to the E4M3 finite range (±448), and marks the
block initialized when the caller tracks initialization.

The layout is the one of the operator's ``triton`` backend (the float scatter
UniServe wrote): a page holds ``block_size`` tokens' rows of ``heads * dim``
values, slots address physical tokens, ``-1`` skips a token. The encoding is
this package's addition: an FP8 cache is only readable by an attention kernel
that decodes with the same per-page scale, which the ``fp8`` backend of
``paged_attention`` does.
"""

import triton
import triton.language as tl


@triton.jit
def _paged_kv_write_fp8_kernel(
    k_cache_ptr,
    v_cache_ptr,
    slots_ptr,
    k_src_ptr,
    v_src_ptr,
    k_scale_ptr,
    v_scale_ptr,
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
    """Encode and scatter borrowed key and value rows into their slots.

    ``num_pages`` and ``block_size`` are the cache's block count and tokens per
    block; ``tile`` is this kernel's column width. Grid: (token rows,
    ceil(row_width / tile)).
    """
    row = tl.program_id(0)
    columns = tl.program_id(1) * tile + tl.arange(0, tile)
    location = tl.load(slots_ptr + row)
    # -1 excludes the token from both payload and initialization state.
    cache_row = location
    persists = location >= 0
    valid_address = location < num_pages * block_size
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

    # A skipped token addresses no page; clamping its block to zero keeps the
    # scale load in bounds while the mask keeps the store off.
    page = tl.where(persists, cache_row // block_size, 0)
    k_scale = tl.load(k_scale_ptr + page)
    v_scale = tl.load(v_scale_ptr + page)
    k = tl.load(k_src_ptr + k_source_offsets, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_src_ptr + v_source_offsets, mask=mask, other=0.0).to(tl.float32)
    # Codes are `value / scale`, clamped to the E4M3 finite range; the store
    # rounds to nearest even in the cache's element type.
    k_code = tl.maximum(-448.0, tl.minimum(448.0, k / k_scale))
    v_code = tl.maximum(-448.0, tl.minimum(448.0, v / v_scale))
    tl.store(k_cache_ptr + cache_offsets, k_code, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, v_code, mask=mask)

    # A ``None`` flag pointer is a compile-time constant, so this branch
    # disappears when the caller tracks no initialization state.
    if k_initialized_ptr is not None:
        if tl.program_id(1) == 0:
            block = cache_row // block_size
            tl.store(
                k_initialized_ptr + block, 1, mask=persists & valid_address
            )
            tl.store(
                v_initialized_ptr + block, 1, mask=persists & valid_address
            )
