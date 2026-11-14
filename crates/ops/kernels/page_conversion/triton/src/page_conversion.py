"""Decoding one staged host page of K/V into the conversion scratch.

A rank that loads a page stored in another representation (another cache dtype, or FP8 with
another head grouping of its scales) stages that page's bytes in its conversion scratch, in the
host page encoding of the buffer pool's host regions, which the Storage Server holds. This program
is the conversion's first launch: it decodes one field of the staged page into the workspace's FP32
block, from which `cache_copy_region` writes the destination's units with the destination's
encoding. Between them, a loaded value rounds and scales exactly as a locally written one does.

## The source: the host page encoding

A host K/V page is layer-major (`crates/engine/src/memory/cache/host.rs`): keys
``[layers, block_size, heads, head_dim]``, then values in the same shape, then, for an FP8 page,
the scales ``[2, layers]`` as FP32, one per field and layer. A page staged from several source parts
(another tensor-parallel split, one host part per KV rank) carries one scale row per source part's
head range, so the scales generalise to ``[2, layers, groups]`` with
``groups = ceil(heads / scale_heads)``; a page of one source part has ``groups = 1`` and is the host
page's own ``[2, layers]``.

## The scratch layout

The workspace holds ``20 E + 8 S`` bytes, where ``E`` is the elements of one field of the page
(``layers · block_size · heads · head_dim``) and ``S`` the page's scale entries
(``2 · layers · groups``, zero for a dense page):

```text
[0, 8E)          raw keys      [layers, block_size, heads, head_dim], the source's element type
[8E, 16E)        raw values    the same shape
[16E, 20E)       the FP32 block this program writes, the same shape
[20E, 20E + 8S)  the page's FP32 scales [2, layers, groups]
```

Each area starts where the staged page's field lands with one copy: the keys of the page at the
base, its values eight ``E`` bytes along, its scale tail at ``20 E``. The program takes the scratch
base and both extents and finds its own sub-addresses, so no caller restates the layout.

## The scale of a value

A scaled page holds ``key / scale``: the stored value times its (field, layer, group) scale, in
FP32, is the value the source rank wrote. That product then rounds through the source's logical type
(``compute``), which is what makes a loaded value equal to a locally written one rather than one
rounding better. A dense page carries no scales and is widened straight through the same rounding.

One program handles one (layer, token) row of the field: every head of it.
"""

import triton
import triton.language as tl


@triton.jit
def _page_conversion_kernel(
    scratch,  # *u8: the conversion workspace
    elements,  # u64: E, elements of one field of the page
    scales,  # u64: S, scale entries of the page, 0 when dense
    block_size,  # u32: tokens per page
    scale_heads,  # u32: heads of one scale group, 0 when dense
    FIELD: tl.constexpr,  # 0 for keys, 1 for values
    SOURCE: tl.constexpr,  # the stored element type
    COMPUTE: tl.constexpr,  # the source's logical element type
    HEAD_DIM: tl.constexpr,  # elements per head
    HEADS: tl.constexpr,  # heads of this rank's K/V group
    LAYERS: tl.constexpr,  # layers of this rank's K/V group
    SCALED: tl.constexpr,  # whether the page carries scales
    SOURCE_BYTES: tl.constexpr,  # bytes of one stored element
):
    program = tl.program_id(0).to(tl.int64)
    tokens = block_size.to(tl.int64)
    layer = program // tokens

    # The page's row of this (layer, token), in elements: the program index is that row's own index
    # in the layer-major page, and the raw areas and the FP32 block are indexed the same way.
    row = program * HEADS
    # The field's raw area starts at the workspace's base for keys and past E eight-byte units for
    # values; both in stored elements.
    if FIELD == 1:
        raw_base = 8 * elements.to(tl.int64) // SOURCE_BYTES + row * HEAD_DIM
    else:
        raw_base = row * HEAD_DIM
    raw = scratch.to(tl.pointer_type(SOURCE), bitcast=True) + raw_base
    decoded = scratch.to(tl.pointer_type(tl.float32), bitcast=True)
    decoded = decoded + 4 * elements.to(tl.int64) + row * HEAD_DIM
    columns = tl.arange(0, HEAD_DIM)

    # The field's scale row of this layer: `[2, layers, groups]` FP32 past 20E bytes, which is 5E
    # FP32 entries along.
    groups = (HEADS + scale_heads - 1) // scale_heads if SCALED else 1
    scale_pointer = scratch.to(tl.pointer_type(tl.float32), bitcast=True)
    scale_pointer = scale_pointer + 5 * elements.to(tl.int64)
    scale_pointer = scale_pointer + (FIELD * LAYERS + layer) * groups

    for head in range(HEADS):
        offset = head * HEAD_DIM + columns
        value = tl.load(raw + offset).to(tl.float32)
        if SCALED:
            value = value * tl.load(scale_pointer + head // scale_heads)
        # The logical type's rounding, then the FP32 block the next launch reads.
        tl.store(decoded + offset, value.to(COMPUTE).to(tl.float32))
