"""Decoding a span of imported K/V pages into the conversion scratch.

An importing rank receives another rank's cache as raw keys and values of the
source's own element type, and, when the source stores FP8, one FP32 scale per
source page and head group. This program is the prefix cache's first conversion
launch: it decodes one field of the span into the workspace's FP32 block, from
which `cache_copy_region` writes the destination's pages with the destination's
encoding. Between them, an imported value rounds and scales exactly as a
locally written one does.

## The scratch layout

The workspace holds ``20 E + 8 S`` bytes, where ``E`` is the elements of the
largest group page and ``S`` the scale entries of the field this launch
converts:

```text
[0, 8E)          raw keys
[8E, 16E)        raw values
[16E, 20E)       the FP32 block this program writes
[20E, 20E + 8S)  the source pages' FP32 scales
```

The program takes the scratch base and both extents and finds its own
sub-addresses, so no caller restates the layout. Both raw areas hold the span
``[count, layers, heads, head_dim]`` of the source's stored elements, and the
FP32 block has that same structure.

## The scale of a value

A source page holds ``key / scale``: the stored value times the page's scale,
in FP32, is the value the source rank wrote. That product then rounds through
the source's logical type (``compute``), which is what makes an imported value
equal to a locally written one rather than one rounding better. A dense source
carries no scales and is widened straight through the same rounding.

One program handles one (token, layer) and up to ``head_dim`` heads of it: the
grid's second axis tiles the heads by ``head_dim``, which is one tile for every
head count a K/V group has.
"""

import triton
import triton.language as tl


@triton.jit
def _page_conversion_kernel(
    scratch,  # *u8: the conversion workspace
    elements,  # u64: E, elements of the largest group page
    scales,  # u64: S, scale entries of this field's largest group page
    count,  # u64: tokens of the span
    source_offset,  # u64: token offset of the span's first token in its first page
    source_block_size,  # u32: tokens per page of the source
    scale_heads,  # u32: heads of one source scale group, 0 when dense
    pages,  # u64: source pages the span covers
    FIELD: tl.constexpr,  # 0 for keys, 1 for values
    SOURCE: tl.constexpr,  # the stored element type
    COMPUTE: tl.constexpr,  # the source's logical element type
    HEAD_DIM: tl.constexpr,  # elements per head, and heads one program covers
    HEADS: tl.constexpr,  # heads of this rank's K/V group
    LAYERS: tl.constexpr,  # layers of this rank's K/V group
    SCALED: tl.constexpr,  # whether the source carries per-page scales
    SOURCE_BYTES: tl.constexpr,  # bytes of one stored element
):
    program = tl.program_id(0)
    tile = tl.program_id(1)
    token = program // LAYERS
    layer = program % LAYERS

    # The span's row of this (token, layer), in stored elements: the raw area of
    # the field, then the row's own offset. Both the raw areas and the FP32
    # block are indexed the same way.
    row = (token.to(tl.int64) * LAYERS + layer) * HEADS
    # The field's raw area starts at the workspace's base for keys and past E
    # eight-byte units for values; both in stored elements.
    if FIELD == 1:
        raw_base = 8 * elements.to(tl.int64) // SOURCE_BYTES + row * HEAD_DIM
    else:
        raw_base = row * HEAD_DIM
    raw = scratch.to(tl.pointer_type(SOURCE), bitcast=True) + raw_base
    decoded = scratch.to(tl.pointer_type(tl.float32), bitcast=True)
    decoded = decoded + 4 * elements.to(tl.int64) + row * HEAD_DIM

    # The grid's second axis and the head tile are integers; the signedness of
    # `tl.program_id` is an implementation detail of Triton's, so the head
    # arithmetic is done in one signed type from here on.
    first_head = (tile * HEAD_DIM).to(tl.int64)
    heads_here = HEADS - first_head
    columns = tl.arange(0, HEAD_DIM)

    # The scale row of the token's source page, counted from the span's first
    # page: the caller packs one row per page the span touches, starting with
    # the page `source_offset` falls in, so a span that starts mid-page does not
    # skip that many rows. One scale per head group, one group row per layer.
    page_size = source_block_size.to(tl.int64)
    page = (source_offset.to(tl.int64) + token.to(tl.int64)) // page_size
    page = page - source_offset.to(tl.int64) // page_size
    groups = (HEADS + scale_heads - 1) // scale_heads if SCALED else 1
    scale_base = (page * LAYERS + layer.to(tl.int64)) * groups
    scale_pointer = scratch.to(tl.pointer_type(tl.float32), bitcast=True)
    scale_pointer = scale_pointer + 5 * elements.to(tl.int64)

    for step in range(heads_here):
        head = first_head + step
        offset = head * HEAD_DIM + columns
        value = tl.load(raw + offset).to(tl.float32)
        if SCALED:
            entry = scale_base + head // scale_heads.to(tl.int64)
            value = value * tl.load(scale_pointer + entry)
        # The logical type's rounding, then the FP32 block the next launch reads.
        tl.store(decoded + offset, value.to(COMPUTE).to(tl.float32))
