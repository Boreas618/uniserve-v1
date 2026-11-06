"""The ``copy_block`` operator's Triton source: one logical block into another view of it.

```text
out[i0, i1, i2, i3] = source[i0, i1, i2, i3]
```

Every index of the block is decomposed once and addressed through **each side's own stride vector**,
so the same program copies, places a block inside a larger buffer, and permutes axes — a permutation
is the source's strides permuted, which is why the operator has no mode and no permutation field.

**Nothing about the geometry is compiled.** The four extents and both stride vectors are launch
words, so a clip whose tile shape changes between requests launches the same module; that is the whole
reason `Op::Args` carries the strides instead of the descriptor.

**The destination is not assumed to have a unit innermost stride.** The operator deliberately has no
unit-stride rule, so the decomposition below is the only thing that may assume anything, and it
assumes nothing: each axis contributes `coordinate · stride` on its own side.

Every index is 64-bit: a clip's latent block and a raster region both exceed 2**31 elements at the
sizes this serves, and a 32-bit product would wrap into another tile's rows with no shape to complain
about.
"""

import triton
import triton.language as tl


@triton.jit
def _copy_block_kernel(
    source_ptr,
    out_ptr,
    e0,
    e1,
    e2,
    e3,
    s0,
    s1,
    s2,
    s3,
    d0,
    d1,
    d2,
    d3,
    total,
    block,
    pdl,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    """One element a lane, addressed through both sides' strides.

    The decomposition runs from the last axis, because that is the axis whose extent divides first
    when a flat index is turned back into coordinates; the extents are the destination's axis order
    and the source's strides are given in that same order, which is where a permutation lives.
    """
    # The extents arrive unsigned and the index is signed: one signedness for the decomposition, or
    # Triton refuses the modulo rather than guessing which one was meant.
    e0 = e0.to(tl.int64)
    e1 = e1.to(tl.int64)
    e2 = e2.to(tl.int64)
    e3 = e3.to(tl.int64)
    program = tl.program_id(0).to(tl.int64)

    # A wide block's rows are unit-stride and aligned on both sides, which the module states, so the
    # compiler moves whole vectors, and nothing below this branch is compiled into its module: the
    # general path's four 64-bit divisions a lane are what kept a copy at a quarter of the memory's
    # speed through the registers they hold.
    if WIDE:
        blocks = (e3 + BLOCK - 1) // BLOCK
        row = program // blocks
        lanes = (program % blocks) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        inside = lanes < e3
        c2 = row % e2
        rest = row // e2
        c1 = rest % e1
        c0 = rest // e1
        value = tl.load(source_ptr + (c0 * s0 + c1 * s1 + c2 * s2) + lanes, mask=inside, other=0)
        tl.store(out_ptr + (c0 * d0 + c1 * d1 + c2 * d2) + lanes, value, mask=inside)
        return

    # A last axis at least one block wide is walked a row at a time: the three leading coordinates
    # are one scalar decomposition per program, where the general path below divides four times per
    # lane in 64 bits, which is what a copy of a wide block spends its time on.
    if e3 >= BLOCK:
        blocks = (e3 + BLOCK - 1) // BLOCK
        row = program // blocks
        lanes = (program % blocks) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        inside = lanes < e3
        c2 = row % e2
        rest = row // e2
        c1 = rest % e1
        c0 = rest // e1
        source_row = c0 * s0 + c1 * s1 + c2 * s2
        out_row = c0 * d0 + c1 * d1 + c2 * d2
        value = tl.load(source_ptr + source_row + lanes * s3, mask=inside, other=0)
        tl.store(out_ptr + out_row + lanes * d3, value, mask=inside)
        return

    index = program * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    live = index < total

    # Unrolled rather than looped over a list of tuples: Triton's AST transformer reads a Python
    # ``for`` over a literal list as a loop over tensors and refuses it.
    rest = index
    source_offset = tl.zeros_like(index)
    out_offset = tl.zeros_like(index)

    coordinate = rest % e3
    rest = rest // e3
    source_offset += coordinate * s3
    out_offset += coordinate * d3

    coordinate = rest % e2
    rest = rest // e2
    source_offset += coordinate * s2
    out_offset += coordinate * d2

    coordinate = rest % e1
    rest = rest // e1
    source_offset += coordinate * s1
    out_offset += coordinate * d1

    coordinate = rest % e0
    source_offset += coordinate * s0
    out_offset += coordinate * d0

    value = tl.load(source_ptr + source_offset, mask=live, other=0)
    tl.store(out_ptr + out_offset, value, mask=live)
