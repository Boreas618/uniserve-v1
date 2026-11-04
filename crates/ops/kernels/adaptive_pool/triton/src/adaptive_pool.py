"""Adaptive width pooling over an already head-averaged row: the Triton program.

The audio encoder's attention projection ends with two reductions that land on the
same output element, so this program performs both in one pass: `out[r, i]` is the
mean over every head and over the input positions `start(i) .. end(i)`, where
`start(i) = floor(i · width / output)` and `end(i) = ceil((i + 1) · width / output)`.
The two are summed together because splitting them would need an intermediate row
the block would have to reserve and write; the fused sum differs from the
reference's two-step mean only in the order of one FP32 accumulation, which the
operator's tolerance states.

A window is never empty: when `output > width` every window holds one position, and
some positions are averaged alone. `floor(i·width/output)` and
`ceil((i+1)·width/output)` give an empty window there, so `end` is raised to
`start + 1`, which is what adaptive pooling does with more outputs than inputs.

One program produces `BLOCK` consecutive output elements of `[rows, output]`. A
window is walked head by head, so the loads of one program are strided slices of the
row — `rows · heads · width` is contiguous, and the whole reduction is FP32 whatever
the element type, rounded once on the store.
"""

import triton
import triton.language as tl


@triton.jit
def _adaptive_pool_kernel(
    x,
    out,
    rows,
    heads,
    width,
    output,
    pdl,
    HEADS: tl.constexpr,  # noqa: N803
    WIDTH: tl.constexpr,  # noqa: N803
    OUTPUT: tl.constexpr,  # noqa: N803
    WINDOW: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of pooled output elements.

    Grid: ``ceil(rows · output / BLOCK)`` programs. ``WINDOW`` is the longest window
    the descriptor's width and output give (`ceil(width / output)`), the bound the
    unrolled loop below covers; a shorter window is masked out term by term.
    """
    program = tl.program_id(0)
    index = (program * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    total = rows * output
    live = index < total

    row = index // output
    position = index % output
    start = (position * width) // output
    end = ((position + 1) * width + output - 1) // output
    # A window is never empty: with more outputs than inputs, `end` would equal
    # `start` and the window would average nothing.
    end = tl.maximum(end, start + 1)
    span = (end - start).to(tl.float32)

    accumulator = tl.zeros([BLOCK], dtype=tl.float32)
    for head in tl.static_range(HEADS):
        base = (row * HEADS + head) * WIDTH + start
        for step in tl.static_range(WINDOW):
            inside = live & (step < (end - start))
            accumulator += tl.load(x + base + step, mask=inside, other=0.0).to(
                tl.float32
            )

    # One division: `heads · window` terms summed in FP32, rounded once into the
    # output's element type.
    pooled = accumulator / (span * HEADS)
    tl.store(out + index, pooled.to(out.dtype.element_ty), mask=live)
