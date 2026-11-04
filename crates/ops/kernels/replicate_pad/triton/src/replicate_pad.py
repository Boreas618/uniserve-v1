"""ATen's replicate pad: the Triton source of the ``replicate_pad`` operator.

One program covers one block of the output, one element per lane. An output
element's index on each padded axis maps to the input index `o - left`, clamped
to `[0, in_extent - 1]`, which repeats the edge value outward; a negative
`left` or `right` crops, and the input index then still lies inside the input
because the output extent shrank. The trailing axes are padded from the
innermost outward (ATen's order), and `AXES` is the descriptor's count: the
launch passes three slots and zeroes the unused ones, which the kernel ignores.

The map moves elements unchanged, so the result is exact.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(
    do_not_specialize=[
        "leading",
        "in0",
        "out0",
        "left0",
        "in1",
        "out1",
        "left1",
        "in2",
        "out2",
        "left2",
    ]
)
def _replicate_pad_kernel(
    x_ptr,
    out_ptr,
    leading,
    in0,
    out0,
    left0,
    right0,
    in1,
    out1,
    left1,
    right1,
    in2,
    out2,
    left2,
    right2,
    pdl,
    BLOCK: tl.constexpr,
    AXES: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Copy the padded output from the input, one element per lane.

    ``leading`` is the product of the axes before the padded ones, and each of
    the three slots carries `in_extent`, `out_extent`, `left` and `right` for one
    padded axis. `right` is not read: the output extent already carries it.
    """
    _pdl_prologue(PDL)

    # The launch passes unsigned extents and signed pads; every index in the
    # map is signed so the clamps and the remainder agree in type.
    leading = leading.to(tl.int64)
    in0, out0, left0 = in0.to(tl.int64), out0.to(tl.int64), left0.to(tl.int64)
    in1, out1, left1 = in1.to(tl.int64), out1.to(tl.int64), left1.to(tl.int64)
    in2, out2, left2 = in2.to(tl.int64), out2.to(tl.int64), left2.to(tl.int64)

    total = leading * out0
    if AXES >= 2:
        total = total * out1
    if AXES == 3:
        total = total * out2

    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < total

    o0 = offsets % out0
    rest = offsets // out0
    i0 = tl.minimum(tl.maximum(o0 - left0, 0), in0 - 1)

    i1 = tl.zeros_like(offsets)
    inner1 = 1
    if AXES >= 2:
        o1 = rest % out1
        rest = rest // out1
        i1 = tl.minimum(tl.maximum(o1 - left1, 0), in1 - 1)
        inner1 = in1

    i2 = tl.zeros_like(offsets)
    inner2 = 1
    if AXES == 3:
        o2 = rest % out2
        rest = rest // out2
        i2 = tl.minimum(tl.maximum(o2 - left2, 0), in2 - 1)
        inner2 = in2

    source = ((rest * inner2 + i2) * inner1 + i1) * in0 + i0
    value = tl.load(x_ptr + source, mask=mask)
    tl.store(out_ptr + offsets, value, mask=mask)
