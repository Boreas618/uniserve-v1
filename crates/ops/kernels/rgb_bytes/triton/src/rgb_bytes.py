"""Normalized float raster to packed 8-bit RGB: the Triton program.

One program converts ``BLOCK`` consecutive bytes of the output. The output is
``[frames, rows, columns, channels]`` and the program walks it as one flat run, so
the index arithmetic below is the only layout knowledge in the kernel: the channel
is the innermost axis of both sides, the columns next, and the frames outermost. The
four read-side and four write-side axes between them appear as the six strides the
operator hands over, which is what lets a decoder write one tile into a sub-region of
a clip raster without a pass and a copy.

The arithmetic is the operator's, in this order and in FP32 whatever the input's
element type: de-normalize (multiply by the channel's standard deviation, add its
mean), scale by 255, clamp into ``[0, 255]``, then round **half to even**. The
rounding is the visible part of the artifact — a tie must go to the even neighbour,
which is what ``numpy.rint`` and Rust's ``round_ties_even`` do — so the check script
compares bytes and the ties are in its samples.

The rounding is written as an addition and a subtraction of 2^23 rather than as a
library call, and that is the rounding, not an approximation of it: at 2^23 an FP32
value's unit in the last place is exactly one, so the addition *is* a round to the
nearest integer with ties to even, and the subtraction is exact for the range the
clamp leaves (0 to 255, far below 2^23). Writing it this way keeps the kernel running
under Triton's interpreter, which has no libdevice, so the arithmetic here can be
checked on a host with no GPU.

Reading ``mean`` and ``std`` with the vector index ``k`` makes them one load per
element rather than three host scalars per launch: they are the checkpoint's
ImageNet constants, held as device weights by the layer.

``frames``, ``rows``, ``columns`` and the strides are launch values; nothing about
the geometry is compiled in, so one module serves every raster.
"""

import triton
import triton.language as tl


#: 2^23: the magnitude at which an FP32's ulp is one, and therefore the constant that
#: turns an addition into a round-half-to-even at one byte. It is a `tl.constexpr`
#: because a kernel may only read globals declared as one.
MAGNITUDE = tl.constexpr(8388608.0)


@triton.jit
def _rgb_bytes_kernel(
    input,
    mean,
    std,
    out,
    frames,
    rows,
    columns,
    channels,
    in0,
    in1,
    in2,
    out0,
    out1,
    out2,
    pdl,
    CHANNELS: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of packed pixels.

    Grid: ``ceil(frames · rows · columns · channels / BLOCK)`` programs. ``pdl`` is
    the operator's own flag and the compiled ``PDL`` says what this module was
    built for; the kernel does not branch on either, because a dependent launch is
    expressed in how the module is launched, not in what it computes.
    """
    program = tl.program_id(0)
    # The extents and strides arrive as 64-bit scalars, so the index is widened to
    # match them once: a frame count times a raster's sides outgrows 32 bits on a
    # long clip, and mixing the two widths is not arithmetic Triton accepts.
    index = (program * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    total = frames * rows * columns * channels
    live = index < total

    # The output's layout: channel innermost, then column, row and frame.
    channel = index % CHANNELS
    rest = index // CHANNELS
    column = rest % columns
    rest = rest // columns
    row = rest % rows
    frame = rest // rows

    # The read side is channel-major, so its three outer axes are strides the
    # operator computed; the columns axis is unit-strided by the crate's row rule.
    source = channel * in0 + frame * in1 + row * in2 + column
    value = tl.load(input + source, mask=live, other=0.0).to(tl.float32)
    deviation = tl.load(std + channel, mask=live, other=0.0)
    average = tl.load(mean + channel, mask=live, other=0.0)

    scaled = (value * deviation + average) * 255.0
    clamped = tl.minimum(tl.maximum(scaled, 0.0), 255.0)
    # Round half to even: at 2^23 an FP32's ulp is one, so this addition rounds to the
    # nearest integer with ties to the even neighbour, which is the reference's rule,
    # and the subtraction is exact over the clamp's range.
    byte = ((clamped + MAGNITUDE) - MAGNITUDE).to(tl.uint8)

    target = frame * out0 + row * out1 + column * out2 + channel
    tl.store(out + target, byte, mask=live)
