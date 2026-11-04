"""torchvision's antialiased uint8 resize: the Triton source of the
``antialiased_resize_uint8`` operator.

This is ``interpolate(scale, mode="bilinear" | "bicubic", antialias=True, align_corners=False)`` on a
``[rows, channels, height, width]`` uint8 frame, which is the video pipeline's downscaling. The filter
is separable: for one output index ``i`` along an axis, ``center = (i + 0.5) · scale − 0.5``,
``support = (bicubic ? 2 : 1) · max(1, scale)``, the taps are ``floor(center ± support + 0.5)``
clamped to the input's extent, and a tap's weight is the kernel evaluated at its distance from the
center over the scale, divided by the taps' sum so a constant image stays constant. The triangle is
``max(0, 1 − |x|)`` and the cubic is the one ATen's antialiased resize uses, α = −0.75.

``scale = input_extent / output_extent`` and the support widening with it is what makes the resize
antialiased: when the image shrinks, the filter widens to the fraction of the source each destination
pixel covers. The weights are derived on device from the two scales, so a graph capture carries no
host-side coefficient table and the program's answer is independent of its caller.

One lane is one output row of one channel of one image, which is the grid the operator launches; the
row's pixels are a loop, each a ``taps × taps`` sum in FP32 over the separable weights. ``MAX_TAPS``
bounds the taps per axis: eight cover a halving with the cubic filter (scale 2 needs about eight) and
a quartering with the triangle, which is the range this operator's scales stay in. The output rounds
to the nearest code value and clamps.

The envelope of the error is one code value: the same FP32 arithmetic, a possibly different rounding
tie.
"""

import triton
import triton.language as tl

#: The taps one axis may use, instantiated so a kernel may read it. Eight cover a
#: halving with the cubic filter and a quartering with the triangle, which is the range
#: the video path's scales stay in.
MAX_TAPS = tl.constexpr(8)

#: The cubic convolution's parameter, as ATen's antialiased resize uses it.
_CUBIC_ALPHA = tl.constexpr(-0.75)

#: The largest code value of the frame, and the offset that rounds to nearest.
_CODE_MAX = tl.constexpr(255.0)
_HALF = tl.constexpr(0.5)


@triton.jit
def _kernel_of(distance, BICUBIC: tl.constexpr):  # noqa: N803
    """The separable filter's value at ``distance``, in units of the source scale."""
    absolute = tl.abs(distance)
    if BICUBIC:
        inner = (
            ((_CUBIC_ALPHA + 2.0) * absolute - (_CUBIC_ALPHA + 3.0)) * absolute * absolute
            + 1.0
        )
        outer = _CUBIC_ALPHA * (
            ((absolute - 5.0) * absolute + 8.0) * absolute - 4.0
        )
        return tl.where(
            absolute <= 1.0, inner, tl.where(absolute < 2.0, outer, 0.0)
        )
    else:
        return tl.maximum(0.0, 1.0 - absolute)


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _antialiased_resize_uint8_kernel(
    x_ptr,
    out_ptr,
    rows,
    channels,
    height,
    width,
    out_height,
    out_width,
    scale_h,
    scale_w,
    pdl,
    CHANNELS: tl.constexpr,
    BICUBIC: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of the resized frame.

    Grid: ``ceil(rows · channels · out_height / BLOCK)`` programs. A lane is one output row of one
    channel of one image; it walks its row's pixels, each a separable sum of the taps the two scales
    give.
    """
    _pdl_prologue(PDL)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    height_of = height.to(tl.int64)
    width_of = width.to(tl.int64)
    out_height_of = out_height.to(tl.int64)
    out_width_of = out_width.to(tl.int64)
    channels_of = channels.to(tl.int64)
    units = rows * channels_of * out_height_of
    live = index < units
    row_unit = index // (channels_of * out_height_of)
    channel = (index // out_height_of) % channels_of
    out_row = index % out_height_of

    # The vertical axis: one center for the lane, and its taps and weights.
    support_h = (2.0 if BICUBIC else 1.0) * tl.maximum(1.0, scale_h)
    center_h = (out_row.to(tl.float32) + _HALF) * scale_h - _HALF
    start_h = tl.floor(center_h - support_h + _HALF).to(tl.int64)
    count_h = tl.floor(center_h + support_h + _HALF).to(tl.int64) - start_h
    count_h = tl.minimum(count_h, MAX_TAPS)
    sum_h = tl.zeros((BLOCK,), dtype=tl.float32)
    for tap in tl.static_range(MAX_TAPS):
        inside = tap < count_h
        weight = tl.where(
            inside,
            _kernel_of(((start_h + tap).to(tl.float32) - center_h) / scale_h, BICUBIC),
            0.0,
        )
        sum_h += weight

    support_w = (2.0 if BICUBIC else 1.0) * tl.maximum(1.0, scale_w)
    source_base = (
        (row_unit * channels_of + channel) * height_of
    ) * width_of
    for out_column in tl.range(0, out_width_of):
        center_w = (out_column.to(tl.float32) + _HALF) * scale_w - _HALF
        start_w = tl.floor(center_w - support_w + _HALF).to(tl.int64)
        count_w = tl.floor(center_w + support_w + _HALF).to(tl.int64) - start_w
        count_w = tl.minimum(count_w, MAX_TAPS)
        total = tl.zeros((BLOCK,), dtype=tl.float32)
        sum_w = tl.zeros((BLOCK,), dtype=tl.float32)
        for horizontal in tl.static_range(MAX_TAPS):
            width_weight = tl.where(
                horizontal < count_w,
                _kernel_of(
                    ((start_w + horizontal).to(tl.float32) - center_w) / scale_w,
                    BICUBIC,
                ),
                0.0,
            )
            sum_w += width_weight
            column = tl.minimum(
                tl.maximum(start_w + horizontal, 0), width_of - 1
            )
            for vertical in tl.static_range(MAX_TAPS):
                height_weight = tl.where(
                    vertical < count_h,
                    _kernel_of(
                        ((start_h + vertical).to(tl.float32) - center_h) / scale_h,
                        BICUBIC,
                    ),
                    0.0,
                )
                source_row = tl.minimum(
                    tl.maximum(start_h + vertical, 0), height_of - 1
                )
                code = tl.load(
                    x_ptr + source_base + source_row * width_of + column,
                    mask=live,
                    other=0,
                )
                total += height_weight * width_weight * code.to(tl.float32)
        normalized = total / (sum_h * sum_w)
        quantized = tl.minimum(
            tl.maximum(normalized + _HALF, 0.0), _CODE_MAX
        ).to(tl.uint8)
        tl.store(
            out_ptr
            + ((row_unit * channels_of + channel) * out_height_of + out_row)
            * out_width_of
            + out_column,
            quantized,
            mask=live,
        )
