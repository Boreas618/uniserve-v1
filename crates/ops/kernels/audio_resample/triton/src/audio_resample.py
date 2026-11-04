"""The audio decoder's anti-aliased up- and downsampler: the Triton source of the
``audio_resample`` operator.

Both directions are one depthwise convolution with a Kaiser-windowed sinc filter whose taps sum to
one, and both reproduce the reference module's replicate pad exactly, because the pad is what keeps
the resampler from ringing at the ends of a waveform:

- **Downsample** pads ``pad_left`` samples on the left and ``pad_right`` on the right by
  replication and convolves with stride ``ratio``: ``out[j] = Σ_t x_pad[j·ratio + t] · filter[t]``.
  Its output length is ATen's ``floor((padded − kernel) / ratio) + 1``.
- **Upsample** pads ``pad`` samples on both sides by replication, applies the transposed convolution
  of stride ``ratio`` scaled by ``ratio`` — ``out[q] = ratio · Σ_{t : (q − t) mod ratio = 0}
  x_pad[(q − t) / ratio] · filter[t]`` — and crops ``pad_left`` from the front and ``pad_right`` from
  the end. The caller's crop is exactly what the pad and the filter add, so the result is ``width ·
  ratio`` samples of the reference's window.

The filter is FP32 and shared by every channel; the program expands it rather than materialising
``[channels, 1, kernel]``. One lane is one output sample of one row, ``j`` being the sample and the
padding and crop living in the index; the channels are a loop, since the convolution is depthwise.
``pad`` is the upsampler's and is unused when downsampling.

This is the reference's formula (``kaiser_sinc_filter1d`` and the reference's ``UpSample1d`` /
``DownSample1d``), whose per-sample sum is a ``kernel``-term dot product in FP32 with one rounding
into the output, which is the operator's tolerance.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _audio_resample_kernel(
    x_ptr,
    filter_ptr,
    out_ptr,
    rows,
    channels,
    width,
    out_width,
    pad,
    pad_left,
    pad_right,
    pdl,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    RATIO: tl.constexpr,
    UPSAMPLE: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    """Store one block of the resampled signal.

    Grid: ``ceil(rows · out_width / BLOCK)`` programs. Each lane is one output sample of one row;
    the channels are a loop over the depthwise filter's depth.
    """
    _pdl_prologue(PDL)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    width_of = width.to(tl.int64)
    out_width_of = out_width.to(tl.int64)
    channels_of = channels.to(tl.int64)
    live = index < rows * out_width_of
    row = index // out_width_of
    sample = index % out_width_of
    # The padded signal starts this many samples before the signal's first one.
    left = (pad if UPSAMPLE else pad_left).to(tl.int64)
    last = width_of - 1

    for channel in tl.range(0, channels_of):
        plane = (row * channels_of + channel) * width_of
        accumulate = tl.zeros((BLOCK,), dtype=tl.float32)
        if UPSAMPLE:
            # The transposed convolution's position, before the crop.
            position = sample + pad_left.to(tl.int64)
            for tap in tl.static_range(KERNEL):
                shifted = position - tap
                source = tl.where(
                    (shifted % RATIO) == 0, shifted // RATIO, 0
                )
                inside = (
                    live
                    & (shifted >= 0)
                    & ((shifted % RATIO) == 0)
                    & (source < width_of + 2 * left)
                )
                value = tl.load(
                    x_ptr
                    + plane
                    + tl.minimum(tl.maximum(source - left, 0), last),
                    mask=inside,
                    other=0.0,
                ).to(tl.float32)
                accumulate += value * tl.load(filter_ptr + tap).to(tl.float32)
            accumulate *= RATIO
        else:
            for tap in tl.static_range(KERNEL):
                source = sample * RATIO + tap - left
                accumulate += tl.load(
                    x_ptr + plane + tl.minimum(tl.maximum(source, 0), last),
                    mask=live,
                    other=0.0,
                ).to(tl.float32) * tl.load(filter_ptr + tap).to(tl.float32)
        tl.store(
            out_ptr + (row * channels_of + channel) * out_width_of + sample,
            accumulate.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=live,
        )
