"""The BigVGAN decoder's activation block, fused: the ``audio_activation`` operator's Triton source.

Five stages that must not materialise the upsampled signal, which is ``ratio`` times the audio and
the decoder's largest allocation by far:

```text
1. replicate pad the signal by kernel/ratio − 1 on each side
2. depthwise transposed convolution, stride `ratio`, `kernel` taps, one filter shared by every
   channel, scaled by `ratio`
3. crop 3·(kernel − ratio)/2 samples from each end, which leaves length·ratio samples
4. SnakeBeta:  y = x + sin²(exp(alpha)·x) / (exp(beta) + 1e-9), per channel
5. replicate pad by kernel/2 − 1 and kernel/2, then a depthwise convolution of `kernel` taps at
   stride `ratio`, which yields floor((length·ratio − 1)/ratio) + 1 = length samples
```

One program owns a channel's slice of output samples and computes the upsampled values it needs
from the input itself, so the intermediate of stage 3 never exists in memory. A lane's work is
`kernel` low-pass taps, each needing one SnakeBeta-activated transposed-convolution value, which
is itself a `kernel`-tap sum in which only the taps matching the stride contribute.

Every product and sum is FP32 and the result rounds once into the output's element type, which is
the operator's tolerance.
"""

import triton
import triton.language as tl


@triton.jit
def _snake_beta(value, alpha, beta, beta_floor):
    """SnakeBeta of one value, with the channel's two log-space parameters."""
    angle = tl.sin(tl.exp(alpha) * value)
    return value + (angle * angle) / (tl.exp(beta) + beta_floor)


@triton.jit
def _audio_activation_kernel(
    values_ptr,
    alpha_ptr,
    beta_ptr,
    filter_up_ptr,
    filter_down_ptr,
    out_ptr,
    channels,
    kernel,
    ratio,
    length,
    out_length,
    block,
    pdl,
    KERNEL: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Store one channel's block of the fused activation.

    Grid: ``(channels, ceil(length / BLOCK))``. Both lengths are launch words — an audio length
    follows the request's duration — while the tap count, the ratio and the block width are the
    compilation.
    """
    channel = tl.program_id(0).to(tl.int64)
    samples = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    length_of = length.to(tl.int64)
    out_length_of = out_length.to(tl.int64)
    live = samples < out_length_of

    # Stage 1's pad and stage 3's crop, derived as the operator derives them.
    upsample_pad: tl.constexpr = KERNEL // RATIO - 1
    crop: tl.constexpr = (3 * (KERNEL - RATIO)) // 2
    pad_left: tl.constexpr = KERNEL // 2 - 1

    alpha = tl.load(alpha_ptr + channel).to(tl.float32)
    beta = tl.load(beta_ptr + channel).to(tl.float32)
    plane = channel * length_of
    upsampled = length_of * RATIO
    last_input = length_of - 1
    last_upsampled = upsampled - 1

    accumulate = tl.zeros((BLOCK,), dtype=tl.float32)
    for tap in tl.static_range(KERNEL):
        # The low-pass reads the padded upsampled signal at `samples · ratio + tap`.
        position = samples * RATIO + tap - pad_left
        # Replicate at both ends of the upsampled signal.
        inner = tl.minimum(tl.maximum(position, 0), last_upsampled)
        # The transposed convolution's position before the crop, and its tap sum.
        convolution = inner + crop
        upsampled_value = tl.zeros((BLOCK,), dtype=tl.float32)
        for up_tap in tl.static_range(KERNEL):
            shifted = convolution - up_tap
            source = shifted // RATIO
            contributing = (shifted % RATIO == 0) & (shifted >= 0) & (
                source < length_of + 2 * upsample_pad
            )
            # Stage 1's pad, by replication of the input's own ends.
            input_index = tl.minimum(
                tl.maximum(source - upsample_pad, 0), last_input
            )
            value = tl.load(
                values_ptr + plane + input_index, mask=live & contributing, other=0.0
            ).to(tl.float32)
            upsampled_value += value * tl.load(filter_up_ptr + up_tap).to(tl.float32)
        upsampled_value *= RATIO
        # Stage 4, per channel, then stage 5's tap weight.
        activated = _snake_beta(upsampled_value, alpha, beta, 1e-9)
        accumulate += activated * tl.load(filter_down_ptr + tap).to(tl.float32)

    tl.store(
        out_ptr + channel * out_length_of + samples,
        accumulate.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=live,
    )
