"""The ``posterior`` operator's Triton source: the moments to the sample, through its FP16 round trip.

```text
log_variance = clamp(log_variance, -30, 20)
sample       = mean + exp(0.5 · log_variance) · noise
out          = round_to_f16(sample)                     # stored as f32
```

Five f32 roundings in the reference's order, and the order is the specification: `exp` after the
clamp, and the FP16 round trip after the sum. A kernel that keeps the sample in f32 until the store
and rounds once at the end produces a different tensor -- `0.25000014901161194` is stored as `0.25`
by this chain and would not be by that one -- which is why the operator's tolerance is exact.

**Every operand is read through its own row stride.** The two moments are the halves of the encoder's
`[positions, 2 · channels]` output, cut on the channel axis, so their rows are twice as wide as the
sample's and a kernel that assumed a contiguous row would read the noise as a variance.

**Nothing is fused in.** The per-channel affine a conditioner applies afterwards is a diagonal `1×1`
convolution and belongs to the model, not here: this kernel's arithmetic must not depend on a weight
layout.
"""

import triton
import triton.language as tl

#: The reference's clamp of the log-variance (`uniserve/nn/vae/layers.py:134-203`). The lower bound is
#: why an encoder variance of `1e-40` still conditions.
CLAMP_MIN = tl.constexpr(-30.0)

#: The upper bound of the same clamp.
CLAMP_MAX = tl.constexpr(20.0)


@triton.jit
def _posterior_kernel(
    mean_ptr,
    log_variance_ptr,
    noise_ptr,
    out_ptr,
    channels,
    mean_row,
    log_variance_row,
    noise_row,
    out_row,
    total,
    pdl,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    """One element a lane, addressed through its operand's own row stride.

    The flat index is the element's position in the sample's `[positions, channels]` layout; the row
    comes from dividing by the channel count and the column from what is left, and each of the four
    operands turns that pair into its own offset. The casts to int64 are not decoration: the extents
    and strides arrive unsigned and the index is signed, and Triton refuses the modulo across two
    signednesses rather than guessing.
    """
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    live = index < total
    channels = channels.to(tl.int64)
    row = index // channels
    column = index - row * channels
    mean_offset = row * mean_row.to(tl.int64) + column
    log_variance_offset = row * log_variance_row.to(tl.int64) + column
    noise_offset = row * noise_row.to(tl.int64) + column
    out_offset = row * out_row.to(tl.int64) + column

    mean = tl.load(mean_ptr + mean_offset, mask=live, other=0.0)
    log_variance = tl.load(log_variance_ptr + log_variance_offset, mask=live, other=0.0)
    noise = tl.load(noise_ptr + noise_offset, mask=live, other=0.0)

    # The clamp first: `exp` of an unclamped variance is `inf` or `0`, and the reference clamps.
    log_variance = tl.minimum(tl.maximum(log_variance, CLAMP_MIN), CLAMP_MAX)
    scaled = 0.5 * log_variance
    scaled = tl.exp(scaled)
    sample = mean + scaled * noise
    # The round trip, inside the chain: to FP16 and back to FP32, round to nearest even in both
    # directions, before the store and not as part of it.
    sample = sample.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + out_offset, sample, mask=live)
