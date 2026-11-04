"""FP8 block rescaling: the Triton source of the ``rescale_fp8_blocks``
operator.

An FP8 KV page holds E4M3 codes under one scale per block. When a block's scale
grows, the codes already written decode under the old scale and must be
re-encoded under the new one before the new scale is committed. One program
walks one block in tile-wide steps: decode through the logical cache dtype,
re-encode against the grown scale and clamp to the E4M3 finite range (±448).

The kernel body is the one UniServe re-encoded with
(``uniserve_kernels/src/uniserve_kernels/cache.py``), unchanged except that it
imports nothing but Triton: the scale computation, the initialized flags and
the torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit
def _rescale_fp8_kernel(
    values,
    old_scales,
    new_scales,
    initialized,
    width: tl.constexpr,
    tile: tl.constexpr,
    compute_dtype: tl.constexpr,
):
    # values: [blocks, width] contiguous E4M3 codes rewritten in place;
    # old_scales, new_scales and initialized: [blocks]. One program owns one
    # block and walks it in ``tile``-element steps.
    block = tl.program_id(0)
    old = tl.load(old_scales + block)
    new = tl.load(new_scales + block)
    active = tl.load(initialized + block) & (new > old)

    # Uninitialized and non-growing blocks perform only the metadata reads.
    # Scales are never written here: every element of a growing block decodes
    # with its old scale, and the caller commits the new scales after this
    # launch.
    if active:
        for start in range(0, width, tile):
            indices = start + tl.arange(0, tile)
            value = tl.load(
                values + block * width + indices, indices < width, other=0.0
            ).to(tl.float32)
            # Decode, round through the logical cache dtype, re-encode against
            # the grown scale and clamp to the E4M3 finite range (+/-448); the
            # store converts the FP32 result to E4M3.
            decoded = (value * old).to(compute_dtype).to(tl.float32)
            encoded = tl.maximum(-448.0, tl.minimum(448.0, decoded / new))
            tl.store(values + block * width + indices, encoded, indices < width)
