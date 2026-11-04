"""Interleaved 16-bit PCM from two float channels.

The Triton source of the ``pcm_interleave`` operator, the audio decoder ends with one ``[samples]`` float channel per side and the
artifact is 16-bit PCM with the sides interleaved:

```text
out[j, 0] = quantize(left[j])
out[j, 1] = quantize(right[j])

quantize(x) = round_half_to_even(clamp(x, -1, 1) · 32767)     as i16
```

The clamp is the reference's — an out-of-range sample is a decoder artifact
clipped at full scale, not a wrapping overflow. The rounding is the reference's
too: ties go to the even neighbour, which is what ``numpy.rint`` and Rust's
``f32::round_ties_even`` do, so a program that rounds ties away from zero would
differ on exactly the half samples and nowhere else. That is why the rounding is
one PTX instruction (``cvt.rni.f32.f32``, round to nearest integer, ties to even)
rather than an arithmetic trick: the formula is the contract, and the operator's
tolerance is exact.

One program handles ``BLOCK`` samples: a sample's two outputs are written beside
each other by the same program, so the interleaving needs no second pass, and the
sample count travels as a word because a request's duration varies while the
arithmetic does not.
"""

import triton
import triton.language as tl


@triton.jit
def _quantize(x):
    """``round_half_to_even(clamp(x, -1, 1) · 32767)`` as an integral float.

    Ties go to the even neighbour, as ``cvt.rni`` rounds them: the reference's
    ``numpy.rint`` and ``f32::round_ties_even`` do the same, so the two agree on
    the half samples as well as the rest.
    """
    clamped = tl.minimum(tl.maximum(x, -1.0), 1.0) * 32767.0
    return tl.inline_asm_elementwise(
        "cvt.rni.f32.f32 $0, $1;",
        "=r,r",
        [clamped],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):  # noqa: N803
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pcm_interleave_kernel(
    left,
    right,
    out,
    samples,
    block,
    pdl,
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of interleaved 16-bit PCM.

    Grid: ``ceil(samples / BLOCK)`` programs. The two channels are read at the
    same index and written to ``2j`` and ``2j + 1`` of a contiguous output, so the
    interleaving is the address arithmetic of one store pair.

    ``block`` is the module's own block, the same value as the ``BLOCK``
    constexpr: it is in the argument list because the operator launches it and the
    module's signature keeps the operator's order, and this kernel reads the
    constexpr rather than the word.
    """
    _pdl_prologue(PDL)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < samples.to(tl.int64)
    # The conversion truncates, which is exact on an integral float: the value has
    # already been rounded to the nearest integer by `_quantize`.
    tl.store(
        out + 2 * index,
        _quantize(tl.load(left + index, mask=valid)).to(tl.int16),
        mask=valid,
    )
    tl.store(
        out + 2 * index + 1,
        _quantize(tl.load(right + index, mask=valid)).to(tl.int16),
        mask=valid,
    )
