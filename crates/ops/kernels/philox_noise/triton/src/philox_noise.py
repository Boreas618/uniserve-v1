"""Initial noise drawn on the device: the Triton source of the ``philox_noise`` operator.

Every element has an address, so a restored request redraws exactly the same values from its seed and
two requests in one batch draw independent streams. The address is the one ``crates/ipc/src/philox.rs``
gives a random draw:

```text
counter = [pair_low, pair_high, stage, draw]      pair = element_index / 2 (normal only)
key     = [key_low, key_high]                     the row's key, the bits of its i64
words   = philox4x32_10(counter, key)
u1      = 1 − (words[0] >> 8) / 2^24              in (0, 1]
u2      = (words[1] >> 8) / 2^24                  in [0, 1)
r       = sqrt(−2 · ln u1)
out[i]  = r · cos(2π · u2)                        for an even element index
out[i]  = r · sin(2π · u2)                        for an odd element index
```

The **uniform** draw is the same addressing with one word per element: `out[i] = (words[0] >> 8) /
2^24` in `[0, 1)`, one sample per element rather than one pair, so its counter carries the element
index itself. The kind is part of the compilation because the two draw different values from the same
counter and their envelopes differ.

Counter and key words are held in 64-bit registers and masked back to 32 bits after every step, which
is how the reference keeps the 32×32-bit products and the Weyl key additions from overflowing. The
uniform is the exactly representable 24-bit value Philox's high word yields; the normal's arithmetic
is single precision and the store rounds once into the output type.
"""

import triton
import triton.language as tl

#: Philox4x32-10's multipliers and the ten rounds' key increments. The constants are
#: instantiated so a kernel may read them.
_M0 = tl.constexpr(0xD2511F53)
_M1 = tl.constexpr(0xCD9E8D57)
_KEY_BUMP_0 = tl.constexpr(0x9E3779B9)
_KEY_BUMP_1 = tl.constexpr(0xBB67AE85)
_MASK32 = tl.constexpr(0xFFFFFFFF)
_ROUNDS = tl.constexpr(10)

#: A uniform keeps the top 24 bits of its word: exactly representable in FP32, and so
#: is their product with this power of two.
_UNIFORM_SHIFT = tl.constexpr(8)
_UNIFORM_SCALE = tl.constexpr(1.0 / 16_777_216.0)

#: `2π`: the angle the normal's cosine and sine take.
_TURN = tl.constexpr(6.283185307179586)


@triton.jit
def _philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Apply the ten-round Philox4x32 bijection to 32-bit counter and key words.

    Every word is held in a 64-bit register and masked after each step, so the products and the key
    additions cannot overflow and the result is the reference's.
    """
    for round_index in tl.static_range(_ROUNDS):
        # Round zero consumes the caller's key; every later round advances both key words by the
        # Weyl constants before permuting the counter.
        if round_index > 0:
            k0 = (k0 + _KEY_BUMP_0) & _MASK32
            k1 = (k1 + _KEY_BUMP_1) & _MASK32
        product0 = _M0 * c0
        product1 = _M1 * c2
        c0, c1, c2, c3 = (
            ((product1 >> 32) ^ c1 ^ k0) & _MASK32,
            product1 & _MASK32,
            ((product0 >> 32) ^ c3 ^ k1) & _MASK32,
            product0 & _MASK32,
        )
    return c0, c1, c2, c3


@triton.jit
def _philox_words(counter0, counter1, stage, draw, k0, k1):
    """The first two output words of one Philox draw."""
    word0, word1, _unused2, _unused3 = _philox4x32_10(
        counter0, counter1, stage & _MASK32, draw & _MASK32, k0, k1
    )
    return word0, word1


@triton.jit
def _philox_noise_kernel(
    out_ptr,
    keys_ptr,
    elements,
    stage,
    draw,
    BLOCK: tl.constexpr,
    NORMAL: tl.constexpr,  # noqa: N803
):
    """Store one block of one row's noise.

    Grid: ``(rows, ceil(elements / BLOCK))``. Each row's key comes from ``keys``; ``stage`` and
    ``draw`` are the processor stage and the draw index within the token and stage.
    """
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    lanes = block * BLOCK + tl.arange(0, BLOCK)
    live = lanes < elements

    key = tl.load(keys_ptr + row)
    k0 = key.to(tl.uint64) & _MASK32
    k1 = (key >> 32).to(tl.uint64) & _MASK32

    # Both elements of a pair share one counter; the uniform draw takes one element.
    index = (lanes // 2) if NORMAL else lanes
    index = index.to(tl.uint64)
    word0, word1 = _philox_words(
        index & _MASK32,
        (index >> 32) & _MASK32,
        stage.to(tl.uint64),
        draw.to(tl.uint64),
        k0,
        k1,
    )
    first = (word0 >> _UNIFORM_SHIFT).to(tl.float32) * _UNIFORM_SCALE
    if NORMAL:
        second = (word1 >> _UNIFORM_SHIFT).to(tl.float32) * _UNIFORM_SCALE
        radius = tl.sqrt(-2.0 * tl.log(1.0 - first))
        angle = _TURN * second
        value = tl.where(
            lanes % 2 == 0, radius * tl.cos(angle), radius * tl.sin(angle)
        )
    else:
        value = first
    tl.store(
        out_ptr + row * elements + lanes,
        value.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=live,
    )
