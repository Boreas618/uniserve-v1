"""Element type conversion over a strided input: the ``cast`` operator's source.

```text
out[i] = convert(x[i])
```

One program converts one block of the output's row-major elements. The input
is read through its strides (``strided.offsets``) and the output is
contiguous; the two element types and the number of axes are compiled in,
the shape and the strides are words of the launch.

The conversion is the host's ``as``: between integers the low bits are kept,
a signed source extending its sign; a float truncates toward zero into an
integer; a value rounds to nearest even into a float.

A conversion to or from a half-precision type goes through FP32, which is
the one such conversion Triton lowers and the one its interpreter evaluates.
``THROUGH`` says the value takes that step first, and ``NARROWS`` that the
last step is a float narrowing, the only conversion that names a rounding
mode. The step into FP32 from FP64 narrows too, to nearest even, which is
Triton's rule for a narrowing that names none.
"""

import strided
import triton
import triton.language as tl


@triton.jit
def _cast_kernel(
    x,
    out,
    elements,
    d1,
    d2,
    d3,
    d4,
    d5,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    pdl,
    THROUGH: tl.constexpr,  # noqa: N803
    NARROWS: tl.constexpr,  # noqa: N803
    RANK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of the converted input.

    Grid: ``ceil(elements / BLOCK)`` programs. ``d1..d5`` are the output's
    extents and ``s0..s5`` the input's strides, right-aligned.
    """
    strided.pdl_prologue(PDL)

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = index < elements
    source = strided.offsets(
        index, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5, RANK
    )

    value = tl.load(x + source, mask=live, other=0)
    if THROUGH:
        value = value.to(tl.float32)

    if NARROWS:
        converted = value.to(out.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        converted = value.to(out.dtype.element_ty)
    tl.store(out + index, converted, mask=live)
