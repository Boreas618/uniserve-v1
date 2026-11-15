"""The gelu program against both of PyTorch's GELU forms.

```text
exact:  0.5 x (1 + erf(x / sqrt(2)))
tanh:   x sigmoid(sqrt(8/pi) (x + 0.044715 x^3))
```

The two are separate compilations of one operator, and the operator's bound
is ``rounding(1, f32, dtype)`` with ``8 * 8 * u`` of absolute room for the
``erf``'s or the ``exp``'s last place. Each form is checked against **its
own** formula, and the cases are where the two formulas part far enough that
a kernel evaluating the wrong one cannot hide:

- ``x = +-2, +-3, +-6``: the forms differ by about ``1e-4`` here, three orders
  past the bound, so a module compiled ``exact`` that evaluates ``tanh`` (or
  the other way round) fails;
- ``x = 0`` and ``x = +-1e-3``: the neighbourhood of the identity, where both
  forms are ``0.5 x`` and a kernel missing the ``0.5``, or the ``sqrt(2)`` in
  ``erf``, shows;
- ``x = +-10`` and ``x = +-40``: ``erf`` and the logistic both saturate, where
  a kernel that clamps its argument, or that computes ``1 + tanh(u)`` in a way
  that loses the far tail, leaves the function but not the bound;
- a row that does not fill the last block, and every dtype the manifest
  declares.

The ``exact`` form evaluates ``libdevice.erf``, which Triton's interpreter
does not implement, so it runs on a device only.

The reference makes the kernel's rounds, in the kernel's order, at FP32 (the
cube, the ``0.044715`` product, ``sqrt(8/pi)``'s product, the exponential, the
logistic) and takes only ``erf`` and ``exp`` themselves exactly, in FP64,
before rounding them back. A reference that keeps the whole formula in FP64
measures the distance between two precisions instead.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: `gelu.rs`: one rounding into the output plus the transcendental's last
#: place over `|x| <= 8`.
ROUNDINGS = 1
VALUE_ENVELOPE = 8.0
SQRT_8_OVER_PI = 1.5957691216057308
SQRT_HALF = 0.7071067811865476

#: Where the two formulas part: whatever a module was compiled for, these
#: values say it evaluated that one and not the other.
PARTING = [0.0, 1e-3, -1e-3, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 6.0, -6.0]

#: Both ends of the saturation, where the transcendental's own argument range
#: matters. At `x = -40` the exponent overflows to infinity and the logistic to
#: zero, which is the form's limit and the right answer.
SATURATED = [10.0, -10.0, 40.0, -40.0, 1e-8, -1e-8]

#: The rows a form is launched over: the decided values in every dtype, a wide
#: row whose tail does not fill the block, and a row of two blocks.
ROWS = {
    "decided values, f32": ("f32", PARTING + SATURATED, 256),
    "decided values, bf16": ("bf16", PARTING + SATURATED, 256),
    "decided values, f16": ("f16", PARTING + SATURATED, 256),
    "a row of 130 with a masked tail, bf16": ("bf16", 130, 256),
    "a row of 1024 in blocks of 512, f32": ("f32", 1024, 512),
}

#: The seed of a row drawn at random.
SEED = 11


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(element):
    """`Tolerance::rounding(1, f32, dtype).with_atol(8 * envelope * u)`."""
    unit = {
        torch.bfloat16: 2.0**-8,
        torch.float16: 2.0**-11,
        torch.float32: 2.0**-24,
    }[element]
    return 2.0 * gamma(ROUNDINGS, 2.0**-24) + unit, 8.0 * VALUE_ENVELOPE * unit


def reference(values, approximate):
    """The chosen form at FP32 step by step, with `erf` and `exp` exact and
    rounded back in.
    """  # noqa: D205
    x = values.to(torch.float32)
    if approximate == "tanh":
        inner = torch.full_like(x, SQRT_8_OVER_PI) * (
            x + 0.044715 * (x * x) * x
        )
        exponential = torch.exp((-inner).to(torch.float64)).to(torch.float32)
        return (x / (1.0 + exponential)).to(torch.float64)
    error = torch.erf((x * SQRT_HALF).to(torch.float64)).to(torch.float32)
    return (0.5 * x * (1.0 + error)).to(torch.float64)


def launch(kernels, values, out, columns, approximate, block):
    """One launch over `values`, with the grid the operator computes."""
    elements = values.numel()
    kernels._gelu_kernel[((elements + block - 1) // block, 1, 1)](
        values,
        out,
        elements,
        columns,
        APPROXIMATE=approximate,
        BLOCK=block,
        PDL=False,
    )


@pytest.mark.parametrize("row", ROWS)
@pytest.mark.parametrize(
    "approximate", ["tanh", pytest.param("exact", marks=pytest.mark.gpu)]
)
def test_a_form_matches_its_own_formula_within_the_operators_bound(
    approximate, row, torch_device, kernel_source
):
    dtype, values, block = ROWS[row]
    element = DTYPES[dtype]
    if isinstance(values, int):
        columns = values
        torch.manual_seed(SEED)
        values = torch.randn(3, columns, dtype=torch.float32)
    else:
        columns = len(values)
    values = (
        torch.as_tensor(values, dtype=torch.float32)
        .reshape(-1, columns)
        .to(element)
        .to(torch_device)
    )
    out = torch.full_like(values, float("nan"))

    launch(kernel_source("gelu"), values, out, columns, approximate, block)

    expected = reference(values, approximate)
    got = out.to(torch.float64)
    relative, absolute = bound(element)
    error = (got - expected).abs()
    outside = (~(error <= absolute + relative * expected.abs())).reshape(-1)
    if outside.any():
        position = int(outside.nonzero()[0])
        pytest.fail(
            f"{approximate} {dtype} {tuple(values.shape)} block {block}: "
            f"{int(outside.sum())} of {values.numel()} elements outside the "
            f"bound; first at {position}: {float(out.reshape(-1)[position])} "
            f"vs {float(expected.reshape(-1)[position])} for x = "
            f"{float(values.reshape(-1)[position])}, rtol {relative:.3e} atol "
            f"{absolute:.3e}"
        )
