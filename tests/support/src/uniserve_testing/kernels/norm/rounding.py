"""The rounding arithmetic the normalization checks' references share.

Every bound of the family is an expression of ``crates/ops/src/tolerance.rs``
over the unit roundoff of an element type: ``gamma`` of a reduction in FP32,
plus a count of roundings into the output type. The expressions are written
here once, in the operator crate's terms.

**A narrow store on Triton's interpreter.** A device rounds an FP32 value to
the nearest BF16 when it stores one; Triton's interpreter truncates it, one
unit in the last place low at worst, and stores an F16 as a device does. A
check whose bound counts the output's rounding truncates its exact reference
on the interpreter (:func:`interpreter_store`), on the value's own mantissa:
truncating a value already rounded to FP32 would step it across a boundary
the kernel's arithmetic stays below. A check that follows every rounding of a
kernel's chain stores the FP32 value the kernel holds at each step
(:func:`store`).
"""

import torch

#: Torch's element type for each element type the operators name.
TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: `ops::tolerance::unit_roundoff`, per element type.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

#: `ops::tolerance::underflow`, per element type: the spacing of its
#: subnormal grid, the absolute part of a bound on a value rounded into it.
#: Below the type's smallest normal a stored value is within half this unit
#: of exact, not within `UNIT_ROUNDOFF` of itself.
UNDERFLOW = {"f16": 2.0**-24, "bf16": 2.0**-133, "f32": 2.0**-149}

#: The significand bits of each element type, with the implicit one.
SIGNIFICAND = {"f16": 11, "bf16": 8, "f32": 24}

#: The significand bits of FP64, with the implicit one.
FP64_SIGNIFICAND = 53


def gamma(reduction, unit=UNIT_ROUNDOFF["f32"]):
    """`Tolerance::gamma`: `n u / (1 - n u)` for `n` roundings of `unit`, one
    past the point where it says nothing.
    """  # noqa: D205
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def truncate(values, dtype):
    """FP64 `values` truncated to the significand of `dtype`, still FP64.

    This is the interpreter's narrow store of the exact value: the bits of the
    FP64 form below the narrow type's significand are cleared, not rounded.
    """
    dropped = FP64_SIGNIFICAND - SIGNIFICAND[dtype]
    bits = values.contiguous().view(torch.int64)
    return (bits & ~((1 << dropped) - 1)).view(torch.float64)


def store(values, dtype, interpreted):
    """FP64 `values` as a kernel stores them into `dtype`, still FP64.

    A device rounds to nearest; Triton's interpreter truncates a BF16 store
    and rounds an F16 one. An FP32 value is the FP64 value's own rounding.
    """
    if dtype == "bf16" and interpreted:
        return truncate(values, dtype)
    return values.to(TORCH_TYPES[dtype]).double()


def interpreter_store(values, dtype, interpreted):
    """The exact `values` a bound holds a stored output to.

    A bound that counts the output's rounding covers a device's store, so the
    reference stays exact there; on the interpreter a BF16 store truncates
    instead, which the reference takes the same way.
    """
    if dtype == "bf16" and interpreted:
        return truncate(values, dtype)
    return values


def outside(difference, allowed):
    """The elements whose difference is not within `allowed`: a NaN difference
    is never within a bound.
    """  # noqa: D205
    return int((~(difference <= allowed)).sum())


def largest_relative(difference, expected):
    """The largest difference relative to its expected value, a NaN counted as
    infinite.
    """  # noqa: D205
    relative = difference / expected.abs().clamp(min=1e-30)
    return relative.nan_to_num(nan=float("inf")).max().item()
