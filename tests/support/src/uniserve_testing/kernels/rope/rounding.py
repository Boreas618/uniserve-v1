"""The rounding arithmetic the rotary embedding checks' references share.

Every bound of the family is an expression of ``crates/ops/src/tolerance.rs``
over the unit roundoff of an element type; the expressions are written here
once, in the operator crate's terms.

**A rotation's store, apart from its arithmetic.** A rotation check launches
its kernel twice: with an FP32 output, whose values it holds to the reference
at the bound, and with the case's own output, which must be bit for bit a
store of the wide result. A device rounds to nearest; Triton's interpreter
truncates a BF16 store and rounds an F16 one, so a narrow output is a copy of
the wide one either rounded to nearest or truncated (:func:`store_models`).
"""

import torch

#: Torch's element type for each element type the operators name.
TORCH_TYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}

#: `ops::tolerance::unit_roundoff`, per torch element type.
UNIT_ROUNDOFF = {
    torch.bfloat16: 2.0**-8,
    torch.float16: 2.0**-11,
    torch.float32: 2.0**-24,
}

#: The significand bits of each narrow element type, with the implicit one,
#: and of FP64.
SIGNIFICAND = {torch.bfloat16: 8, torch.float16: 11}
FP64_SIGNIFICAND = 53


def gamma(reduction, unit=UNIT_ROUNDOFF[torch.float32]):
    """`Tolerance::gamma`: `n u / (1 - n u)` for `n` roundings of `unit`, one
    past the point where it says nothing.
    """  # noqa: D205
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def outside(difference, allowed):
    """The elements whose difference is not within `allowed`: a NaN difference
    is never within a bound.
    """  # noqa: D205
    return ~(difference <= allowed)


def store_models(element, wide):
    """The two narrow copies of FP32 `wide` a store may leave in `element`:
    rounded to nearest, and truncated on the significand.
    """  # noqa: D205
    values = wide.to(torch.float64)
    dropped = FP64_SIGNIFICAND - SIGNIFICAND[element]
    bits = values.contiguous().view(torch.int64)
    truncated = (bits & ~((1 << dropped) - 1)).view(torch.float64)
    return {
        "rounded to nearest": values.to(element),
        "truncated": truncated.to(element),
    }


def stored_copy(where, narrow, wide):
    """Assert `narrow` is a rounded or a truncated copy of FP32 `wide`.

    Raises:
        AssertionError: Naming the first element that is neither, with both
            copies' values there.
    """
    models = store_models(narrow.dtype, wide)
    agreeing = [
        kind for kind, model in models.items() if torch.equal(narrow, model)
    ]
    if agreeing:
        return
    nearest = models["rounded to nearest"]
    differing = int((narrow != nearest).sum())
    position = int((narrow != models["truncated"]).reshape(-1).nonzero()[0])
    raise AssertionError(
        f"{where}: the stored value is neither a rounded nor a truncated copy "
        f"of the wide result ({differing} of {narrow.numel()} differ from the "
        f"nearest rounding); first at {position}: "
        f"{float(narrow.reshape(-1)[position])} vs nearest "
        f"{float(nearest.reshape(-1)[position])} and truncated "
        f"{float(models['truncated'].reshape(-1)[position])}"
    )
