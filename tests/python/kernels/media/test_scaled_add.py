"""The scaled_add program against the operator's update and its two identities.

The step is scalar FP32 arithmetic per element:

```text
target <- beta · target + alpha · source
```

Two coefficient pairs carry the operator's exact results: with ``beta = 0``
and ``alpha = 1`` it is a copy, and with ``beta = 1`` and ``alpha = 0`` the
identity, both bit for bit in FP32. A general pair is compared in FP32 within
three units of FP32 rounding of the update's largest magnitude.

A ``beta = 0`` step is a replacement: the reference's FP32 copies widen a
source into a target nothing has written yet, so a target holding NaNs must
come out as ``alpha · source`` exactly. ``0 · NaN`` is a NaN, so a kernel that
loaded the target before scaling it writes the poisoning through, which no
finite target shows.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

#: (rows, columns, block, alpha, beta, exact): a copy and the identity, exact,
#: then two general pairs, the last over several blocks of 1024.
STEPS = [
    pytest.param(2, 64, 64, 1.0, 0.0, True, id="a copy"),
    pytest.param(2, 64, 64, 0.0, 1.0, True, id="the identity"),
    pytest.param(3, 100, 128, 0.25, 0.75, False, id="alpha 0.25 beta 0.75"),
    pytest.param(4, 5376, 1024, 0.5, 1.0, False, id="alpha 0.5 beta 1"),
]

#: (rows, columns, block, alpha) of a replacement over a NaN target.
REPLACEMENTS = [(2, 64, 64, 1.0), (3, 100, 128, 0.5)]

#: The seed every case draws its operands from.
SEED = 5


def launch(kernels, target, source, alpha, beta, block):
    """One launch updating every element of `target` in place."""
    kernels._scaled_add_kernel[(-(-target.numel() // block), 1, 1)](
        target,
        source,
        alpha,
        beta,
        target.numel(),
        BLOCK=block,
        num_warps=8 if block >= 1024 else 4,
    )


@pytest.mark.parametrize(
    ("rows", "columns", "block", "alpha", "beta", "exact"), STEPS
)
def test_a_step_matches_the_update(
    rows, columns, block, alpha, beta, exact, torch_device, kernel_source
):
    torch.manual_seed(SEED)
    target = torch.rand(rows, columns, dtype=torch.float32)
    source = torch.rand(rows, columns, dtype=torch.float32)
    expected = beta * target + alpha * source
    target = target.to(torch_device)

    launch(
        kernel_source("scaled_add"),
        target,
        source.to(torch_device),
        alpha,
        beta,
        block,
    )

    got = target.cpu()
    label = f"{rows}x{columns} block {block}, alpha {alpha} beta {beta}"
    if exact:
        differing = int((got != expected).sum())
        assert differing == 0, (
            f"{label}: {differing} of {got.numel()} elements differ from the "
            "exact result"
        )
        return
    error = (got - expected).abs().max().item()
    tolerance = (
        torch.finfo(torch.float32).eps * 3.0 * expected.abs().max().item()
    )
    assert error <= tolerance, (
        f"{label}: max |error| = {error:.3e}, tolerance {tolerance:.3e}"
    )


@pytest.mark.parametrize(("rows", "columns", "block", "alpha"), REPLACEMENTS)
def test_a_replacement_never_reads_the_target(
    rows, columns, block, alpha, torch_device, kernel_source
):
    torch.manual_seed(SEED)
    target = torch.full(
        (rows, columns), float("nan"), dtype=torch.float32, device=torch_device
    )
    source = torch.rand(rows, columns, dtype=torch.float32)

    launch(
        kernel_source("scaled_add"),
        target,
        source.to(torch_device),
        alpha,
        0.0,
        block,
    )

    got = target.cpu()
    assert torch.equal(got, source * alpha), (
        f"{rows}x{columns} block {block}, alpha {alpha}, beta 0 over a NaN "
        f"target: {int(torch.isnan(got).sum())} NaN(s) survived, "
        f"{int((got != source * alpha).sum())} of {got.numel()} elements "
        "differ"
    )
