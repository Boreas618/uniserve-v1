"""The rows program in its two arithmetic modes, bit for bit.

The operator places or accumulates the rows a call picks out by index::

    replace: destination[indices[j]] = source[j]   a copy
    add:     destination[indices[j]] += source[j]  one rounding into its type

``replace`` copies, so it is exact. ``add`` sums in FP32 and rounds once on
the store, and the reference is the operator's own two lines: the sum formed
in FP64 (exact for two BF16 or F16 values, correctly rounded for two FP32
ones) and stored the way the program's store leaves it, so both modes are
compared bit for bit rather than within a bound.

The cases are the ones a plausible but wrong program passes:

- a row wider than the program's block, whose tail program masks lanes, and
  a row narrower than it;
- rows that leave most of the destination untouched, because a program that
  rewrote the whole destination passes every case where every row is named;
- ``add`` into rows that already hold a value, so a copy or a zeroed sum shows;
- a repeated index, which the operator allows. The grid orders each source
  row's own writes and nothing orders two programs naming one row, so the
  interpreter's sequential walk is the only ordering there is: the case pins
  the semantics the program implements there, and the package states the
  limit (a device orders nothing between those programs).

The launch is the package's: ``place`` (``SKIP``) stores nothing for a
negative index and is no mode of these cases, so ``SKIP`` is false here.
"""

import pytest
import torch

from tests.python.kernels import chain

pytestmark = pytest.mark.unit

#: The element types the cases store in.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: The columns one program covers.
BLOCK = 1024

#: The cases: the element type, the mode, the destination's columns and rows,
#: and the source rows' indices (one source row per index).
CASES = {
    # The fixture's width against a block-sized lane mask, with a repeated
    # index.
    "replace, bf16, the fixture's width, a repeated index": (
        "bf16",
        "replace",
        5120,
        40,
        [0, 39, 7, 7, 12, 3],
    ),
    "add, bf16, the fixture's width": (
        "bf16",
        "add",
        5120,
        40,
        [0, 39, 7, 12, 3, 30],
    ),
    "replace, f16, two blocks": ("f16", "replace", 2048, 8, [1, 6, 2]),
    "add, f32, a row narrower than the block": (
        "f32",
        "add",
        130,
        5,
        [4, 0, 3, 1],
    ),
    # Rows wider than one block whose tail is not a multiple of it.
    "replace, bf16, a tail past whole blocks": (
        "bf16",
        "replace",
        5376,
        6,
        [5, 0],
    ),
    "add, f16, a tail past whole blocks": ("f16", "add", 5376, 6, [2, 5, 1]),
}


def launch(kernels, source, destination, indices, mode, columns, pdl=False):
    """One launch: a program per source row and block of columns, compiled
    for a dependent launch when `pdl` is.
    """  # noqa: D205
    count = indices.numel()
    kernels._rows_kernel[(count, -(-columns // BLOCK), 1)](
        source,
        destination,
        indices,
        count,
        columns,
        BLOCK,
        COLUMNS=columns,
        REPLACE=mode == "replace",
        BLOCK=BLOCK,
        PDL=pdl,
        SKIP=False,
        launch_pdl=pdl,
    )


def stored(element, exact, interpreted):
    """What the program's store leaves in ``element``, from the exact FP64
    value.

    A device and the interpreter cast F16 and F32 to the nearest value. The
    interpreter **truncates a BF16 store**: it drops the low 16 bits of the
    FP32 mantissa instead of rounding them (measured: a plain store of 4,096
    FP32 values into BF16 matches the truncation of all 4,096 and torch's
    nearest rounding of 2,038). Reproducing the store's own cast keeps the
    comparison exact instead of an allowance of two units, which would judge
    the interpreter rather than the program.
    """  # noqa: D205
    exact = exact.to(torch.float64)
    if element is torch.bfloat16 and interpreted:
        # Clear the low 45 bits of the FP64 mantissa (BF16 keeps seven of the
        # fifty-two), and never round to a narrower type in between: rounding
        # to FP32 first carries a value just below a BF16 boundary onto it,
        # one unit above a correct program. The cast that follows is exact,
        # the value being on the BF16 grid already.
        truncated = exact.contiguous().view(torch.int64) & -(1 << 45)
        return truncated.view(torch.float64).to(torch.bfloat16)
    return exact.to(element)


@pytest.mark.parametrize("case", CASES)
def test_every_row_lands_bit_for_bit_and_the_rest_stay(
    case, torch_device, interpreted, kernel_source
):
    dtype, mode, columns, rows, indices = CASES[case]
    if len(set(indices)) < len(indices) and not interpreted:
        pytest.skip(
            "a repeated index lands two programs on one row, which a device "
            "orders nothing between"
        )
    element = DTYPES[dtype]
    torch.manual_seed(17)
    source = torch.randn(len(indices), columns, dtype=element)
    before = torch.randn(rows, columns, dtype=element)
    # A copy: on the host `to` would hand back `before` itself.
    destination = before.clone().to(torch_device)

    launch(
        kernel_source("rows"),
        source.to(torch_device),
        destination,
        torch.tensor(indices, dtype=torch.int32, device=torch_device),
        mode,
        columns,
    )
    destination = destination.cpu()

    # The operator's text, walked in the order the indices give.
    expected = before.clone()
    for row, target in enumerate(indices):
        if mode == "replace":
            expected[target] = source[row]
        else:
            expected[target] = stored(
                element,
                expected[target].double() + source[row].double(),
                interpreted,
            )
    difference = (destination.double() - expected.double()).abs()
    position = int(difference.reshape(-1).argmax())
    assert torch.equal(destination, expected), (
        f"{case}: {int((difference > 0).sum())} of {destination.numel()} "
        f"elements differ; worst at {position}: "
        f"{float(destination.reshape(-1)[position])} vs "
        f"{float(expected.reshape(-1)[position])}"
    )

    # A row no index names keeps its values, which a zero difference above
    # would not show if the program had rewritten it with themselves.
    touched = torch.zeros(rows, dtype=torch.bool)
    touched[indices] = True
    assert not touched.all(), (
        f"{case}: every row is named, so nothing tests the untouched rows"
    )
    assert torch.equal(destination[~touched], before[~touched]), (
        f"{case}: a row no index names was written"
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("rows")
    count, columns = 8, BLOCK

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["source"],
            outputs["destination"],
            inputs["indices"],
            "replace",
            columns,
            pdl,
        )
        return [outputs["destination"]]

    case = chain.Case(
        "rows",
        {
            "source": torch.randn(count, columns).to(torch.bfloat16),
            "indices": torch.randperm(32)[:count].to(torch.int32),
        },
        {"destination": torch.zeros(32, columns, dtype=torch.bfloat16)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"rows launched as a dependent: {failures} of {chain.ROUNDS} rounds "
        "differ from its launch after a synchronization"
    )
