"""The logits_shape program adjusts, penalizes and scales every logit.

The program applies a row's logit adjustments (its slot's bias row, ``-inf``
at the tokens its ban mask names), then the repetition, frequency and
presence penalties to every token the row has seen, then the temperature, to
every column::

    repeated = logit / repetition   if logit > 0 else logit * repetition
    adjusted = repeated - frequency * count - presence   (seen, not -inf)
    out      = adjusted / temperature                     (divide; 0 -> 1)

It is arithmetic, held to the operator's bound ``rounding(5, F32, F32)``:
``2 gamma_5 + u`` against each element's own magnitude, with no absolute
part. The floor the operands' magnitudes imply (the penalties are subtracted
from a value their own magnitudes set, and the bias's sum rounds once before
them) is reported beside the count that decides.

The cases: the fixture's shape, with penalties, a temperature and a masked
``-inf`` logit a count would otherwise penalize; a zero temperature, which
divides by one; two extra tokens counted on the device, one of them counting
nothing; a restricted, biased and banning row beside a banning row and a
padded row; and the fixture's vocabulary adjusted, greedy. A row without
adjustments must read neither the bias row nor the ban masks it is handed, so
both are filled in every case.

Every test needs a device: the program rounds with libdevice's ``add_rn``,
``div_rn``, ``mul_rn`` and ``sub_rn``, which Triton's interpreter does not
evaluate.
"""

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

#: The unit round-off of FP32, as `ops::tolerance::unit_roundoff` gives it.
UNIT_ROUNDOFF = 2.0**-24

#: The fixture's descriptor: `dtype_bf16_vocabulary_151936_block_1024_extras_0
#: _divide_true`.
VOCABULARY = 151936
BLOCK = 1024

#: The most lanes one step of the program holds, the package's `LANES`.
LANES = 4096

#: The element types the operator's `dtype` takes, as the check's `--dtype`
#: offered them.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: The cases: rows, vocabulary, block, extra tokens, whether the temperature
#: divides, the temperature, whether a seen token's logit is masked, whether
#: rows carry adjustments, and the seed.
CASES = {
    "the fixture's shape, penalties and a temperature": (
        2,
        VOCABULARY,
        BLOCK,
        0,
        True,
        0.7,
        True,
        False,
        1,
    ),
    "a zero temperature, which divides by one": (
        2,
        512,
        512,
        0,
        True,
        0.0,
        False,
        False,
        2,
    ),
    "two extra tokens, one of them counting nothing": (
        2,
        500,
        256,
        2,
        False,
        0.7,
        True,
        False,
        3,
    ),
    "a restricted, biased and banning row, a banning row and a padded row": (
        3,
        500,
        256,
        0,
        True,
        0.7,
        True,
        True,
        4,
    ),
    "the fixture's vocabulary adjusted, greedy": (
        3,
        VOCABULARY,
        BLOCK,
        0,
        True,
        0.0,
        False,
        True,
        5,
    ),
}


def launch(kernels, operands, out, extras, divide, block):
    """One launch: a program per row and block of its columns.

    ``operands`` are the logits, counts, slots, parameters, extra tokens,
    bias rows, adjustments and ban masks, in the program's order.
    """
    rows, vocabulary = out.shape
    kernels._logits_shape_kernel[(rows, -(-vocabulary // block))](
        *operands,
        out,
        rows,
        vocabulary,
        int(divide),
        BLOCK=block,
        STEP=min(block, LANES),
        EXTRAS=extras,
        DIVIDE=divide,
        PDL=False,
    )


def gamma(reduction):
    """`gamma_n = n u / (1 - n u)` for `n` FP32 roundings."""
    terms = reduction * UNIT_ROUNDOFF
    return terms / (1.0 - terms)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_every_logit_is_shaped_within_the_bound(
    case, dtype, torch_device, kernel_source
):
    (
        rows,
        vocabulary,
        block,
        extra_count,
        divide,
        temperature,
        masked,
        adjusted,
        seed,
    ) = CASES[case]
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocabulary, generator=generator).to(
        DTYPES[dtype]
    )
    # A handful of tokens already seen, one of them twice, one of them only in
    # a row whose logit is `-inf` (a masked token the penalties leave alone).
    counts = torch.zeros(rows, vocabulary, dtype=torch.int32)
    counts[:, 1] = 3
    counts[:, 5] = 1
    counts[:, 7] = 2
    slots = torch.arange(rows, dtype=torch.int32)
    if masked:
        logits[0, 7] = float("-inf")
    parameters = torch.zeros(rows, 8, dtype=torch.float32)
    parameters[:, 0] = temperature
    parameters[:, 4] = 1.3
    parameters[:, 5] = 0.2
    parameters[:, 6] = 0.1
    extras = None
    if extra_count:
        extras = torch.full((rows, extra_count), -1, dtype=torch.int32)
        extras[:, 0] = 9
    # The bias rows and the ban masks are filled in every case: a row whose
    # flag does not name them must read neither.
    words = -(-vocabulary // 32)
    bias = torch.randn(rows, vocabulary, generator=generator) * 4.0
    bans = torch.randint(
        -(2**31),
        2**31 - 1,
        (rows, words),
        generator=generator,
        dtype=torch.int32,
    )
    adjustments = torch.tensor([[0, -1]] * rows, dtype=torch.int32)
    if adjusted:
        # Row 0 restricts its tokens (its bias row is `-inf` outside the
        # allowed set) and bans tokens at word edges, the sign bit and the
        # last token; the last row is padding, whose flag names a bias row its
        # slot cannot address.
        bias[0, ::3] = float("-inf")
        bans.zero_()
        for token in (2, 31, 32, vocabulary - 1):
            bans[0, token // 32] |= torch.tensor(1 << (token % 32)).to(
                torch.int32
            )
        bans[1, 0] = 1 << 4
        adjustments[0] = torch.tensor([1, 0])
        adjustments[1] = torch.tensor([0, 1])
        slots[-1] = -1
        adjustments[-1] = torch.tensor([1, -1])
    out = torch.full(
        (rows, vocabulary),
        float("nan"),
        dtype=torch.float32,
        device=torch_device,
    )

    # A call without extra tokens hands the program any int32 buffer there,
    # which it never reads.
    operands = [
        logits,
        counts,
        slots,
        parameters,
        extras if extras is not None else slots,
        bias,
        adjustments,
        bans,
    ]
    launch(
        kernel_source("logits_shape"),
        [operand.to(torch_device) for operand in operands],
        out,
        extra_count,
        divide,
        block,
    )

    # The adjustments first: a flagged row with a slot adds its bias row, and
    # a row with a mask bans the tokens its bits name.
    exact = logits.double()
    token = torch.arange(vocabulary)
    for row in range(rows):
        if adjustments[row, 0] != 0 and slots[row] >= 0:
            exact[row] += bias[slots[row].long()].double()
        if adjustments[row, 1] >= 0:
            mask = bans[adjustments[row, 1].long()].long() & 0xFFFFFFFF
            banned = ((mask[token // 32] >> (token % 32)) & 1) != 0
            exact[row, banned] = float("-inf")
    exact_counts = counts.double()
    # A padded row reads no counts.
    exact_counts[slots < 0] = 0.0
    if extras is not None:
        for column in range(extra_count):
            for row in range(rows):
                if extras[row, column] >= 0:
                    exact_counts[row, extras[row, column].long()] += 1.0
    temperatures, repetition, frequency, presence = (
        parameters[:, column].double().unsqueeze(-1) for column in (0, 4, 5, 6)
    )

    repeated = torch.where(exact > 0.0, exact / repetition, exact * repetition)
    shaped = repeated - frequency * exact_counts - presence
    applies = (exact_counts > 0.0) & (exact != float("-inf"))
    expected = torch.where(applies, shaped, exact)
    if divide:
        expected = expected / torch.where(
            temperatures > 0.0, temperatures, torch.ones_like(temperatures)
        )
    biased = torch.zeros(rows, vocabulary, dtype=torch.float64)
    for row in range(rows):
        if adjustments[row, 0] != 0 and slots[row] >= 0:
            biased[row] = exact[row].abs()

    got = out.cpu().double()
    relative = 2.0 * gamma(5) + UNIT_ROUNDOFF
    floor = UNIT_ROUNDOFF * (
        repeated.abs().nan_to_num(posinf=0.0)
        + (frequency * exact_counts).abs()
        + presence.abs()
        + biased.nan_to_num(posinf=0.0)
    )
    # Equal values differ by nothing, where a masked `-inf` minus itself would
    # say NaN; a NaN difference is outside every bound.
    difference = torch.where(
        got == expected, torch.zeros_like(got), (got - expected).abs()
    )
    outside = int((~(difference <= relative * expected.abs())).sum())
    outside_floor = int(
        (~(difference <= relative * expected.abs() + floor)).sum()
    )
    largest = (
        (difference / expected.abs().clamp(min=1e-30))
        .nan_to_num(nan=float("inf"))
        .max()
        .item()
    )
    assert outside == 0, (
        f"{case}, {dtype}: {rows} x {vocabulary}, block {block}, extras "
        f"{extra_count}, "
        f"divide {divide}, temperature {temperature:g}: {outside} of "
        f"{expected.numel()} outside rtol {relative:.3e} (max relative "
        f"{largest:.3e}, max absolute "
        f"{difference.nan_to_num(nan=float('inf')).max().item():.3e}); with "
        f"the operands' floor {outside_floor} outside"
    )
