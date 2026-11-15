"""The inverse_cdf program draws the token a uniform draw reaches, exactly.

The draw takes a row's candidates in ascending token order, forms the
inclusive cumulative sum of their probabilities, and picks the lowest id whose
share the draw reaches: the first candidate whose cumulative mass exceeds
``draw * total``, strictly, so a zero-mass candidate is never drawn and a draw
of zero lands on the first candidate holding mass. A token id, so the
comparison is exact. The candidates arrive in ascending id order, the
operator's precondition (its caller orders them through ``candidate_order``),
so each case sorts its row before the draw.

The cases: a row far wider than one tile, walked in ten of them; the fixture's
width with four draws across the mass; a zero-mass candidate at the lowest id,
which a draw of zero must skip; mass only the highest id holds, reachable by a
draw near the top of the unit interval; and one survivor, on which every draw
ends.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`tests.python.kernels.chain`); that launch needs a device.
"""

import pytest
import torch

from tests.python.kernels import chain

pytestmark = pytest.mark.unit

#: The fixture's candidates per row, and its tile.
WIDTH = 2048
TILE = 2048

#: The cases: rows, width, tile, the draws, and the shape of the mass (a
#: zero-mass lowest id, one survivor, or most of the mass on the highest id),
#: and the seed.
CASES = {
    "a row far wider than one tile, walked in ten of them": (
        2,
        20480,
        2048,
        (0.0, 0.5),
        None,
        5,
    ),
    "the fixture's width, four draws across the mass": (
        4,
        WIDTH,
        TILE,
        (0.0, 0.25, 0.75, 0.9999999),
        None,
        1,
    ),
    "a zero-mass candidate at the lowest id: a draw of zero must skip it": (
        2,
        WIDTH,
        TILE,
        (0.0, 0.5),
        "zero mass",
        2,
    ),
    "the mass only the last candidate by id holds is reachable": (
        2,
        WIDTH,
        TILE,
        (0.99, 0.9999999),
        "last holds mass",
        4,
    ),
    "one survivor, so every draw ends on it": (
        2,
        WIDTH,
        TILE,
        (0.0, 0.9999999),
        "one survivor",
        3,
    ),
}


def launch(kernels, probabilities, ids, draws, tokens, tile, pdl=False):
    """One launch: a program per row, walking it in tiles of `tile`; compiled
    for a dependent launch when `pdl` is.
    """  # noqa: D205
    rows, width = probabilities.shape
    kernels._inverse_cdf_kernel[(rows,)](
        probabilities,
        ids,
        draws,
        tokens,
        rows,
        width,
        BLOCK=tile,
        PDL=pdl,
        launch_pdl=pdl,
    )


@pytest.mark.parametrize("case", CASES)
def test_every_draw_lands_on_the_lowest_id_it_reaches(
    case, torch_device, kernel_source
):
    rows, width, tile, draws, mass, seed = CASES[case]
    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.rand(
        rows, width, generator=generator, dtype=torch.float32
    )
    if mass == "zero mass":
        # The lowest id of every row (id 0, column 0) holds no mass, so a draw
        # of zero must skip it: an inclusive comparison lands on it.
        probabilities[:, 0] = 0.0
    if mass == "one survivor":
        probabilities[:, 1:] = 0.0
    # Distinct ids per row, out of order, so "the lowest id the draw reaches"
    # is defined and a row does not arrive sorted.
    vocabulary = max(4 * width, 1024)
    ids = torch.zeros(rows, width, dtype=torch.int32)
    for row in range(rows):
        if mass == "zero mass":
            ids[row] = (
                torch.randperm(vocabulary - 1, generator=generator)[:width] + 1
            ).to(torch.int32)
            ids[row, 0] = 0
        else:
            ids[row] = torch.randperm(vocabulary, generator=generator)[
                :width
            ].to(torch.int32)
    if mass == "last holds mass":
        # 90 % of the mass on the highest id, the rest spread thin: a draw
        # near the top lands only on it, so a cumulative sum the draw walks
        # past, or a strict comparison at the last element, makes it
        # unsamplable.
        probabilities[:, :] = 0.001
        for row in range(rows):
            probabilities[row, int(torch.argmax(ids[row].long()))] = 0.9
    # The operator's precondition: candidates in ascending id order.
    order = torch.argsort(ids.long(), dim=1, stable=True)
    ids = torch.gather(ids.long(), 1, order).to(torch.int32)
    probabilities = torch.gather(probabilities, 1, order)
    drawn = torch.tensor(draws[:rows], dtype=torch.float32)
    tokens = torch.full((rows,), -1, dtype=torch.int32, device=torch_device)

    launch(
        kernel_source("inverse_cdf"),
        probabilities.to(torch_device),
        ids.to(torch_device),
        drawn.to(torch_device),
        tokens,
        tile,
    )

    wanted = []
    for row in range(rows):
        ordered = probabilities[row].double()
        crossed = torch.cumsum(ordered, dim=0) > drawn[row].double() * (
            ordered.sum()
        )
        if crossed.any():
            wanted.append(int(ids[row][crossed].min()))
        elif (ordered > 0.0).any():
            wanted.append(int(ids[row][ordered > 0.0].max()))
        else:
            wanted.append(0)
    got = tokens.cpu().tolist()
    assert got == wanted, (
        f"{case}: {rows} x {width} in tiles of {tile}, draws {draws[:rows]}: "
        f"drew {got}, the reference draws {wanted}"
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("inverse_cdf")
    rows, width, tile = 4, 16384, 8192

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["probabilities"],
            inputs["ids"],
            inputs["draws"],
            outputs["tokens"],
            tile,
            pdl,
        )
        return [outputs["tokens"]]

    case = chain.Case(
        "inverse_cdf",
        {
            "probabilities": torch.softmax(
                torch.randn(rows, width) * 3.0, dim=-1
            ),
            "ids": torch.arange(width, dtype=torch.int32).repeat(rows, 1),
            "draws": torch.rand(rows),
        },
        {"tokens": torch.full((rows,), -7, dtype=torch.int32)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"inverse_cdf launched as a dependent: {failures} of {chain.ROUNDS} "
        "rounds differ from its launch after a synchronization"
    )
