"""The top_logprobs program keeps a row's few most likely entries, exactly.

A row's ``candidates`` largest entries come out of two launches of one
program: each chunk of ``tile`` entries keeps its own, then one program per
row keeps the row's among the chunks' (``indirect``, reading them with their
ids). The result is sorted descending, ties by ascending id, and past a row's
entries a candidate is ``-inf`` with id ``-1``. A selection, so the comparison
is exact; the ties and a row narrower than its candidates are the cases it
turns on.

The cases: the vocabulary row keeping the deployment's twenty; tied entries
across and within chunks; and a row narrower than its candidates.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`uniserve_testing.kernels.chain`); that launch needs a device.
"""

import pytest
import torch
from uniserve_testing.kernels import chain

pytestmark = pytest.mark.unit

#: The cases: rows, width, candidates kept, the chunk's tile, whether entries
#: tie, and the seed.
CASES = {
    "the vocabulary row, the deployment's twenty": (
        3,
        2048,
        20,
        4096,
        False,
        1,
    ),
    "tied entries across and within chunks": (2, 300, 5, 128, True, 2),
    "a row narrower than its candidates": (2, 3, 5, 4, False, 3),
}


def launch(
    kernels,
    values,
    ids,
    out_values,
    out_ids,
    tiles,
    candidates,
    tile,
    indirect,
    pdl=False,
):
    """One launch: a program per row and chunk of `tile` entries, each keeping
    its `candidates` largest; `indirect` reads the entries' ids from `ids`.
    Compiled for a dependent launch when `pdl` is.
    """  # noqa: D205
    rows, width = values.shape
    kernels._top_logprobs_kernel[(rows, tiles)](
        values,
        ids,
        out_values,
        out_ids,
        width,
        tiles,
        CANDIDATES=candidates,
        TILE=tile,
        INDIRECT=indirect,
        PDL=pdl,
        launch_pdl=pdl,
    )


@pytest.mark.parametrize("case", CASES)
def test_every_row_keeps_its_largest_entries_in_order(
    case, torch_device, kernel_source
):
    rows, width, candidates, tile, ties, seed = CASES[case]
    kernels = kernel_source("top_logprobs")
    generator = torch.Generator().manual_seed(seed)
    values = -torch.rand(rows, width, generator=generator) * 8
    if ties:
        # Equal values in two chunks and inside one: the lower ids first.
        values[0, [3, 1, tile + 2, 2 * tile - 1]] = 0.0

    def filled(columns, value, dtype):
        return torch.full(
            (rows, columns), value, dtype=dtype, device=torch_device
        )

    # Each chunk's candidates, then the row's among them.
    chunks = -(-width // tile)
    merged = chunks * candidates
    chunk_values = filled(merged, float("nan"), torch.float32)
    chunk_ids = filled(merged, -7, torch.int32)
    # A launch that is not `indirect` reads no ids: the chunks' own stand in.
    launch(
        kernels,
        values.to(torch_device),
        chunk_ids,
        chunk_values,
        chunk_ids,
        chunks,
        candidates,
        tile,
        False,
    )
    out_values = filled(candidates, float("nan"), torch.float32)
    out_ids = filled(candidates, -7, torch.int32)
    launch(
        kernels,
        chunk_values,
        chunk_ids,
        out_values,
        out_ids,
        1,
        candidates,
        1 << (merged - 1).bit_length(),
        True,
    )
    out_values, out_ids = out_values.cpu(), out_ids.cpu()

    kept = min(candidates, width)
    differing = 0
    for row in range(rows):
        order = torch.argsort(
            values[row].double(), descending=True, stable=True
        )
        differing += int((out_ids[row, :kept] != order[:kept]).sum())
        differing += int(
            (out_values[row, :kept] != values[row][order][:kept]).sum()
        )
        differing += int((out_ids[row, kept:] != -1).sum())
    assert differing == 0, (
        f"{case}: {rows} x {width} keeping {candidates} in chunks of {tile}: "
        f"{differing} entries differ from the reference; got ids "
        f"{out_ids.tolist()}"
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("top_logprobs")
    rows, width, tile, candidates = 4, 4096, 1024, 8
    tiles = width // tile

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["values"],
            inputs["values"],
            outputs["values"],
            outputs["ids"],
            tiles,
            candidates,
            tile,
            False,
            pdl,
        )
        return [outputs["values"], outputs["ids"]]

    case = chain.Case(
        "top_logprobs",
        {"values": torch.randn(rows, width)},
        {
            "values": torch.full((rows, tiles * candidates), float("nan")),
            "ids": torch.full(
                (rows, tiles * candidates), -1, dtype=torch.int32
            ),
        },
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"top_logprobs launched as a dependent: {failures} of {chain.ROUNDS} "
        "rounds differ from its launch after a synchronization"
    )
