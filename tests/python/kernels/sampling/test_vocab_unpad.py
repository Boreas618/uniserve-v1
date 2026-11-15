"""The vocab_unpad program drops a sharded vocabulary's padding, exactly.

A tensor-parallel ``lm_head`` gathers its ranks' shards one after another:
shard ``r`` is a whole ``[rows, padded / ranks]`` block and holds the ids from
``r * ceil(vocabulary / ranks)`` on, its padding after them. The program copies
each row's real columns out. Values move, so the comparison is exact.

The cases: the fixture's width, gathered and vocabulary equal; a gathered
width above the vocabulary with the padding marked, so a copy taking the
wrong offset shows at once; two ranks' shards, each padded; and four ranks'
shards, the last one short.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`tests.python.kernels.chain`); that launch needs a device.
"""

import pytest
import torch

from tests.python.kernels import chain

pytestmark = pytest.mark.unit

#: The fixture's descriptor: the vocabulary, gathered unpadded, in blocks of
#: 1024 elements.
VOCABULARY = 151936
BLOCK = 1024

#: The element types the operator's `dtype` takes, as the check's `--dtype`
#: offered them.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: The cases: rows, vocabulary, gathered width, ranks, the program's block,
#: whether the padding is marked, and the seed.
CASES = {
    "the fixture's width, gathered and vocabulary equal": (
        2,
        VOCABULARY,
        VOCABULARY,
        1,
        BLOCK,
        False,
        1,
    ),
    "a gathered width above the vocabulary, the padding marked": (
        3,
        1000,
        1024,
        1,
        512,
        True,
        2,
    ),
    "two ranks' shards of three rows, each padded, gathered rank by rank": (
        3,
        1000,
        1008,
        2,
        512,
        True,
        3,
    ),
    "four ranks' shards, the last one short": (5, 1001, 1024, 4, 512, True, 4),
}


def launch(kernels, gathered, out, padded, ranks, block, pdl=False):
    """One launch over every unpadded element, compiled for a dependent
    launch when `pdl` is.
    """  # noqa: D205
    rows, vocabulary = out.shape
    kernels._vocab_unpad_kernel[(-(-rows * vocabulary // block),)](
        gathered,
        out,
        rows,
        padded,
        vocabulary,
        ranks,
        BLOCK=block,
        PDL=pdl,
        launch_pdl=pdl,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_every_real_column_is_copied(case, dtype, torch_device, kernel_source):
    rows, vocabulary, padded, ranks, block, marks_padding, seed = CASES[case]
    generator = torch.Generator().manual_seed(seed)
    width, shard = padded // ranks, -(-vocabulary // ranks)
    held = [min(shard, vocabulary - rank * shard) for rank in range(ranks)]
    shards = torch.randn(ranks, rows, width, generator=generator).to(
        DTYPES[dtype]
    )
    if marks_padding:
        for rank in range(ranks):
            shards[rank, :, held[rank] :] = -7.0
    # The gather lays the ranks' whole shards one after another, rank-major:
    # the buffer is `[ranks, rows, width]`, handed over as `rows * padded`.
    gathered = shards.reshape(rows, padded)
    out = torch.full(
        (rows, vocabulary),
        float("nan"),
        dtype=DTYPES[dtype],
        device=torch_device,
    )

    launch(
        kernel_source("vocab_unpad"),
        gathered.to(torch_device),
        out,
        padded,
        ranks,
        block,
    )

    expected = torch.cat(
        [shards[rank, :, : held[rank]] for rank in range(ranks)], dim=1
    )
    differing = int((out.cpu() != expected).sum())
    assert differing == 0, (
        f"{case}, {dtype}: {rows} x {vocabulary} of {padded} gathered "
        f"columns over {ranks} ranks, block {block}: {differing} of "
        f"{expected.numel()} "
        "differ from the reference"
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("vocab_unpad")
    rows, padded, vocabulary, ranks = 4, 1024, 1000, 2

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["gathered"],
            outputs["out"],
            padded,
            ranks,
            1024,
            pdl,
        )
        return [outputs["out"]]

    case = chain.Case(
        "vocab_unpad",
        {
            "gathered": torch.randn(ranks * rows * padded // ranks).to(
                torch.bfloat16
            )
        },
        {"out": torch.zeros(rows, vocabulary)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"vocab_unpad launched as a dependent: {failures} of {chain.ROUNDS} "
        "rounds differ from its launch after a synchronization"
    )
