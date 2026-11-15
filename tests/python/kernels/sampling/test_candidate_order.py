"""The candidate_order program ranks every row's candidates by token id.

The operator is a permutation and its tolerance is exact: each counted
candidate (an id inside the vocabulary) is ranked by its token id, the row is
filled to its width with ``-1`` ids and ``-inf`` values, and both outputs are
compared bit for bit with that rule. Equal ids are indistinguishable to the
draw, so the order of two copies of one id is not specified: their values are
compared as one group.

The width is a launch value rather than a descriptor field, so the cases are
widths around the program's tile (one candidate, below a tile, exactly one,
one past it, several), each plain, with a repeated id (both copies rank
consecutively), and with an id that is not a token (it takes no rank, and the
row's tail stays the sentinel). The destinations start as ``-1`` ids and
``nan`` values, which no candidate and no filled slot can be, so a slot the
program failed to write fails the comparison. ``0`` would not do: it is a
legal token, and two candidates of one id left a hole that read exactly like
an untouched slot.

A permutation is integer comparison, ranks and moves with no narrowing cast,
so Triton's interpreter answers it as a device does. The built module of the
package's ``f32``/``i32`` descriptor is launched on a device as well, through
the driver as a runner launches it; a module compiled for a dependent launch
reads only what the grid before it wrote
(:mod:`uniserve_testing.kernels.chain`).
"""

import ctypes
import json

import numpy as np
import pytest
import torch
from uniserve_testing.kernels import chain

pytestmark = pytest.mark.unit

#: The row tile the program walks a row in, the package's `BLOCK`.
TILE = 256

#: The widths, around the tile: below it, exactly one, one past it, several.
WIDTHS = [1, 7, 64, 128, 129, 200, TILE, TILE + 1, 300, 3 * TILE + 44]

#: The rows each width is given.
ROWS = 4

SEED = 11

#: The descriptor of the built module the device test launches.
KEY = "values_f32_ids_i32_pdl_false"

#: How a scalar of each Triton type of a module's signature is passed.
SCALARS = {
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
}

#: The row shapes: plain, a repeated id, an id outside the vocabulary.
SHAPES = ("plain", "duplicate", "outside")

CASES = [
    pytest.param(width, shape, id=f"width {width}, {shape}")
    for width in WIDTHS
    for shape in SHAPES
    if width >= 4 or shape == "plain"
]


def words(values, ids, ordered_values, ordered_ids, counts):
    """A launch's arguments in the program's order, the source's and the
    built module's alike. The vocabulary is the row's width, and `counts` the
    program's scratch of ranks, whose contents on entry are not part of the
    interface.
    """  # noqa: D205
    rows, width = values.shape
    return [
        values,
        ids,
        ordered_values,
        ordered_ids,
        counts,
        rows,
        width,
        width,
    ]


def launch(kernels, *operands, pdl=False):
    """One launch of the source over :func:`words` of ``operands``: a program
    per row, compiled for a dependent launch when `pdl` is.
    """  # noqa: D205
    kernels._candidate_order_kernel[(operands[0].shape[0],)](
        *words(*operands),
        BLOCK=TILE,
        PDL=pdl,
        launch_pdl=pdl,
    )


def scenario(width, shape):
    """One case's values and ids, as the top-k operator writes them."""
    generator = np.random.default_rng(SEED)
    values = generator.random((ROWS, width), dtype=np.float32)
    ids = generator.permutation(np.arange(width, dtype=np.int32))
    ids = np.stack([ids] * ROWS)
    if shape == "duplicate":
        # Both copies are candidates and take consecutive ranks.
        ids[:, 1] = ids[:, 0]
    if shape == "outside":
        # What a defect upstream produces: not a candidate, so no rank.
        ids[:, 0] = width + 7
    return values, ids


def wanted(values, ids):
    """The rows the operator writes: the counted candidates ranked, then the
    ``-1``/``-inf`` tail.
    """  # noqa: D205
    rows, width = ids.shape
    want_ids = np.full((rows, width), -1, dtype=np.int32)
    want_values = np.full((rows, width), -np.inf, dtype=np.float32)
    for row in range(rows):
        keep = np.flatnonzero((ids[row] >= 0) & (ids[row] < width))
        order = keep[np.argsort(ids[row][keep], kind="stable")]
        want_ids[row, : order.size] = ids[row][order]
        want_values[row, : order.size] = values[row][order]
    return want_values, want_ids


def faults(got_values, got_ids, values, ids):
    """Every way the rows differ from the rule, naming where."""
    want_values, want_ids = wanted(values, ids)
    found = []
    if not np.array_equal(got_ids, want_ids):
        column = int(np.flatnonzero(got_ids != want_ids)[0])
        found.append(f"ids differ from flat column {column}")
    for row in range(want_ids.shape[0]):
        for token in np.unique(want_ids[row]):
            mine = got_values[row][got_ids[row] == token]
            theirs = want_values[row][want_ids[row] == token]
            if mine.size != theirs.size:
                found.append(
                    f"row {row}: id {token} appears {mine.size} times, want "
                    f"{theirs.size}"
                )
            elif not np.allclose(
                np.sort(mine), np.sort(theirs), equal_nan=True
            ):
                found.append(f"row {row}: id {token} carries other values")
    return found


def destinations(width, device):
    """The outputs and the scratch, holding what no candidate can be."""
    return (
        torch.full((ROWS, width), float("nan"), device=device),
        torch.full((ROWS, width), -1, dtype=torch.int32, device=device),
        torch.full((ROWS, width), 12345, dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize(("width", "shape"), CASES)
def test_every_row_is_ranked_by_token_id(
    width, shape, torch_device, kernel_source
):
    values, ids = scenario(width, shape)
    ordered_values, ordered_ids, counts = destinations(width, torch_device)

    launch(
        kernel_source("candidate_order"),
        torch.from_numpy(values).to(torch_device),
        torch.from_numpy(ids).to(torch_device),
        ordered_values,
        ordered_ids,
        counts,
    )

    found = faults(
        ordered_values.cpu().numpy(), ordered_ids.cpu().numpy(), values, ids
    )
    assert not found, f"width {width}, {shape}: " + "; ".join(found)


@pytest.mark.gpu
def test_the_built_module_ranks_every_row_by_token_id(kernel_module):
    """The cases through the package's built cubin, launched by the driver.

    The words are packed by the module's own recorded signature, and the two
    scratch addresses the compiler appends to every Triton module's
    parameters follow them (``ops::kernel``'s launch passes them too).
    """
    from cuda.bindings import driver

    # The driver calls run in the device's primary context, which torch
    # creates and makes current with its first allocation.
    torch.zeros(1, device="cuda")
    directory = kernel_module("candidate_order", "triton", KEY)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["global_scratch_size"] == 0, (
        f"{KEY} asks for {metadata['global_scratch_size']} bytes of global "
        "scratch, which this launch does not allocate"
    )
    error, module = driver.cuModuleLoadData(
        (directory / "module.cubin").read_bytes()
    )
    assert error == driver.CUresult.CUDA_SUCCESS, f"loading {KEY}: {error}"
    error, function = driver.cuModuleGetFunction(
        module, metadata["name"].encode()
    )
    assert error == driver.CUresult.CUDA_SUCCESS, f"{metadata['name']}: {error}"

    found = []
    for width, shape in (case.values for case in CASES):
        values, ids = scenario(width, shape)
        ordered_values, ordered_ids, counts = destinations(width, "cuda")
        arguments = words(
            torch.from_numpy(values).cuda(),
            torch.from_numpy(ids).cuda(),
            ordered_values,
            ordered_ids,
            counts,
        )
        # Each word's value lives in its own slot until the launch completed,
        # and the tensors stay referenced by `arguments` as long.
        holders = [
            ctypes.c_void_p(word.data_ptr())
            if kind.startswith("*")
            else SCALARS[kind](word)
            for word, kind in zip(
                arguments, metadata["signature"].values(), strict=True
            )
        ]
        holders += [ctypes.c_void_p(0), ctypes.c_void_p(0)]
        parameters = (ctypes.c_void_p * len(holders))(
            *[ctypes.addressof(holder) for holder in holders]
        )
        (error,) = driver.cuLaunchKernel(
            function,
            ROWS,
            1,
            1,
            metadata["num_warps"] * 32,
            1,
            1,
            metadata["shared"],
            torch.cuda.current_stream().cuda_stream,
            parameters,
            0,
        )
        assert error == driver.CUresult.CUDA_SUCCESS, (
            f"launching width {width}: {error}"
        )
        torch.cuda.synchronize()
        found += [
            f"width {width}, {shape}: {fault}"
            for fault in faults(
                ordered_values.cpu().numpy(),
                ordered_ids.cpu().numpy(),
                values,
                ids,
            )
        ]
    driver.cuModuleUnload(module)
    assert not found, f"{KEY}: " + "; ".join(found)


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("candidate_order")
    rows, width = 4, 300

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["values"],
            inputs["ids"],
            outputs["values"],
            outputs["ids"],
            outputs["counts"],
            pdl=pdl,
        )
        return [outputs["values"], outputs["ids"]]

    case = chain.Case(
        "candidate_order",
        {
            "values": torch.rand(rows, width),
            "ids": torch.randperm(width).to(torch.int32).repeat(rows, 1),
        },
        {
            "values": torch.full((rows, width), float("nan")),
            "ids": torch.full((rows, width), -1, dtype=torch.int32),
            "counts": torch.zeros(rows, width, dtype=torch.int32),
        },
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"candidate_order launched as a dependent: {failures} of "
        f"{chain.ROUNDS} rounds differ from its launch after a "
        "synchronization"
    )
