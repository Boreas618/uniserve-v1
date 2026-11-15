"""A row's few most likely entries: the Triton source of the ``top_logprobs``
operator.

The candidates a request asks for at a position (``top_logprobs``,
``prompt_logprobs``) are a handful of a vocabulary-wide row, and a row is scored
for every sampled token whether or not its request asked, so the selection must
cost the device little. One program reads one ``TILE``-lane chunk of one row and
keeps the chunk's ``CANDIDATES`` largest entries, the largest first, by as many
max reductions over its registers; the grid is ``(rows, tiles)``, so a row's
chunks are selected in parallel. A second launch over the chunks' candidates,
``INDIRECT``, reads each candidate's id beside its value and keeps the row's
``CANDIDATES`` of them, so two launches select a vocabulary's top entries.

Ties keep the lower id first: within a chunk lanes are in id order, and the
chunks' candidates are laid out chunk by chunk, so the lower lane among equal
values is the lower id in both launches. A lane is taken once; a row with fewer
entries than candidates fills the rest with ``-inf`` and id ``-1``.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch


@triton.jit(do_not_specialize=["width", "tiles"])
def _top_logprobs_kernel(
    values_ptr,
    ids_ptr,
    out_values_ptr,
    out_ids_ptr,
    width,
    tiles,
    CANDIDATES: tl.constexpr,
    TILE: tl.constexpr,
    INDIRECT: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Keep the ``CANDIDATES`` largest entries of one chunk of one row.

    ``values`` is ``[rows, width]`` FP32; with ``INDIRECT``, ``ids`` is the same
    shape in i32 and names each entry's id, and otherwise an entry's id is its
    column. ``out_values`` and ``out_ids`` are ``[rows, tiles * CANDIDATES]``:
    chunk ``t`` of row ``r`` writes its candidates at ``r * tiles + t``, the
    largest first. Grid: ``(rows, tiles)``.
    """
    dependent_launch.prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    lanes = tl.arange(0, TILE)
    columns = tile * TILE + lanes
    alive = columns < width
    values = tl.load(values_ptr + row * width + columns, mask=alive, other=-float("inf"))
    if INDIRECT:
        ids = tl.load(ids_ptr + row * width + columns, mask=alive, other=-1)
    else:
        ids = columns.to(tl.int32)

    out = (row * tiles + tile) * CANDIDATES
    for k in tl.static_range(CANDIDATES):
        best = tl.max(tl.where(alive, values, -float("inf")), axis=0)
        # The lowest live lane holding the largest value: the lowest id among equals.
        lane = tl.min(tl.where(alive & (values == best), lanes, TILE), axis=0)
        picked = lanes == lane
        found = lane < TILE
        chosen = tl.sum(tl.where(picked, ids, 0), axis=0)
        tl.store(out_values_ptr + out + k, tl.where(found, best, -float("inf")))
        tl.store(out_ids_ptr + out + k, tl.where(found, chosen, -1))
        alive = alive & ~picked
