#!/usr/bin/env python3
"""Run `vsa_select_tiles` and check it against the operator's stated ranking, on the CPU.

    TRITON_INTERPRET=1 crates/program-server/python/.venv/bin/python selection_check.py [--package DIR]

The output is indices and counts, and the operator declares them **exact**: the selection is a total
order over the scores, not an approximation of one. So the reference here is not the tree's other
implementation but the operator's own words, transcribed: a dense query tile keeps every live key
tile, a video query tile keeps every live dense tile and each segment's `segment_keep` highest
scores, equal scores in ascending tile order, NaN above every number and -0 equal to +0, and an
empty query tile keeps none. The kept tiles come first in ascending order, then the others. The
reference sorts; the kernel bisects a 32-bit order key instead, so agreeing on every case is the
claim.

The cases are the ones a plausible-but-wrong selection passes:
- **a tie at the threshold**, where `keep` copies of one score straddle the cut and only the first
  ones in tile order may be kept — the case a `>=` instead of `>` in the tie rule breaks;
- **NaN scores**, which rank above every number including `+inf`, and **-0 against +0**, which rank
  equal;
- **`segment_keep` larger than a segment's live members**, so the clamp to the member count matters;
- **a dense query tile with dead key tiles**, **an empty query tile** whose count is zero, and a
  **video query tile in a sequence with no dense tiles at all**;
- **a non-zero `query_offset`**, so the tables are read at the query tile's own place in the
  sequence rather than at the start;
- **a tile count that is not a power of two**, since the kernel's lane block is the next one.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

import torch


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "vsa_select_tiles.py"
    spec = importlib.util.spec_from_file_location("vsa_select_tiles_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select(scores_row, segments, keep):
    """The operator's ranking for one (head, query tile) row: a mask over the key tiles.

    A segment's members are its tiles, **live or not**: the operator says a video query tile keeps
    "of each segment the `segment_keep` tiles of that segment with the highest scores", and only the
    dense half of the mask is restricted to live tiles. A dead tile's pooled scores are zeros, so it
    can rank, and the operator does not exclude it — my first reference did, and the kernel was
    right.
    """
    chosen = torch.zeros_like(segments, dtype=torch.bool)
    for segment in range(int(segments.max()) + 1):
        members = segments == segment
        count = min(int(keep[segment]), int(members.sum()))
        if count == 0:
            continue
        order = rank(scores_row, members)
        chosen[order[:count]] = True
    return chosen


def rank(scores_row, members):
    """The members in the operator's order: descending score, ties by ascending tile, NaN last."""
    tiles = torch.arange(scores_row.numel())
    candidates = tiles[members]
    values = scores_row[candidates]
    nan = torch.isnan(values)
    # NaN ranks above every number, `+inf` included; -0 and +0 rank equal, so the sort key is the
    # value with NaN's sign removed and the tile index breaks ties.
    numeric = torch.where(nan, torch.zeros_like(values), values)
    descending = torch.argsort(-numeric, stable=True)
    ordered = torch.cat([descending[~nan[descending]], descending[nan[descending]]])
    return candidates[ordered]


def reference(scores, segments, live, keep, query_offset):
    """Every (head, query tile)'s chosen mask, and the operator's index order from it."""
    heads, query_tiles, tiles = scores.shape
    masks = torch.zeros_like(scores, dtype=torch.bool)
    for head in range(heads):
        for query in range(query_tiles):
            tile = query_offset + query
            if not bool(live[tile]):
                continue
            # A dense query tile (a negative segment) keeps every live key tile; a video one keeps
            # the live dense tiles and every segment's own leading scores.
            masks[head, query] = live.clone()
            if int(segments[tile]) >= 0:
                masks[head, query] = live & (segments < 0)
                masks[head, query] |= select(scores[head, query], segments, keep)
    return masks


def order_from_mask(mask):
    """The kept tiles ascending, then the unkept ones ascending, as the operator stores them."""
    tiles = torch.arange(mask.numel())
    kept = tiles[mask]
    others = tiles[~mask]
    return torch.cat([kept, others])


def run(module, *, scores, segments, live, keep, query_offset=0, seed=29):
    """One launch, and the operator's own ranking as the reference."""
    heads, query_tiles, tiles = scores.shape
    sizes = torch.where(live, torch.ones_like(live), torch.zeros_like(live)).to(torch.int32)
    indices = torch.full((heads, query_tiles, tiles), -1, dtype=torch.int32)
    counts = torch.full((heads, query_tiles), -1, dtype=torch.int32)
    block = 1
    while block < tiles:
        block *= 2

    module._select_tiles_kernel[(query_tiles, heads)](
        scores,
        segments.to(torch.int32),
        sizes,
        keep.to(torch.int32),
        indices,
        counts,
        query_tiles,
        tiles,
        query_offset,
        block=block,
    )

    masks = reference(scores, segments, live, keep, query_offset)
    expected_counts = masks.reshape(heads, query_tiles, tiles).sum(-1).to(torch.int32)
    if not torch.equal(counts, expected_counts):
        raise SystemExit(
            f"{what(scores, live, keep, query_offset)}: the counts differ; "
            f"{int((counts != expected_counts).sum())} of {counts.numel()} query tiles"
        )
    for head in range(heads):
        for query in range(query_tiles):
            expected = order_from_mask(masks[head, query])
            if not torch.equal(indices[head, query], expected.to(torch.int32)):
                raise SystemExit(
                    f"{what(scores, live, keep, query_offset)}: head {head} query {query}'s "
                    f"indices are {indices[head, query].tolist()}, the ranking says "
                    f"{expected.tolist()}"
                )
    return int(indices.numel())


def what(scores, live, keep, query_offset):
    return (
        f"heads {scores.shape[0]} query tiles {scores.shape[1]} tiles {scores.shape[2]} "
        f"live {int(live.sum())} keep {keep.tolist()} offset {query_offset}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/vsa_select_tiles.py",
    )
    options = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("run this with TRITON_INTERPRET=1: the kernel is checked on the CPU")
    module = load_kernel(options.package)
    torch.manual_seed(29)

    checked = 0
    # The fixture's shape: dense tiles, two video segments, and a tie across the keep boundary.
    tiles = 21
    segments = torch.tensor([-1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, 0, 0, 0, 1, 1, -1, -1, -1])
    live = torch.ones(tiles, dtype=torch.bool)
    keep = torch.tensor([3, 2])
    scores = torch.randn(2, 4, tiles) * 0.5
    scores[0, 0, 4] = scores[0, 0, 5] = scores[0, 0, 6]  # a tie straddling segment 0's keep of 3
    checked += run(module, scores=scores, segments=segments, live=live, keep=keep)

    # A segment whose members all tie, more of them than `keep` keeps: the cut falls between equal
    # scores, so only the first `keep` in tile order may be kept. This is the case a `<=` instead of
    # a `<` in the tie rule breaks, and the earlier tie did not reach the boundary.
    tied = torch.randn(1, 2, tiles)
    tied[0, :, :] = torch.arange(tiles, dtype=torch.float32)[None, :] * 0.0 + 1.5
    # The query tiles must be **video** tiles for the segment rule to run at all: at offset 2 the
    # window's queries are segments 0 and 0, and an earlier version of this case sat on two dense
    # queries, passed, and proved nothing (a mutation of the tie rule went through it unnoticed).
    checked += run(
        module, scores=tied, segments=segments, live=live, keep=torch.tensor([3, 4]),
        query_offset=2,
    )

    # NaN above every number, -0 equal to +0, and a keep larger than the segment's membership.
    scores = torch.randn(2, 3, tiles)
    scores[0, 1, 8] = float("nan")
    scores[0, 1, 9] = float("inf")
    scores[0, 2, 10] = 0.0
    scores[0, 2, 11] = -0.0
    checked += run(module, scores=scores, segments=segments, live=live, keep=torch.tensor([99, 1]))

    # A dead key tile, a non-zero offset, and an empty query tile.
    live = torch.ones(tiles, dtype=torch.bool)
    live[[3, 14]] = False
    sizes = torch.where(live, torch.ones_like(live), torch.zeros_like(live)).to(torch.int32)
    scores = torch.randn(2, 3, tiles)
    query_offset = 5
    checked += run(
        module, scores=scores, segments=segments, live=live, keep=torch.tensor([2, 2]),
        query_offset=query_offset,
    )
    empty = live.clone()
    empty[query_offset] = False
    checked += run(
        module, scores=scores, segments=segments, live=empty, keep=torch.tensor([2, 2]),
        query_offset=query_offset,
    )

    # No dense tile anywhere in the window, and a tile count that is not a power of two.
    tiles = 13
    segments = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    scores = torch.randn(1, 2, tiles)
    checked += run(
        module, scores=scores, segments=segments, live=torch.ones(tiles, dtype=torch.bool),
        keep=torch.tensor([2, 2]),
    )
    print(f"ok: {checked} index entries over six tile sets, exact against the operator's ranking")
    return 0


if __name__ == "__main__":
    sys.exit(main())
