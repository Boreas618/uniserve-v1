"""The vsa_select_tiles program against the operator's stated ranking.

The output is indices and counts, and the operator declares them exact: the
selection is a total order over the scores, not an approximation of one. So
the reference is the operator's own words, transcribed: a dense query tile
keeps every live key tile; a video query tile keeps every live dense tile and
each segment's ``segment_keep`` highest scores, equal scores in ascending tile
order, NaN above every number and -0 equal to +0; an empty query tile keeps
none. The kept tiles come first in ascending order, then the others. The
reference sorts; the kernel bisects a 32-bit order key instead, so agreeing on
every case is the claim.

A segment's members are its tiles, live or not: the operator keeps "of each
segment the ``segment_keep`` tiles of that segment with the highest scores",
and only the dense half of the mask is restricted to live tiles. A dead tile's
pooled scores are zeros, so it can rank.

The cases are the ones a plausible-but-wrong selection passes, each on video
query tiles, the only ones the segment rule runs for:

- **a tie at the threshold**, three equal scores straddling a segment's keep,
  of which only the first in tile order may be kept, and **a segment whose
  members all tie**, more of them than it keeps: the case a ``<=`` instead of
  a ``<`` in the tie rule breaks;
- **a NaN against ``+inf``**, which it ranks above, and **-0 before +0** in
  tile order, which rank equal, with a keep larger than a segment's members, so
  the clamp to the member count matters;
- **dead key tiles** with **a non-zero ``query_offset``**, so the tables are
  read at the query tile's own place in the sequence, and **an empty query
  tile**, whose count is zero;
- **a sequence with no dense tile**, of a tile count that is not a power of
  two, since the kernel's lane block is the next one.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

#: The fixture's tile set: dense tiles (segment -1) and two video segments.
SEGMENTS = [-1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, 0, 0, 0, 1, 1, -1, -1, -1]

#: The tile set of the case with no dense tile, of thirteen tiles.
VIDEO_ONLY = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]

#: The seed every case draws its scores from.
SEED = 29


def tie_at_the_threshold():
    scores = torch.randn(2, 4, len(SEGMENTS)) * 0.5
    # Query tile 2 is a video tile: of segment 0 (tiles 2-6 and 13-15) it
    # keeps 3, and three equal scores straddle that cut, so only tile 4 of
    # them may be kept.
    scores[0, 2, [2, 3, 4, 5, 6, 13, 14, 15]] = torch.tensor(
        [3.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.25, 0.125]
    )
    return scores, SEGMENTS, [], [3, 2], 0


def a_segment_of_ties():
    scores = torch.full((1, 2, len(SEGMENTS)), 1.5)
    # At offset 2 the window's query tiles are segment 0's, video tiles.
    return scores, SEGMENTS, [], [3, 4], 2


def nan_infinity_and_signed_zeros():
    scores = torch.randn(2, 3, len(SEGMENTS))
    # At offset 2 every query tile is a video tile. Segment 1 (tiles 7-11,
    # 16 and 17) keeps one: a NaN above +inf in query 0's row, and -0 before
    # +0 in tile order, both above the rest, in query 1's.
    scores[0, 0, 8] = float("nan")
    scores[0, 0, 9] = float("inf")
    scores[0, 1, [7, 8, 9, 16, 17]] = -1.0
    scores[0, 1, 10] = -0.0
    scores[0, 1, 11] = 0.0
    return scores, SEGMENTS, [], [99, 1], 2


def dead_tiles_at_an_offset():
    return torch.randn(2, 3, len(SEGMENTS)), SEGMENTS, [3, 14], [2, 2], 5


def an_empty_query_tile():
    # Query tile 5, the first of the window at offset 5, is dead too.
    return torch.randn(2, 3, len(SEGMENTS)), SEGMENTS, [3, 5, 14], [2, 2], 5


def no_dense_tile():
    return torch.randn(1, 2, len(VIDEO_ONLY)), VIDEO_ONLY, [], [2, 2], 0


#: Each case: (scores, tile segments, dead tiles, segment keep, query offset).
CASES = {
    "a tie at the threshold": tie_at_the_threshold,
    "a segment of ties": a_segment_of_ties,
    "a NaN, infinity and signed zeros": nan_infinity_and_signed_zeros,
    "dead tiles at an offset": dead_tiles_at_an_offset,
    "an empty query tile": an_empty_query_tile,
    "no dense tile": no_dense_tile,
}


def launch(kernels, scores, segments, sizes, keep, indices, counts, offset):
    """One launch over every (query tile, head) of `scores`."""
    heads, query_tiles, tiles = scores.shape
    kernels._select_tiles_kernel[(query_tiles, heads)](
        scores,
        segments,
        sizes,
        keep,
        indices,
        counts,
        query_tiles,
        tiles,
        offset,
        block=1 << (tiles - 1).bit_length(),
    )


def rank(scores_row, members):
    """The members in the operator's order: a stable descending sort, which
    ranks NaN above every number and -0 equal to +0.
    """  # noqa: D205
    candidates = torch.arange(scores_row.numel())[members]
    order = torch.argsort(scores_row[candidates], descending=True, stable=True)
    return candidates[order]


def select(scores_row, segments, keep):
    """Each segment's `keep` leading members, as a mask over the key tiles."""
    chosen = torch.zeros_like(segments, dtype=torch.bool)
    for segment in range(int(segments.max()) + 1):
        members = segments == segment
        count = min(int(keep[segment]), int(members.sum()))
        if count:
            chosen[rank(scores_row, members)[:count]] = True
    return chosen


def reference(scores, segments, live, keep, offset):
    """Every (head, query tile)'s kept mask."""
    heads, query_tiles, _ = scores.shape
    masks = torch.zeros_like(scores, dtype=torch.bool)
    for head in range(heads):
        for query in range(query_tiles):
            tile = offset + query
            if not bool(live[tile]):
                continue
            # A dense query tile keeps every live key tile; a video one keeps
            # the live dense tiles and every segment's own leading scores.
            if int(segments[tile]) < 0:
                masks[head, query] = live
            else:
                masks[head, query] = (live & (segments < 0)) | select(
                    scores[head, query], segments, keep
                )
    return masks


def order(mask):
    """The kept tiles ascending, then the others ascending."""
    tiles = torch.arange(mask.numel())
    return torch.cat([tiles[mask], tiles[~mask]])


@pytest.mark.parametrize("case", CASES)
def test_the_selection_is_the_operators_ranking(
    case, torch_device, kernel_source
):
    torch.manual_seed(SEED)
    scores, segments, dead, keep, offset = CASES[case]()
    segments = torch.tensor(segments)
    live = torch.ones(segments.numel(), dtype=torch.bool)
    live[dead] = False
    keep = torch.tensor(keep)
    heads, query_tiles, tiles = scores.shape
    indices = torch.full(
        (heads, query_tiles, tiles), -1, dtype=torch.int32, device=torch_device
    )
    counts = torch.full(
        (heads, query_tiles), -1, dtype=torch.int32, device=torch_device
    )

    launch(
        kernel_source("vsa_select_tiles"),
        scores.to(torch_device),
        segments.to(torch.int32).to(torch_device),
        live.to(torch.int32).to(torch_device),
        keep.to(torch.int32).to(torch_device),
        indices,
        counts,
        offset,
    )

    masks = reference(scores, segments, live, keep, offset)
    label = (
        f"{case}: heads {heads} query tiles {query_tiles} tiles {tiles} live "
        f"{int(live.sum())} keep {keep.tolist()} offset {offset}"
    )
    expected_counts = masks.sum(-1).to(torch.int32)
    got_counts = counts.cpu()
    assert torch.equal(got_counts, expected_counts), (
        f"{label}: the counts of "
        f"{int((got_counts != expected_counts).sum())} of {got_counts.numel()} "
        f"query tiles differ: {got_counts.tolist()} vs "
        f"{expected_counts.tolist()}"
    )
    got = indices.cpu()
    for head in range(heads):
        for query in range(query_tiles):
            expected = order(masks[head, query]).to(torch.int32)
            assert torch.equal(got[head, query], expected), (
                f"{label}: head {head} query {query}'s indices are "
                f"{got[head, query].tolist()}, the ranking says "
                f"{expected.tolist()}"
            )
