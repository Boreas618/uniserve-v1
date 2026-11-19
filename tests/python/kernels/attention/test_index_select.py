"""The index_select program keeps each row's best entries, exactly.

The operator (`ops::attention::IndexSelect`) keeps the ``kept`` largest of a
row's pooled scores::

    pooled[t, b] = max_{c < pool} s[t, b * pool + c]     +inf for the block
                   (visible[t] - 1) // pool when pool > 1 and visible[t] > 0
    ranked[t]    = the blocks below ceil(visible[t] / pool), every column of
                   candidate blocks
    out[t, i]    = the kept ranked blocks, -1 where pooled is -inf or past
                   the ranked ones; mapped through candidate blocks of B
                   groups: candidates[t, c // B] * B + c % B for a kept
                   column c

and states the order they leave in: the strictly greater ones by column, then
the ties at the ``kept``-th score by column, so a tie keeps the lower column.
A selection, so the comparison is exact, slot for slot, against the rule
written in torch over the scores' monotone keys (the order the program ranks
by). The output is pre-filled with a sentinel no entry can be, so a slot
nothing wrote fails the comparison.

The cases: a row of a few thousand scores over several chunks with a ``-inf``
tail, and one whose scores past its visible count are finite and unranked;
scores of a handful of values, where nearly everything ties; rows with
fewer finite scores than are kept, whose surplus slots are ``-1``; the block
mode over a column count no block divides, whose newest visible block is
pinned whatever its scores; and scores of candidate blocks, some unnamed,
mapped back to their groups. The device cases are the model's: decode rows
over 2^20 scores keeping 512, and 131072 blocks of 8 keeping 2048.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

#: The value no slot of the output can hold, which a slot nothing wrote keeps.
SENTINEL = -7

#: The scores one chunk of the program reads in the interpreter's cases.
TILE = 512


def monotone(scores):
    """The unsigned keys the program orders FP32 scores by, as int64."""
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    negative = (bits >> 31) == 1
    return torch.where(negative, bits ^ 0xFFFFFFFF, bits | 0x80000000)


def pooled(scores, visible, pool):
    """The pooled scores of each row: a block's largest, the pinned block
    ``+inf``.
    """  # noqa: D205
    rows, columns = scores.shape
    blocks = (columns + pool - 1) // pool
    if pool == 1:
        return scores.clone()
    padded = torch.full((rows, blocks * pool), float("-inf"))
    padded[:, :columns] = scores
    best = padded.reshape(rows, blocks, pool).amax(dim=2)
    for row in range(rows):
        if visible[row] > 0:
            best[row, (int(visible[row]) - 1) // pool] = float("inf")
    return best


def reference(scores, visible, kept, pool=1, candidates=None, block=0):
    """The kept entries of each row in the order the operator states."""
    ranked = pooled(scores, visible, pool)
    if candidates is None:
        # A row ranks the blocks its visible groups lie in alone.
        for row in range(scores.shape[0]):
            ranked[row, (int(visible[row]) + pool - 1) // pool :] = float(
                "-inf"
            )
    out = torch.empty(scores.shape[0], kept, dtype=torch.int32)
    for row in range(scores.shape[0]):
        keys = monotone(ranked[row])
        threshold = torch.topk(keys, kept).values[-1]
        greater = (keys > threshold).nonzero().flatten()
        equal = (keys == threshold).nonzero().flatten()
        chosen = torch.cat((greater, equal[: kept - greater.numel()]))
        if candidates is None:
            entry = chosen.to(torch.int64)
        else:
            named = candidates[row][chosen // block].to(torch.int64)
            entry = torch.where(named >= 0, named * block + chosen % block, -1)
        entry = torch.where(ranked[row][chosen] == float("-inf"), -1, entry)
        out[row] = entry.to(torch.int32)
    return out


#: The words `IndexSelect::launch` packs, in its order.
WORDS = ("scores", "visible", "candidates", "out", "columns", "kept")


def key(pool, block, tile):
    """The descriptor key `IndexSelectDescriptor` prints."""
    return f"pool_{pool}_candidates_{block}_tile_{tile}"


def launch(
    kernels,
    scores,
    visible,
    kept,
    device,
    pool=1,
    candidates=None,
    block=0,
    tile=TILE,
    stated=None,
):
    """One launch: a program per row, reading ``tile`` scores a chunk,
    compiled as the call states it or as ``stated``, the compilation the
    package's launch table states for its key.
    """  # noqa: D205
    rows, columns = scores.shape
    out = torch.full((rows, kept), SENTINEL, dtype=torch.int32, device=device)
    unread = torch.zeros(1, dtype=torch.int32, device=device)
    words = {
        "scores": scores.to(device),
        "visible": visible.to(device),
        "candidates": unread if candidates is None else candidates.to(device),
        "out": out,
        "columns": columns,
        "kept": kept,
    }
    if stated is None:
        kernel = kernels._index_select_kernel
        constants = {"POOL": pool, "CANDIDATES": block, "LANES": tile // pool}
    else:
        kernel, constants = stated.kernel, stated.constexprs
    kernel[(rows,)](*(words[name] for name in WORDS), **constants)
    return out.cpu()


def assert_kept(where, got, expected):
    """Assert the slots agree, naming the first row and slot that do not."""
    wrong = (got != expected).nonzero()
    assert wrong.numel() == 0, (
        f"{where}: {wrong.shape[0]} slots differ; first at row "
        f"{int(wrong[0, 0])} slot {int(wrong[0, 1])}: "
        f"{int(got[tuple(wrong[0])])} kept where the rule keeps "
        f"{int(expected[tuple(wrong[0])])}"
    )


def test_a_long_row_keeps_its_best_across_chunks(torch_device, kernel_source):
    generator = torch.Generator().manual_seed(1)
    scores = torch.randn(3, 5000, generator=generator)
    scores[1, 4000:] = float("-inf")
    visible = torch.tensor([5000, 4000, 2500], dtype=torch.int32)

    got = launch(
        kernel_source("index_select"), scores, visible, 512, torch_device
    )
    assert_kept("a long row", got, reference(scores, visible, 512))


def test_ties_keep_the_lower_columns(torch_device, kernel_source):
    generator = torch.Generator().manual_seed(2)
    # Four values over 1500 columns: the kept count falls inside a tie in
    # every row, and the ties span every chunk.
    scores = torch.randint(0, 4, (2, 1500), generator=generator).float()
    visible = torch.full((2,), 1500, dtype=torch.int32)

    got = launch(
        kernel_source("index_select"), scores, visible, 37, torch_device
    )
    assert_kept("ties", got, reference(scores, visible, 37))


def test_a_row_of_fewer_finite_scores_keeps_none_past_them(
    torch_device, kernel_source
):
    generator = torch.Generator().manual_seed(3)
    scores = torch.full((2, 700), float("-inf"))
    scores[0, :10] = torch.randn(10, generator=generator)
    visible = torch.tensor([10, 0], dtype=torch.int32)

    got = launch(
        kernel_source("index_select"), scores, visible, 64, torch_device
    )
    expected = reference(scores, visible, 64)
    assert_kept("fewer finite scores", got, expected)
    assert int((got[0] >= 0).sum()) == 10 and bool((got[1] == -1).all())


def test_the_block_mode_pins_the_newest_visible_block(
    torch_device, kernel_source
):
    generator = torch.Generator().manual_seed(4)
    # 1003 columns are 126 blocks of 8, the last one partly past the row; the
    # columns past each row's visible count are -inf, and the newest visible
    # block holds the row's smallest scores, so only the pin keeps it.
    scores = torch.randn(3, 1003, generator=generator)
    visible = torch.tensor([1003, 517, 9], dtype=torch.int32)
    for row in range(3):
        seen = int(visible[row])
        scores[row, seen:] = float("-inf")
        newest = (seen - 1) // 8 * 8
        scores[row, newest:seen] = -10.0

    got = launch(
        kernel_source("index_select"), scores, visible, 20, torch_device, pool=8
    )
    expected = reference(scores, visible, 20, pool=8)
    assert_kept("blocks", got, expected)
    for row in range(3):
        newest = (int(visible[row]) - 1) // 8
        assert newest in got[row].tolist(), f"row {row} lost its pinned block"


def test_candidate_columns_map_back_to_their_groups(
    torch_device, kernel_source
):
    generator = torch.Generator().manual_seed(5)
    block, count = 8, 100
    candidates = torch.randint(0, 5000, (2, count), generator=generator).to(
        torch.int32
    )
    candidates[:, ::7] = -1
    scores = torch.randn(2, count * block, generator=generator)
    # An unnamed block's columns score -inf, as the scoring leaves them.
    unnamed = (candidates < 0).repeat_interleave(block, dim=1)
    scores[unnamed] = float("-inf")
    visible = torch.full((2,), 40000, dtype=torch.int32)

    got = launch(
        kernel_source("index_select"),
        scores,
        visible,
        50,
        torch_device,
        candidates=candidates,
        block=block,
    )
    expected = reference(
        scores, visible, 50, candidates=candidates, block=block
    )
    assert_kept("candidates", got, expected)


def test_the_key_the_operator_prints_compiles_the_block_mode(
    torch_device, stated_launch, launch_target
):
    stated = stated_launch("index_select", key(8, 0, TILE), launch_target)
    assert tuple(stated.words) == WORDS, (
        f"the kernel takes {stated.words}, and the operator packs {WORDS}"
    )
    generator = torch.Generator().manual_seed(7)
    scores = torch.randn(2, 3001, generator=generator)
    visible = torch.tensor([3001, 1200], dtype=torch.int32)
    scores[1, 1200:] = float("-inf")

    got = launch(None, scores, visible, 40, torch_device, stated=stated)
    assert_kept("stated blocks", got, reference(scores, visible, 40, pool=8))


#: The model's selections: rows of decode steps over a context of 2^20
#: groups keeping 512, and the candidate source's 131072 blocks of 8 keeping
#: 2048, each over the operator's chunk of 4096 scores.
MODEL_ROWS = {
    "one row over 2^20 scores keeping 512": (1, 1, 512),
    "256 rows over 2^20 scores keeping 512": (256, 1, 512),
    "4 rows over 131072 blocks of 8 keeping 2048": (4, 8, 2048),
}


@pytest.mark.gpu
@pytest.mark.parametrize("name", MODEL_ROWS)
def test_the_model_selections(name, torch_device, stated_launch, launch_target):
    rows, pool, kept = MODEL_ROWS[name]
    stated = stated_launch("index_select", key(pool, 0, 4096), launch_target)
    columns = 1 << 20
    generator = torch.Generator().manual_seed(6)
    scores = torch.randn(rows, columns, generator=generator)
    visible = torch.randint(
        columns // 2, columns + 1, (rows,), generator=generator
    ).to(torch.int32)
    for row in range(rows):
        scores[row, int(visible[row]) :] = float("-inf")

    got = launch(
        None, scores, visible, kept, torch_device, pool=pool, stated=stated
    )
    assert_kept(name, got, reference(scores, visible, kept, pool=pool))
