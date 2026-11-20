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
mapped back to their groups; and BF16 scores, which are ranked by their FP32
values and tie far more often. The device cases are the model's: decode rows
over 2^20 scores keeping 512, and 131072 blocks of 8 keeping 2048.

The CUDA module (``index_select/cuda``, resolved at the session's Build
Server) is called with the operator's call structure
(``uniserve_index_select_call``) over the same rule: the cases above, rows a
cluster splits (2^20 scores, one row and many), a row whose threshold lies
in a run of keys too long for the module's shared memory, and the model's
selections, each in FP32 and BF16 scores.
"""

import ctypes

import pytest
import torch
from uniserve_testing.kernels.packages import abi_enumerator, assert_abi_layout

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


def key(pool, block, tile, scores="f32"):
    """The descriptor key `IndexSelectDescriptor` prints: the scores' type
    only for BF16 scores.
    """  # noqa: D205
    text = f"pool_{pool}_candidates_{block}_tile_{tile}"
    return text + ("_scores_bf16" if scores == "bf16" else "")


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


def test_bf16_scores_rank_by_their_values_and_ties_keep_the_lower_columns(
    torch_device, stated_launch, launch_target
):
    stated = stated_launch(
        "index_select", key(1, 0, TILE, "bf16"), launch_target
    )
    generator = torch.Generator().manual_seed(8)
    # Scores of a few hundred BF16 values over 3000 columns: the kept
    # count falls inside a tie, and a -inf tail past the second row's count.
    scores = (torch.randn(2, 3000, generator=generator) * 3).to(torch.bfloat16)
    visible = torch.tensor([3000, 2100], dtype=torch.int32)
    scores[1, 2100:] = float("-inf")

    got = launch(None, scores, visible, 300, torch_device, stated=stated)
    assert_kept(
        "bf16 scores", got, reference(scores.to(torch.float32), visible, 300)
    )


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


class SelectCall(ctypes.Structure):
    """``uniserve_index_select_call`` (``ops::attention::SelectCall``), field
    for field.
    """  # noqa: D205

    _fields_ = [
        ("scores", ctypes.c_void_p),
        ("visible", ctypes.c_void_p),
        ("candidates", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("tokens", ctypes.c_int64),
        ("columns", ctypes.c_int64),
        ("kept", ctypes.c_int64),
        ("pool", ctypes.c_int64),
        ("candidate_block", ctypes.c_int64),
        ("scores_dtype", ctypes.c_int64),
        ("multiprocessors", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


def test_the_call_mirror_is_the_headers_call_structure():
    """The generated header asserts the size and every offset it holds."""
    assert_abi_layout(SelectCall, "uniserve_index_select_call")


def select_module(
    kernel_module, scores, visible, kept, pool=1, candidates=None, block=0
):
    """One call of the CUDA module of the call's key: the kept entries."""
    device = torch.device("cuda")
    dtype = "bf16" if scores.dtype == torch.bfloat16 else "f32"
    library = ctypes.CDLL(
        str(
            kernel_module("index_select", "cuda", key(pool, block, 4096, dtype))
            / "module.so"
        )
    )
    function = library.uniserve_index_select
    function.argtypes = [ctypes.POINTER(SelectCall), ctypes.c_void_p]
    function.restype = ctypes.c_int32

    rows, columns = scores.shape
    scores, visible = scores.to(device), visible.to(device)
    named = None if candidates is None else candidates.to(device)
    out = torch.full((rows, kept), SENTINEL, dtype=torch.int32, device=device)
    call = SelectCall(
        scores=scores.data_ptr(),
        visible=visible.data_ptr(),
        candidates=None if named is None else named.data_ptr(),
        out=out.data_ptr(),
        tokens=rows,
        columns=columns,
        kept=kept,
        pool=pool,
        candidate_block=block,
        scores_dtype=abi_enumerator(f"UNISERVE_DTYPE_{dtype.upper()}"),
        multiprocessors=torch.cuda.get_device_properties(
            device
        ).multi_processor_count,
    )
    stream = torch.cuda.current_stream(device).cuda_stream
    status = function(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(device)
    assert status == 0, (
        f"the module refused: {call.error.decode(errors='replace')}"
    )
    return out.cpu()


def long_rows(rows, columns, seed, spread=1.0):
    """Rows of ``columns`` scores around one of ``spread`` and visible counts
    of at least half of them, the first row's every column.
    """  # noqa: D205
    generator = torch.Generator().manual_seed(seed)
    scores = 1.0 + torch.randn(rows, columns, generator=generator) * spread
    visible = torch.randint(
        columns // 2, columns + 1, (rows,), generator=generator
    ).to(torch.int32)
    visible[0] = columns
    return scores, visible


def ties(seed):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randint(0, 4, (2, 1500), generator=generator).float()
    return scores, torch.full((2,), 1500, dtype=torch.int32), 37, {}


def fewer_finite(seed):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.full((2, 700), float("-inf"))
    scores[0, :10] = torch.randn(10, generator=generator)
    return scores, torch.tensor([10, 0], dtype=torch.int32), 64, {}


def blocks(seed):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(3, 1003, generator=generator)
    visible = torch.tensor([1003, 517, 9], dtype=torch.int32)
    for row in range(3):
        seen = int(visible[row])
        scores[row, seen:] = float("-inf")
        scores[row, (seen - 1) // 8 * 8 : seen] = -10.0
    return scores, visible, 20, {"pool": 8}


def candidate_blocks(seed):
    generator = torch.Generator().manual_seed(seed)
    candidates = torch.randint(0, 5000, (2, 100), generator=generator).to(
        torch.int32
    )
    candidates[:, ::7] = -1
    scores = torch.randn(2, 800, generator=generator)
    scores[(candidates < 0).repeat_interleave(8, dim=1)] = float("-inf")
    visible = torch.full((2,), 40000, dtype=torch.int32)
    return scores, visible, 50, {"candidates": candidates, "block": 8}


def split_rows(seed):
    scores, visible = long_rows(8, 1 << 20, seed)
    return scores, visible, 512, {}


def one_split_row(seed):
    scores, visible = long_rows(1, 1 << 20, seed)
    return scores, visible, 512, {}


def a_long_tie_run(seed):
    # Scores within a few thousandths of one: the run of keys around the
    # threshold outgrows what a CTA keeps, so the module reads the row again.
    scores, visible = long_rows(1, 1 << 20, seed, spread=1e-3)
    return scores, visible, 512, {}


def candidate_source(seed):
    scores, visible = long_rows(4, 1 << 20, seed)
    for row in range(4):
        scores[row, int(visible[row]) :] = float("-inf")
    return scores, visible, 2048, {"pool": 8}


def prefill_rows(seed):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(512, 8192, generator=generator)
    visible = (torch.arange(512, dtype=torch.int32) * 16 + 1).to(torch.int32)
    for row in range(512):
        scores[row, int(visible[row]) :] = float("-inf")
    return scores, visible, 512, {}


#: The CUDA module's cases: each builds FP32 scores, the visible counts, the
#: kept count and the call's pool and candidates.
MODULE_CASES = {
    "ties keep the lower columns": ties,
    "fewer finite scores than kept": fewer_finite,
    "blocks of 8 pin the newest": blocks,
    "candidate columns map back to their groups": candidate_blocks,
    "8 rows of 2^20 scores, split over clusters": split_rows,
    "one row of 2^20 scores": one_split_row,
    "a long run of keys around the threshold": a_long_tie_run,
    "the candidate source's 131072 blocks of 8 keeping 2048": candidate_source,
    "512 prefill rows of growing visible counts": prefill_rows,
}


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"]
)
@pytest.mark.parametrize("name", MODULE_CASES)
def test_the_cuda_module_keeps_what_the_rule_keeps(name, dtype, kernel_module):
    scores, visible, kept, options = MODULE_CASES[name](9)
    scores = scores.to(dtype)

    got = select_module(kernel_module, scores, visible, kept, **options)
    expected = reference(scores.to(torch.float32), visible, kept, **options)
    assert_kept(name, got, expected)
