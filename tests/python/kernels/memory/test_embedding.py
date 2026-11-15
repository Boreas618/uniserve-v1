"""The embedding program copies the rows an id table selects, bit for bit.

``embedding`` computes ``out[token] = weight[id - first]`` over
``[tokens, hidden]``, where the weight table holds this rank's ``rows`` rows
starting at the vocabulary offset ``first``. **An id outside ``[first, first +
rows)`` writes zeros**: a token whose embedding lives on another rank
contributes nothing, and a row of another rank's values would be a silent
aliasing rather than an absent embedding. The tolerance is exact: a lookup
copies values and rounds nothing.

The cases: every id inside the shard; a shard offset with ids on both sides of
it, so the zeros sit next to real rows; one row serving several tokens, which
a program walking its rows instead of its ids fails; and a ``hidden`` the
inner tile does not divide, so the column tail is masked. Each runs with the
id types the operator takes (``i32``; ``u32``, as the model library passes
them; ``i64``, whose arithmetic is wider than the table) and the shard sizes
the check states (16 and 32 rows). The shard is small on purpose: the
operator is the same at 151,936 rows.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

#: The element types the table's `dtype` takes, as the check's `--dtype`
#: offered them.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

ID_TYPES = {"u32": torch.uint32, "i32": torch.int32, "i64": torch.int64}

#: The columns one inner tile copies: the package's `tile` parameter.
TILE = 256

#: The token rows one program copies, the descriptor's `block`.
BLOCK = 8

HIDDEN = 2048
TOKENS = 5
FIRST = 100

#: The cases: the vocabulary offset, whether ids fall outside the shard,
#: whether one row serves several tokens, the hidden width, the token count
#: and the seed.
CASES = {
    "every id inside the shard": (0, False, False, HIDDEN, TOKENS, 1),
    # The zeros are as much the operator's output as the copied rows.
    "a shard's offset, and ids outside it": (
        FIRST,
        True,
        False,
        HIDDEN,
        TOKENS,
        2,
    ),
    "one row serving several tokens": (
        FIRST,
        False,
        True,
        HIDDEN,
        max(TOKENS, 4),
        3,
    ),
    "a column tail the tile does not cover": (
        0,
        False,
        False,
        TILE + 100,
        TOKENS,
        4,
    ),
}

#: The id types and shard sizes the cases run with.
OPTIONS = {
    "i32 ids, 16 rows": ("i32", 16),
    "u32 ids, 16 rows": ("u32", 16),
    "i64 ids, 16 rows": ("i64", 16),
    "i32 ids, 32 rows": ("i32", 32),
}


def launch(kernels, ids, weight, out, first):
    """One launch: a program per `BLOCK` tokens."""
    tokens, hidden = out.shape
    kernels._embedding_kernel[(-(-tokens // BLOCK),)](
        ids,
        weight,
        out,
        tokens,
        hidden,
        first,
        weight.shape[0],
        BLOCK=BLOCK,
        TILE=TILE,
        PDL=False,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("case", CASES)
def test_every_token_takes_its_row_or_zeros(
    case, option, dtype, torch_device, kernel_source
):
    first, outside, duplicate, hidden, tokens, seed = CASES[case]
    id_type, rows = OPTIONS[option]
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, hidden, generator=generator).to(DTYPES[dtype])
    ids = torch.full((tokens,), first, dtype=torch.int64)
    if outside:
        # Ids below the shard and above it: every one of them writes zeros.
        ids[0] = first - 1 if first > 0 else 0
        ids[-1] = first + rows
        ids[len(ids) // 2] = first + rows + 5
    if duplicate:
        ids[1] = ids[0]
    out = torch.full(
        (tokens, hidden),
        float("nan"),
        dtype=DTYPES[dtype],
        device=torch_device,
    )

    launch(
        kernel_source("embedding"),
        ids.to(ID_TYPES[id_type]).to(torch_device),
        weight.to(torch_device),
        out,
        first,
    )

    local = ids - first
    inside = (local >= 0) & (local < rows)
    expected = torch.zeros(tokens, hidden, dtype=torch.float64)
    expected[inside] = weight.double()[local[inside]]
    got = out.cpu().double()
    wrong = (got != expected).nonzero()
    detail = "; ".join(
        f"[{token}, {column}]: got {got[token, column].item():.9e}, the table "
        f"says {expected[token, column].item():.9e} (id {ids[token].item()})"
        for token, column in wrong[:3].tolist()
    )
    assert wrong.numel() == 0, (
        f"{case}, {option}, {dtype}: {tokens} tokens of hidden {hidden} "
        f"from {rows} rows at first {first}: {wrong.size(0)} of "
        f"{expected.numel()} "
        f"elements differ; {detail}"
    )
