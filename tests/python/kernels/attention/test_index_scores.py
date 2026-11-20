"""The index_scores program against the operator's formula.

The operator (`ops::attention::IndexScores`) scores the index keys of the
groups a token may select among::

    s[t, c] = sum_h relu(q[t, h] . key[g]) * w[t, h]     FP32

``w`` one for every head of an unweighted call, ``-inf`` for a column of no
visible group. A dense row's column ``c`` is group ``c``, visible below the
token's count; a row restricted to candidate blocks of ``B`` groups scores
group ``candidates[t, c // B] * B + c % B``, visible when the block is named
and the group below the count. Group ``g`` lies at slot ``g % groups`` of
page ``table[rows[t], g // groups]``. The keys are BF16 rows or MXFP4 records
in the page layout (every record's values, then every record's scales), and
the queries are read as they are or rounded to MXFP4 blocks, both by the
rules of ``crates/ops/src/record.rs``
(:mod:`uniserve_testing.kernels.records`).

The reference is the formula in FP64 over the decoded keys and the rounded
queries. Every value either side holds is exact in BF16, so the products are
exact and only the FP32 sums round: the operator's bound
(`IndexScores::tolerance`, `split_dot(heads · width, weighted, f32, f32)`,
relative to the row's largest score because the weighted sum cancels) holds
the scores, and a column of no visible group is ``-inf`` exactly.

BF16 scores are DeepSeek's indexer's arithmetic::

    s[t, c] = bf16(sum_h bf16(relu(bf16(q[t, h] . key[g])) * bf16(w[t, h])))

each head's FP32 product, its weight, its weighted value and the FP32 sum of
the heads rounded to BF16, nearest even. The reference rounds at the same
points, from the FP64 values through FP32, and the operator's BF16 bound adds
the format's roundings to the FP32 one: a BF16 spacing (``2u``,
``u = 2^-8``) for each of a head's two roundings, ``sqrt(heads)`` times, a
sum of ``heads`` terms, and a spacing for the score's own rounding.

The cases, each a property a defect would break: BF16 keys over a scattered
page table with rows of different visible counts; MXFP4 keys whose scales lie
after the page's values, weighted by heads of both signs, against MXFP4
queries; the rounding of the queries alone; candidate blocks among which some
are unnamed and some reach past the visible groups; and a head count below
the products' tile.
"""

import math

import pytest
import torch
from uniserve_testing.kernels import records

pytestmark = pytest.mark.unit

#: The unit roundoff of the FP32 accumulation.
ROUNDOFF = 2.0**-24

#: The cases: the keys' format, whether the heads are weighted, whether the
#: queries are rounded to MXFP4, the candidate blocks' groups (zero for a
#: dense row), the tokens, the index heads, the groups of a page, the pages a
#: row's table holds, the program's column block, the scores' type and the
#: seed.
CASES = {
    "bf16 keys over a scattered table, rows of different visible counts": (
        "bf16",
        False,
        False,
        0,
        5,
        32,
        16,
        6,
        32,
        "f32",
        1,
    ),
    "mxfp4 keys with the page's scales after its values, weighted heads": (
        "mxfp4",
        True,
        True,
        0,
        4,
        32,
        16,
        5,
        32,
        "f32",
        2,
    ),
    "bf16 keys against queries rounded to mxfp4": (
        "bf16",
        True,
        True,
        0,
        3,
        32,
        8,
        4,
        16,
        "f32",
        3,
    ),
    "candidate blocks, some unnamed and some past the visible groups": (
        "mxfp4",
        True,
        True,
        4,
        4,
        32,
        16,
        6,
        16,
        "f32",
        4,
    ),
    "bf16 scores of mxfp4 keys and queries, weighted heads": (
        "mxfp4",
        True,
        True,
        0,
        4,
        32,
        16,
        5,
        32,
        "bf16",
        8,
    ),
    "bf16 scores inside candidate blocks": (
        "mxfp4",
        True,
        True,
        4,
        4,
        32,
        16,
        6,
        16,
        "bf16",
        9,
    ),
    "four heads padded to the products' tile": (
        "bf16",
        False,
        False,
        0,
        3,
        4,
        16,
        3,
        16,
        "f32",
        5,
    ),
}

#: Index keys and queries of DeepSeek's indexer.
WIDTH = 128


#: The unit roundoff of BF16, and the spacing of its values relative to them.
BF16_ROUNDOFF = 2.0**-8
BF16_SPACING = 2.0 * BF16_ROUNDOFF


def gamma(reduction):
    """``γₙ`` of an ``n``-term FP32 reduction."""
    return reduction * ROUNDOFF / (1.0 - reduction * ROUNDOFF)


def tolerance(heads, weighted, scores="f32"):
    """`IndexScores::tolerance`'s relative factor: a dot product of
    ``heads · width`` terms, one rounding more per term when weighted; BF16
    scores add a spacing for each of a head's two roundings, ``sqrt(heads)``
    times, the heads' FP32 sum and a spacing for the score's rounding.
    """  # noqa: D205
    terms = heads * WIDTH
    products = 2.0 * gamma(terms + int(weighted)) * math.sqrt(terms) + ROUNDOFF
    if scores == "f32":
        return products
    spread = math.sqrt(heads)
    return (
        products
        + 2.0 * BF16_SPACING * spread
        + 2.0 * gamma(heads) * spread
        + BF16_SPACING
    )


def bf16(values):
    """FP64 ``values`` rounded through FP32 to BF16, nearest even, as FP64:
    the program's FP32 result rounded to BF16.
    """  # noqa: D205
    return values.to(torch.float32).to(torch.bfloat16).to(torch.float64)


def operands(case, generator):
    """The call's operands and the reference's values: the queries, the
    weights, the key plane and its decoded keys, the table, the rows, the
    visible counts and the candidate blocks.
    """  # noqa: D205
    keys_format, weighted, _, block, tokens, heads, groups, max_pages = case[:8]
    pages = max_pages + 3
    table_rows = tokens + 1

    # Each table row is its own permutation of the pages, and a token reads
    # the row after its own, so a program that read the token as its row
    # scores another row's keys.
    table = torch.stack(
        [
            torch.randperm(pages, generator=generator)[:max_pages]
            for _ in range(table_rows)
        ]
    ).to(torch.int32)
    rows = (torch.arange(tokens, dtype=torch.int32) + 1) % table_rows
    columns_of_table = max_pages * groups
    visible = torch.randint(
        1, columns_of_table + 1, (tokens,), generator=generator
    ).to(torch.int32)
    visible[0] = columns_of_table

    logical = torch.randn(pages * groups, WIDTH, generator=generator)
    values, scales = records.encode(keys_format, logical)
    plane = records.plane(keys_format, "page", values, scales, groups)
    decoded = records.decode(keys_format, values, scales, WIDTH).reshape(
        pages, groups, WIDTH
    )

    query = torch.randn(tokens, heads, WIDTH, generator=generator).to(
        torch.bfloat16
    )
    weights = (
        torch.randn(tokens, heads, generator=generator)
        if weighted
        else torch.ones(tokens, heads)
    )

    candidates = None
    if block:
        # Blocks within the table, a few unnamed, and the last of each row
        # past its visible groups.
        count = columns_of_table // block // 2
        candidates = torch.randint(
            0, columns_of_table // block, (tokens, count), generator=generator
        ).to(torch.int32)
        candidates[:, 1] = -1
        candidates[:, -1] = columns_of_table // block - 1
        visible[:] = visible.clamp(max=columns_of_table - block)
    return query, weights, plane, decoded, table, rows, visible, candidates


def reference(case, query, weights, decoded, table, rows, visible, candidates):
    """The scores the operator states, in FP64, or rounded where BF16 scores
    round.
    """  # noqa: D205
    _, _, rounded, block, tokens, _, groups, max_pages, _, scores_type, _ = case
    round_bf16 = bf16 if scores_type == "bf16" else (lambda values: values)
    weights = round_bf16(weights.to(torch.float64))
    q = query.to(torch.float32)
    if rounded:
        q = records.fake_quantize("mxfp4", q)
    q = q.to(torch.float64)
    weights = weights.to(torch.float64)
    keys = decoded.to(torch.float64)

    if block:
        count = candidates.shape[1]
        column = torch.arange(count * block)
        named = candidates[:, column // block]
        group = named * block + column % block
        live = named >= 0
    else:
        group = torch.arange(max_pages * groups).expand(tokens, -1)
        live = torch.ones_like(group, dtype=torch.bool)
    live = live & (group < visible.unsqueeze(1)) & (group < max_pages * groups)
    group = torch.where(live, group, 0)

    out = torch.full(group.shape, float("-inf"), dtype=torch.float64)
    for token in range(tokens):
        pages = table[rows[token]][group[token] // groups].long()
        token_keys = keys[pages, group[token] % groups]
        products = round_bf16(q[token] @ token_keys.T).clamp_min(0.0)
        weighted = round_bf16(products * weights[token].unsqueeze(1))
        scores = round_bf16(weighted.sum(0))
        out[token] = torch.where(live[token], scores, float("-inf"))
    return out


#: The words `IndexScores::launch` packs, in its order.
WORDS = (
    "query",
    "weights",
    "keys",
    "page_table",
    "rows",
    "visible",
    "candidates",
    "out",
    "tokens",
    "max_pages",
    "blocks",
)


def key(case):
    """The descriptor key `IndexScoresDescriptor` prints for ``case``: the
    weighting, the query's rounding and the candidates only when set.
    """  # noqa: D205
    keys_format, weighted, rounded, block, _, heads, groups, _, column = case[
        :9
    ]
    text = (
        f"dtype_bf16_keys_{keys_format}_heads_{heads}_width_{WIDTH}_"
        f"groups_{groups}_block_{column}"
    )
    text += "_weighted_true" if weighted else ""
    text += "_query_mxfp4" if rounded else ""
    text += f"_candidates_{block}" if block else ""
    return text + ("_scores_bf16" if case[9] == "bf16" else "")


def launch(kernels, case, operands, device, stated=None):
    """One launch: a program per block of columns of one token, compiled as
    the case states it, or as ``stated``, the compilation the package's
    launch table states for the case's key.
    """  # noqa: D205
    keys_format, weighted, rounded, block, tokens, heads, groups, max_pages = (
        case[:8]
    )
    column_block = case[8]
    query, weights, plane, _, table, rows, visible, candidates = operands
    unread = torch.zeros(1, dtype=torch.int32, device=device)
    blocks = 0 if candidates is None else candidates.shape[1]
    columns = blocks * block if block else max_pages * groups
    dtype = torch.bfloat16 if case[9] == "bf16" else torch.float32
    out = torch.full(
        (tokens, columns), float("nan"), dtype=dtype, device=device
    )
    words = {
        "query": query.to(device),
        "weights": weights.to(torch.float32).to(device) if weighted else unread,
        "keys": plane.to(device),
        "page_table": table.to(device),
        "rows": rows.to(device),
        "visible": visible.to(device),
        "candidates": unread if candidates is None else candidates.to(device),
        "out": out,
        "tokens": tokens,
        "max_pages": max_pages,
        "blocks": blocks,
    }
    constants = {
        "HEADS": heads,
        "HEADS_TILE": max(16, 1 << (heads - 1).bit_length()),
        "WIDTH": WIDTH,
        "GROUPS": groups,
        "BLOCK": column_block,
        "KEYS_MXFP4": keys_format == "mxfp4",
        "WEIGHTED": weighted,
        "QUERY_MXFP4": rounded,
        "CANDIDATES": block,
        "SCORES_BF16": case[9] == "bf16",
    }
    if stated is None:
        kernel = kernels._index_scores_kernel
    else:
        kernel, constants = stated.kernel, stated.constexprs
    grid = ((columns + column_block - 1) // column_block, tokens, 1)
    kernel[grid](*(words[name] for name in WORDS), **constants)
    return out.cpu().to(torch.float32)


def check(where, case, got, expected):
    """Hold ``got`` to ``expected``: ``-inf`` exactly where the reference
    has it, the operator's bound over the largest score elsewhere.
    """  # noqa: D205
    hidden = torch.isinf(expected)
    assert torch.equal(torch.isneginf(got), hidden), (
        f"{where}: {int((torch.isneginf(got) != hidden).sum())} columns "
        f"disagree on being visible"
    )
    seen = ~hidden
    peak = float(expected[seen].abs().max()) if seen.any() else 0.0
    bound = 2.0**-126 + tolerance(case[5], case[1], case[9]) * peak
    error = (got.to(torch.float64) - expected).abs()[seen]
    worst = float(error.max()) if error.numel() else 0.0
    assert worst <= bound, (
        f"{where}: a score {worst:.3e} from the reference, bound {bound:.3e} "
        f"over a peak of {peak:.3e}"
    )


@pytest.mark.parametrize("name", CASES)
def test_every_column_scores_its_visible_group(
    name, torch_device, kernel_source
):
    case = CASES[name]
    generator = torch.Generator().manual_seed(case[-1])
    built = operands(case, generator)
    query, weights, _, decoded, table, rows, visible, candidates = built

    got = launch(kernel_source("index_scores"), case, built, torch_device)
    expected = reference(
        case, query, weights, decoded, table, rows, visible, candidates
    )
    check(name, case, got, expected)


@pytest.mark.parametrize("name", CASES)
def test_the_key_the_operator_prints_compiles_the_case(
    name, torch_device, kernel_source, stated_launch, launch_target
):
    case = CASES[name]
    stated = stated_launch("index_scores", key(case), launch_target)
    assert tuple(stated.words) == WORDS, (
        f"{name}: the kernel takes {stated.words}, and the operator packs "
        f"{WORDS}"
    )
    generator = torch.Generator().manual_seed(case[-1])
    built = operands(case, generator)
    query, weights, _, decoded, table, rows, visible, candidates = built

    got = launch(None, case, built, torch_device, stated)
    expected = reference(
        case, query, weights, decoded, table, rows, visible, candidates
    )
    check(name, case, got, expected)


#: Decode rows at the model's shapes: 32 heads of 128 over MXFP4 keys of 64
#: groups a page, weighted and rounded, in the model's BF16 scores: the rows
#: of a step against a short context, a few rows against a context of 2^20
#: groups, and rows restricted to 2048 candidate blocks of 8 groups, as the
#: indexers after a candidate source score: `(tokens, max_pages, candidate
#: block)`.
MODEL_ROWS = {
    "256 rows over 4096 groups": (256, 64, 0),
    "two rows over 2^20 groups": (2, 16384, 0),
    "32 rows inside 2048 candidate blocks of 8": (32, 512, 8),
}


@pytest.mark.gpu
@pytest.mark.parametrize("name", MODEL_ROWS)
def test_the_model_shapes_score_within_the_bound(
    name, torch_device, stated_launch, launch_target
):
    tokens, max_pages, block = MODEL_ROWS[name]
    case = (
        "mxfp4",
        True,
        True,
        block,
        tokens,
        32,
        64,
        max_pages,
        256,
        "bf16",
        6,
    )
    stated = stated_launch("index_scores", key(case), launch_target)
    generator = torch.Generator().manual_seed(case[-1])
    built = operands(case, generator)
    query, weights, _, decoded, table, rows, visible, candidates = built

    got = launch(None, case, built, torch_device, stated)
    expected = reference(
        case, query, weights, decoded, table, rows, visible, candidates
    )
    check(name, case, got, expected)
