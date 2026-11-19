"""The latent_attention program against the operator's formula.

The operator (``ops::attention::LatentAttention``) attends one shared latent,
both key and value, over each query's window entries and the compressed
entries its selection names, in one softmax with a per-head sink::

    entries(t) = window positions max(0, p - W + 1) ..= p, then the
                 compressed entries selection[t, :] names (>= 0)
    s          = scale * q . c
    m          = max(-1e30, max s)
    o          = sum e^(s - m) c / (sum e^(s - m) + e^(sink - m))

The reference is that formula in FP64 over the latents the records decode to
(:mod:`uniserve_testing.kernels.records`), read from each request's logical
entries rather than through the pages: a program that addressed a page, a slot
or a ring column wrongly reads another entry, and every record no entry names
holds a trap value far from any latent. The bound (:func:`bound`) is the
operator's terms with each element's own magnitude in place of the declared
latent envelope: the FP32 reduction, each probability's rounding into BF16
and the output's rounding.

The cases, each a property a defect of this family breaks:

- **every record format** of each side: a BF16 and an MXFP8 window; BF16,
  MXFP8, NVFP4 and MXFP4 compressed entries;
- **a window-only layer**, compiled without the compressed side;
- **windows that start at zero**: contexts shorter than the window;
- **windows held from a row's start**, as a request resumed from a prefix hit
  holds them: entries before the start are never read;
- **a ring of window pages** narrower than the request, read modulo its width;
- **negative selection slots** among the named ones, and rows that see no
  compressed entry yet;
- **a padding token** (position -1, an empty selection), which reads no entry
  and attends its sink alone: zeros;
- **64 heads in blocks of 16**, and **8 heads**, a block below the sixteen rows
  a tensor-core product needs;
- **decode-like batches** of a few tokens per request, and **a prefill-like
  row** of many tokens of one request, each attending its own causal window.

and, at the model's shapes (512-wide latents, 64 heads, 128 window and 512
selected entries), a decode of 48 requests of 1 to 6 tokens and a prefill
chunk of 512 tokens.

Both of the program's products run on the tensor cores over BF16 operands,
which Triton's interpreter multiplies as the integers it stores them in; so
every case needs a device.
"""

import dataclasses
import math

import pytest
import torch
from uniserve_testing.kernels import records

pytestmark = pytest.mark.unit

#: The largest magnitude a cached latent is declared to have
#: (``attention::latent::LATENT_ENVELOPE``).
LATENT_ENVELOPE = 64.0

#: The unit roundoffs of the FP32 accumulation and of the BF16 output.
F32_UNIT = 2.0**-24
BF16_UNIT = 2.0**-8

#: The latent a record no entry names holds: far from every drawn latent, so a
#: program that reads it answers far outside the bound.
TRAP = 1000.0


@dataclasses.dataclass(frozen=True)
class Case:
    """One case: the heads, the heads one program attends and the latent's
    width; the window, its page and its format; the compressed side's format
    (``None`` for a window-only layer), its page and the selection's width;
    each request's context length and query tokens; whether the window table
    is a ring narrower than the requests; whether a padding token follows;
    and the seed.
    """  # noqa: D205

    heads: int
    block: int
    head_dim: int
    window: int
    page: int
    window_format: str
    compressed: str | None
    entries: int
    columns: int
    requests: tuple[tuple[int, int], ...]
    ring: bool = False
    padding: bool = False
    seed: int = 0
    #: Each request's first held window position, as after a replay from a
    #: prefix hit; empty for requests that computed every token.
    starts: tuple[int, ...] = ()


#: Small shapes, where every edge a row meets fits a few tiles: a latent of
#: 64, a window of 16 in pages of 8, compressed pages of 4 and selections of
#: 8.
SMALL = {"head_dim": 64, "window": 16, "page": 8, "entries": 4, "columns": 8}

CASES = {
    "a window-only layer over BF16 windows that start at zero": Case(
        heads=16,
        block=16,
        window_format="bf16",
        compressed=None,
        requests=((5, 1), (12, 2), (30, 3)),
        padding=True,
        seed=1,
        **SMALL,
    ),
    "an MXFP8 window over a ring of pages narrower than the requests": Case(
        heads=16,
        block=16,
        window_format="mxfp8",
        compressed=None,
        requests=((41, 1), (57, 2), (33, 1), (64, 3)),
        ring=True,
        seed=2,
        **SMALL,
    ),
    "an MXFP8 window and BF16 entries with negative selection slots": Case(
        heads=16,
        block=16,
        window_format="mxfp8",
        compressed="bf16",
        requests=((3, 1), (26, 2), (40, 1)),
        padding=True,
        seed=3,
        **SMALL,
    ),
    "windows held from a replay's start, read from it on": Case(
        heads=16,
        block=16,
        window_format="mxfp8",
        compressed="nvfp4",
        requests=((40, 4), (20, 2), (12, 1)),
        starts=(34, 0, 12),
        seed=8,
        **SMALL,
    ),
    "MXFP8 entries over 64 heads in blocks of 16": Case(
        heads=64,
        block=16,
        window_format="mxfp8",
        compressed="mxfp8",
        requests=((22, 2), (35, 1)),
        seed=4,
        **SMALL,
    ),
    "NVFP4 entries and a prefill row of many tokens of one request": Case(
        heads=16,
        block=16,
        window_format="mxfp8",
        compressed="nvfp4",
        requests=((48, 40),),
        seed=5,
        **SMALL,
    ),
    "NVFP4 entries over 8 heads, a block below a product's 16 rows": Case(
        heads=8,
        block=8,
        window_format="bf16",
        compressed="nvfp4",
        requests=((19, 3), (44, 2)),
        seed=6,
        **SMALL,
    ),
    "MXFP4 entries under a BF16 window": Case(
        heads=16,
        block=16,
        window_format="bf16",
        compressed="mxfp4",
        requests=((31, 2), (52, 1)),
        seed=7,
        **SMALL,
    ),
}

#: The model's shapes: 64 heads of 512 in blocks of 16, a window of 128 in
#: pages of 64, and 512 selected entries.
MODEL = {"heads": 64, "block": 16, "head_dim": 512, "window": 128, "page": 64}

MODEL_CASES = {
    "the model's decode of 1 to 6 tokens per request over NVFP4 entries": Case(
        window_format="mxfp8",
        compressed="nvfp4",
        entries=32,
        columns=512,
        requests=tuple(
            (600 + 47 * index, 1 + index % 6) for index in range(48)
        ),
        seed=11,
        **MODEL,
    ),
    "the model's prefill chunk of 512 tokens over MXFP8 entries": Case(
        window_format="mxfp8",
        compressed="mxfp8",
        entries=64,
        columns=512,
        requests=((2048, 512),),
        seed=12,
        **MODEL,
    ),
    "the model's window-only layer, decode over a BF16 window": Case(
        window_format="bf16",
        compressed=None,
        entries=1,
        columns=1,
        requests=tuple(
            (130 + 61 * index, 1 + index % 4) for index in range(32)
        ),
        seed=13,
        **MODEL,
    ),
}


def bound(case, magnitude, expected, interpreted):
    """The allowed error of each output element.

    The output is a convex combination of the latents it reads, and the
    bound of paged attention applies with the combination's own magnitude
    ``S = Σ p |c| / Σ p`` (``magnitude``, at most ``LATENT_ENVELOPE``, the
    operator's declaration) in place of the largest latent: the FP32
    reduction over the ``n`` entries one softmax reads costs ``(2γₙ + u) · S``,
    each probability's rounding into BF16 a relative ``u_bf16`` of its term,
    so ``u_bf16 · S`` in all, and the output's rounding ``u_bf16 · |o|``. On
    the interpreter a BF16 conversion truncates, twice the unit.
    """  # noqa: D205
    reduction = case.window + (case.columns if case.compressed else 0)
    terms = reduction * F32_UNIT
    gamma = terms / (1.0 - terms)
    unit = 2.0 * BF16_UNIT if interpreted else BF16_UNIT
    return (2.0 * gamma + F32_UNIT + unit) * magnitude + unit * expected.abs()


def key(case):
    """The descriptor key ``LatentAttentionDescriptor`` prints for ``case``."""
    head = (
        f"dtype_bf16_heads_{case.heads}_head_dim_{case.head_dim}_window_format_"
        f"{case.window_format}_window_{case.window}_page_{case.page}"
    )
    if case.compressed is None:
        side = "compressed_none"
    else:
        side = (
            f"compressed_{case.compressed}_entries_{case.entries}_columns_"
            f"{case.columns}"
        )
    return f"{head}_{side}_block_{case.block}"


@dataclasses.dataclass
class Plane:
    """A paged plane: its pages, the table of each request's pages, and each
    request's decoded latents in its own order.
    """  # noqa: D205

    pages: torch.Tensor
    table: torch.Tensor
    latents: list[torch.Tensor]


def paged(fmt, latents, per_page, columns, ring, generator):
    """The plane of ``fmt`` records holding each request's ``latents`` at
    page ``table[request, (e // per_page) % columns]``, slot ``e % per_page``.

    Every request's table columns are their own pages, in a shuffled order so
    that a program assuming the identity reads another request's. A ring
    holds each column's latest page, the request's last ``columns`` pages.
    Every record no entry lands on holds :data:`TRAP`.
    """  # noqa: D205
    width = latents[0].shape[-1]
    count = len(latents) * columns
    physical = torch.randperm(count, generator=generator).reshape(-1, columns)

    trap_values, trap_scales = records.encode(fmt, torch.full((1, width), TRAP))
    values = trap_values.repeat(count * per_page, 1)
    scales = trap_scales.repeat(count * per_page, 1)
    decoded = []
    for request, latent in enumerate(latents):
        entries = latent.shape[0]
        last = (entries - 1) // per_page
        if not ring:
            assert last < columns
        # The entries of the pages the table still maps, each slot once.
        entry = torch.arange(max(last - columns + 1, 0) * per_page, entries)
        page = physical[request, (entry // per_page) % columns]
        slot = page * per_page + entry % per_page
        encoded, encoded_scales = records.encode(fmt, latent[entry])
        values[slot] = encoded
        scales[slot] = encoded_scales
        decoded.append(records.fake_quantize(fmt, latent).double())
    pages = records.plane(fmt, "entry", values, scales, per_page)
    return Plane(pages, physical.to(torch.int32), decoded)


@dataclasses.dataclass
class Batch:
    """A case's operands on the host and the reference's view of them."""

    query: torch.Tensor
    window: Plane
    starts: torch.Tensor
    compressed: Plane | None
    selection: torch.Tensor
    rows: torch.Tensor
    positions: torch.Tensor
    sink: torch.Tensor
    scale: float


def batch(case):
    """The operands of ``case``: each request's tokens are its last query
    positions, a padding token (position -1, row 0, an empty selection)
    follows when the case has one, and each token's selection names a
    random subset of the compressed entries complete before it (one per two
    tokens) in random slots, the others -1.
    """  # noqa: D205
    generator = torch.Generator().manual_seed(case.seed)
    width = case.head_dim
    contexts = [context for context, _ in case.requests]

    def drawn(count):
        return torch.randn(count, width, generator=generator)

    longest = max(contexts)
    if case.ring:
        # The newest pages a decode's window reaches, and one more.
        columns = case.window // case.page + 2
    else:
        columns = math.ceil(longest / case.page)
    window = paged(
        case.window_format,
        [drawn(context) for context in contexts],
        case.page,
        columns,
        case.ring,
        generator,
    )

    compressed = None
    if case.compressed is not None:
        counts = [max(context // 2, 1) for context in contexts]
        compressed = paged(
            case.compressed,
            [drawn(count) for count in counts],
            case.entries,
            math.ceil(max(counts) / case.entries),
            False,
            generator,
        )

    rows, positions, selection = [], [], []
    for request, (context, tokens) in enumerate(case.requests):
        for position in range(context - tokens, context):
            rows.append(request)
            positions.append(position)
            slots = torch.full((case.columns,), -1, dtype=torch.int32)
            visible = (position + 1) // 2
            if compressed is not None and visible > 0:
                named = int(
                    torch.randint(
                        0,
                        min(visible, case.columns) + 1,
                        (),
                        generator=generator,
                    )
                )
                chosen = torch.randperm(visible, generator=generator)[:named]
                where = torch.randperm(case.columns, generator=generator)[
                    :named
                ]
                slots[where] = chosen.to(torch.int32)
            selection.append(slots)
    if case.padding:
        rows.append(0)
        positions.append(-1)
        selection.append(torch.full((case.columns,), -1, dtype=torch.int32))

    tokens = len(rows)
    query = torch.randn(tokens, case.heads, width, generator=generator)
    # Sinks on both sides of the scores, one far above them.
    sink = torch.rand(case.heads, generator=generator) * 6.0 - 3.0
    sink[0] = 8.0
    starts = case.starts or (0,) * len(case.requests)
    return Batch(
        query=query.to(torch.bfloat16),
        window=window,
        starts=torch.tensor(starts, dtype=torch.int32),
        compressed=compressed,
        selection=torch.stack(selection),
        rows=torch.tensor(rows, dtype=torch.int32),
        positions=torch.tensor(positions, dtype=torch.int32),
        sink=sink,
        scale=width**-0.5,
    )


def reference(case, operands):
    """The operator's formula in FP64, token by token, over the decoded
    latents in each request's own order, and the magnitude of each output
    element's combination, ``Σ p |c| / (Σ p + e^(sink - m))``.
    """  # noqa: D205
    out = torch.zeros(operands.query.shape, dtype=torch.float64)
    magnitude = torch.zeros_like(out)
    sink = operands.sink.double()
    for token in range(operands.query.shape[0]):
        request = int(operands.rows[token])
        position = int(operands.positions[token])
        parts = []
        if position >= 0:
            held = int(operands.starts[request])
            first = max(position - case.window + 1, held)
            parts.append(operands.window.latents[request][first : position + 1])
        if operands.compressed is not None:
            named = operands.selection[token]
            named = named[named >= 0].long()
            parts.append(operands.compressed.latents[request][named])
        latents = torch.cat(parts) if parts else torch.zeros(0, case.head_dim)
        latents = latents.double()

        query = operands.query[token].double()
        scores = query @ latents.T * operands.scale
        peak = torch.full((case.heads,), -1e30, dtype=torch.float64)
        if latents.shape[0] > 0:
            peak = torch.maximum(peak, scores.max(dim=1).values)
        weights = torch.exp(scores - peak[:, None])
        total = weights.sum(dim=1) + torch.exp(sink - peak)
        out[token] = weights @ latents / total[:, None]
        magnitude[token] = weights @ latents.abs() / total[:, None]
    return out, magnitude


def launch(stated, case, operands, device):
    """One launch over every head block and token, as the package states it;
    a window-only layer passes placeholder compressed views, which its
    program never reads, and a table width of zero, as the operator does.
    """  # noqa: D205
    tokens = operands.query.shape[0]
    out = torch.full_like(operands.query, float("nan"), device=device)
    if operands.compressed is None:
        compressed = torch.zeros(1, dtype=torch.uint8)
        table, width = torch.zeros((1, 1), dtype=torch.int32), 0
    else:
        compressed = operands.compressed.pages
        table = operands.compressed.table
        width = table.shape[1]
    stated.kernel[(case.heads // case.block, tokens, 1)](
        operands.query.to(device),
        operands.window.pages.to(device),
        operands.window.table.to(device),
        operands.starts.to(device),
        compressed.to(device),
        table.to(device),
        operands.selection.to(device),
        operands.rows.to(device),
        operands.positions.to(device),
        operands.sink.to(device),
        out,
        tokens,
        operands.window.table.shape[1],
        width,
        operands.scale,
        **stated.constexprs,
        **stated.options,
    )
    return out.cpu()


def check(case, stated, device, interpreted):
    """Launch ``case`` and hold the output to the reference within the
    operator's bound.
    """  # noqa: D205
    operands = batch(case)
    out = launch(stated, case, operands, device).double()
    expected, magnitude = reference(case, operands)
    # The drawn latents are within the operator's declared envelope, so the
    # bound below is within its data-free one.
    assert float(magnitude.max()) <= LATENT_ENVELOPE

    allowed = bound(case, magnitude, expected, interpreted)
    error = (out - expected).abs()
    worst = int(torch.argmax(error - allowed))
    assert torch.all(error <= allowed), (
        f"{key(case)}: |out - reference| = {error.flatten()[worst]:.3e} at "
        f"element {worst} exceeds its bound {allowed.flatten()[worst]:.3e} "
        f"(reference {expected.flatten()[worst]:.3e}, magnitude "
        f"{magnitude.flatten()[worst]:.3e})"
    )
    if case.padding:
        assert torch.equal(out[-1], torch.zeros_like(out[-1])), (
            "a token that reads no entry attends its sink alone: zeros"
        )


@pytest.mark.gpu
@pytest.mark.parametrize("name", CASES)
def test_every_query_attends_its_window_and_its_selection_with_its_sink(
    name, torch_device, interpreted, stated_launch, launch_target
):
    case = CASES[name]
    stated = stated_launch("latent_attention", key(case), launch_target)
    check(case, stated, torch_device, interpreted)


@pytest.mark.gpu
@pytest.mark.parametrize("name", MODEL_CASES)
def test_the_model_shapes_attend_within_the_bound(
    name, torch_device, interpreted, stated_launch, launch_target
):
    case = MODEL_CASES[name]
    stated = stated_launch("latent_attention", key(case), launch_target)
    check(case, stated, torch_device, interpreted)
