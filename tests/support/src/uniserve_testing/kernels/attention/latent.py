"""What the sparse latent attention checks share: a case's operands and the
operator's formula over them.

The operator (``ops::attention::LatentAttention``) attends one shared latent,
both key and value, over each query's window entries and the compressed
entries its selection names, in one softmax with a per-head sink::

    entries(t) = window positions max(start, p - W + 1) ..= p, then the
                 compressed entries selection[t, :] names (>= 0)
    s          = scale * q . c
    m          = max(-1e30, max s)
    o          = sum e^(s - m) c / (sum e^(s - m) + e^(sink - m))

:func:`batch` draws one case's operands and :func:`reference` is that formula
in FP64 over the latents the records decode to (:mod:`uniserve_testing.
kernels.records`), read from each request's logical entries rather than
through the pages: a program that addressed a page, a slot or a ring column
wrongly reads another entry, and every record no entry names holds
:data:`TRAP`, far from any latent. :func:`bound` is the operator's tolerance
with each element's own magnitude in place of the declared latent envelope.

Each check states its own cases (:class:`Case`) and launches its own
package; :class:`LatentAttentionCall` and :func:`workspace_bytes` are what a
check of a native module hands its entry.
"""  # noqa: D205

import ctypes
import dataclasses
import math

import torch

from uniserve_testing.kernels import records

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
    the tokens one compressed entry stands for; and the seed.
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
    #: Tokens one compressed entry stands for: the compressor's ratio, which
    #: decides how many entries a context holds and how many a query sees.
    ratio: int = 2
    seed: int = 0
    #: Each request's first held window position, as after a replay from a
    #: prefix hit; empty for requests that computed every token.
    starts: tuple[int, ...] = ()
    #: Whether every selection also names its token's oldest and newest
    #: visible entries, the two ends of the request's table.
    extremes: bool = False


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
    random subset of the compressed entries complete before it (one per
    ``case.ratio`` tokens) in random slots, the others -1; with
    ``case.extremes`` the subset holds the oldest and the newest of them.
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
        counts = [max(context // case.ratio, 1) for context in contexts]
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
            visible = (position + 1) // case.ratio
            if compressed is not None and visible > 0:
                named = int(
                    torch.randint(
                        0,
                        min(visible, case.columns) + 1,
                        (),
                        generator=generator,
                    )
                )
                if case.extremes and visible > 1:
                    # Entry 0 and the newest entry, then distinct others.
                    named = max(named, 2)
                    others = torch.randperm(visible - 2, generator=generator)
                    ends = torch.tensor([0, visible - 1])
                    chosen = torch.cat((ends, others[: named - 2] + 1))
                else:
                    chosen = torch.randperm(visible, generator=generator)[
                        :named
                    ]
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


class LatentAttentionCall(ctypes.Structure):
    """``uniserve_latent_attention_call`` (``ops::attention::latent``'s
    ``LatentAttention`` call to a native package), field for field.
    """  # noqa: D205

    _fields_ = [
        ("query", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("window", ctypes.c_void_p),
        ("window_table", ctypes.c_void_p),
        ("window_starts", ctypes.c_void_p),
        ("compressed", ctypes.c_void_p),
        ("compressed_table", ctypes.c_void_p),
        ("selection", ctypes.c_void_p),
        ("rows", ctypes.c_void_p),
        ("positions", ctypes.c_void_p),
        ("sink", ctypes.c_void_p),
        ("workspace", ctypes.c_void_p),
        ("workspace_bytes", ctypes.c_int64),
        ("tokens", ctypes.c_int64),
        ("heads", ctypes.c_int64),
        ("head_dim", ctypes.c_int64),
        ("window_size", ctypes.c_int64),
        ("page", ctypes.c_int64),
        ("window_pages", ctypes.c_int64),
        ("window_record", ctypes.c_int64),
        ("window_plane_pages", ctypes.c_int64),
        ("entries", ctypes.c_int64),
        ("columns", ctypes.c_int64),
        ("compressed_pages", ctypes.c_int64),
        ("compressed_record", ctypes.c_int64),
        ("compressed_plane_pages", ctypes.c_int64),
        ("multiprocessors", ctypes.c_int64),
        ("scale", ctypes.c_float),
        ("error", ctypes.c_char * 512),
    ]


#: The entries a native module reads as one block, which its workspace holds
#: whole rows of (``attention::latent::SLOT_BLOCK``), and the alignment of
#: every region of the workspace (``WORKSPACE_ALIGNMENT``).
SLOT_BLOCK = 64
WORKSPACE_ALIGNMENT = 256


def workspace_bytes(case, tokens):
    """``attention::latent::workspace_bytes``: the bytes a call of ``case``
    over ``tokens`` query tokens lends a native module.

    One row of ``SLOT_BLOCK``-aligned slots and one length per token for the
    window, the same for the selection of a case with a compressed side, and
    one FP32 normalizer per query head of a token; each region aligned.
    """  # noqa: D205

    def region(bytes_):
        granule = WORKSPACE_ALIGNMENT
        return (bytes_ + granule - 1) // granule * granule

    def slots(columns):
        padded = (columns + SLOT_BLOCK - 1) // SLOT_BLOCK * SLOT_BLOCK
        return region(tokens * padded * 4) + region(tokens * 4)

    compressed = slots(case.columns) if case.compressed is not None else 0
    return slots(case.window) + compressed + region(tokens * case.heads * 4)
