"""Public Qwen3.5 loading agrees with the pinned Transformers model.

A tiny checkpoint of three gated delta layers and one full-attention layer
runs on the CPU through the torch attention backend in FP32, over a unit
pool whose state group keeps the gated delta layers' checkpoints. Every
token's logits equal the reference's whether the prompt is computed whole,
in chunks of any length, decoded token by token, packed beside another
request, or resumed in a fresh request slot from a checkpoint a page
boundary holds; a page the tables leave without units receives no
checkpoint.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.checkpoints import qwen3_5_checkpoint
from uniserve.bindings.context import ExecutionContext
from uniserve.loading import weights
from uniserve.models import TextInput, TextSize
from uniserve.models import registry as models
from uniserve.nn.attention import AttentionBatch, PagedInput
from uniserve.prefix_cache import slots
from uniserve.prefix_cache.units import UnitPool

pytestmark = pytest.mark.integration

# Tokens per K/V page, and so between two checkpoints of the state group.
BLOCK_SIZE = 4
PAGES = 32
REQUEST_SLOTS = 3
PROMPT = 45


def _load(root):
    return models.load_model(
        models.read_config(root),
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    ).model


class _Request:
    """One request's pages: K/V units for every page it reached, and the
    state group's units for the pages its tables give a checkpoint.
    """  # noqa: D205

    def __init__(self, pool: UnitPool, slot: int, units: iter):
        self.pool = pool
        self.slot = slot
        self._units = units
        self.tables: dict[int, list[int]] = {
            table: [] for table in range(len(pool.tables))
        }
        self.start = 0

    def grow(self, tokens: int, checkpoints: set[int]) -> None:
        """Hold every page of ``tokens`` tokens; a state page gets units
        when it ends at one of ``checkpoints``.
        """  # noqa: D205
        for table, layout in enumerate(self.pool.tables):
            group = self.pool.groups[layout.group]
            rows = self.tables[table]
            pages = -(-tokens // group.block_size)
            while len(rows) < pages:
                end = (len(rows) + 1) * group.block_size
                held = not group.states or end in checkpoints
                rows.append(next(self._units) if held else 0)

    def share(self, other: _Request, boundary: int) -> None:
        """Take ``other``'s pages before ``boundary``: its K/V pages and,
        of the state group, only the page ending at the boundary.
        """  # noqa: D205
        for table, layout in enumerate(self.pool.tables):
            group = self.pool.groups[layout.group]
            pages = boundary // group.block_size
            if group.states:
                self.tables[table] = [0] * (pages - 1) + [
                    other.tables[table][pages - 1]
                ]
            else:
                self.tables[table] = list(other.tables[table][:pages])


def _batch(pool, requests, queries, prefixes) -> AttentionBatch:
    """One forward's rows: each request's ``queries`` after ``prefixes``."""
    entries = {}
    for table, layout in enumerate(pool.tables):
        group = pool.groups[layout.group]
        width = max(len(request.tables[table]) for request in requests)
        entry = PagedInput.from_blocks(
            blocks=tuple(
                tuple(request.tables[table])
                + (0,) * (width - len(request.tables[table]))
                for request in requests
            ),
            query_lengths=tuple(queries),
            prefix_lengths=tuple(prefixes),
            block_size=group.block_size,
            causal=True,
            device="cpu",
        )
        if entries:
            first = entries[0]
            entry = replace(
                entry, queries=first.queries, prefixes=first.prefixes
            )
        entries[table] = entry
    return AttentionBatch(entries, entries[0].queries)


def _forward(model, context, rows, *, verification=None, writes=True):
    """Evaluate ``rows`` of ``(request, tokens, prefix)`` in one forward.

    Returns each row's logits, ``[tokens, vocab]``. A ``verification``
    collects the slot advances of a verification forward; rows that do not
    ``write`` read the cache their request holds and keep no state.
    """
    requests = [request for request, _, _ in rows]
    queries = [len(tokens) for _, tokens, _ in rows]
    prefixes = [prefix for _, _, prefix in rows]
    batch = _batch(context.cache, requests, queries, prefixes)
    if not writes:
        batch = AttentionBatch(
            {
                table: replace(
                    entry, slot_mapping=torch.full_like(entry.slot_mapping, -1)
                )
                for table, entry in batch.entries.items()
            },
            batch.queries,
        )
    context.bind_attention(batch)
    ids = torch.cat([tokens for _, tokens, _ in rows])
    positions = torch.cat(
        [
            torch.arange(prefix, prefix + len(tokens))
            for _, tokens, prefix in rows
        ]
    )
    slot_rows = torch.tensor([request.slot for request in requests])
    kept = torch.full((len(rows),), writes)
    with slots.selected(slot_rows, kept=kept, verification=verification):
        hidden = model(TextInput(ids, positions, batch, None))
    logits = model.compute_logits(
        hidden, token_indices=torch.arange(ids.numel())
    ).gather()
    return logits.split(queries)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen3_5")
    reference, mtp = qwen3_5_checkpoint(root)
    return root, reference, mtp


@pytest.fixture
def session(checkpoint):
    root, reference, _ = checkpoint
    model = _load(root)
    config = model.cache_config
    pool = UnitPool.allocate(
        config,
        num_units=PAGES * 8,
        block_size=BLOCK_SIZE,
        device="cpu",
        request_slots=REQUEST_SLOTS,
    )
    context = ExecutionContext(model, cache=pool, attention="torch")
    with context:
        context.prepare(TextSize(64, 3))
        yield model, context, reference, iter(range(1, PAGES * 8))


def _prompt(seed: int, length: int = PROMPT) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 61, (length,), generator=generator)


def _reference(reference, tokens):
    with torch.no_grad():
        return reference(input_ids=tokens[None]).logits[0]


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("cuts", [(), (13,), (8, 30), (1, 2, 3, 39)])
def test_chunked_prompts_and_decoding_match_transformers(session, cuts):
    """A prompt computed in chunks, then decoded token by token, yields the
    reference's logits at every position.
    """  # noqa: D205
    model, context, reference, units = session
    tokens = _prompt(len(cuts))
    expected = _reference(reference, tokens)
    request = _Request(context.cache, 1, units)
    bounds = (0, *cuts, PROMPT - 5)
    with torch.no_grad():
        for start, stop in zip(bounds, bounds[1:], strict=False):
            request.grow(stop, set(range(BLOCK_SIZE, stop + 1, BLOCK_SIZE)))
            (logits,) = _forward(
                model, context, [(request, tokens[start:stop], start)]
            )
            _close(logits, expected[start:stop])
        for position in range(PROMPT - 5, PROMPT):
            request.grow(position + 1, set())
            (logits,) = _forward(
                model,
                context,
                [(request, tokens[position : position + 1], position)],
            )
            _close(logits, expected[position : position + 1])


def test_packed_requests_keep_their_own_states(session):
    """Two requests computed in the same forwards, prompt chunks beside
    decode steps, each yield their own reference logits.
    """  # noqa: D205
    model, context, reference, units = session
    first, second = _prompt(5), _prompt(6, 23)
    expected = (_reference(reference, first), _reference(reference, second))
    a = _Request(context.cache, 1, units)
    b = _Request(context.cache, 2, units)
    with torch.no_grad():
        a.grow(20, set())
        b.grow(23, set())
        # Two consecutive rows of the first request continue each other.
        head, tail, right = _forward(
            model,
            context,
            [(a, first[:7], 0), (a, first[7:20], 7), (b, second, 0)],
        )
        _close(head, expected[0][:7])
        _close(tail, expected[0][7:20])
        _close(right, expected[1])
        a.grow(PROMPT, set())
        (left,) = _forward(model, context, [(a, first[20:], 20)])
        _close(left, expected[0][20:])


def test_a_fresh_slot_resumes_from_the_checkpoint_at_its_reused_prefix(
    session,
):
    """A second request of the same prompt reuses the first's pages up to a
    page boundary and computes the rest in another slot: it starts from the
    checkpoint the first wrote there, not from its slot, and yields the
    reference's logits. A page without units received no checkpoint.
    """  # noqa: D205
    model, context, reference, units = session
    tokens = _prompt(9)
    expected = _reference(reference, tokens)
    first = _Request(context.cache, 1, units)
    with torch.no_grad():
        # The first request's last boundary within each forward holds a
        # checkpoint: 12 for its first chunk, 28 for its second.
        first.grow(13, {12})
        _forward(model, context, [(first, tokens[:13], 0)])
        first.grow(30, {28})
        _forward(model, context, [(first, tokens[13:30], 13)])

        # The second request's slot holds another state: the first's.
        context.cache.state(
            "backbone.layers.0.linear_attention.mixer"
        ).copy_blocks((1,), (3,))
        for boundary in (12, 28):
            second = _Request(context.cache, 3, units)
            second.share(first, boundary)
            second.grow(PROMPT, set())
            (logits,) = _forward(
                model, context, [(second, tokens[boundary:], boundary)]
            )
            _close(logits, expected[boundary:])

        # Boundary 24 lies within the first request's second forward, which
        # wrote no checkpoint there.
        states = context.cache.state("backbone.layers.1.linear_attention.mixer")
        mixer = model.backbone.layers["1"].linear_attention.mixer
        checkpoints = context.cache.checkpoints(mixer.cache_name)
        assert checkpoints is not None
        page = [
            first.tables[table][24 // BLOCK_SIZE - 1]
            for table in checkpoints.tables
        ]
        assert page == [0] * len(page)
        held = checkpoints.read(
            torch.tensor(
                [
                    [
                        first.tables[t][28 // BLOCK_SIZE - 1]
                        for t in checkpoints.tables
                    ]
                ]
            )
        )
        assert bool(held["recurrent.initialized"][0])
        assert held["recurrent.values"].abs().sum() > 0
        assert states.tensors["recurrent"].shape[0] == REQUEST_SLOTS + 1


def test_multi_token_prediction_layers_load_from_the_checkpoint(checkpoint):
    root, _, mtp = checkpoint
    model = _load(root)
    assert model.mtp is not None
    torch.testing.assert_close(
        model.mtp.projection.weight, mtp["mtp.fc.weight"]
    )
    torch.testing.assert_close(
        model.mtp.layers["0"].attention.qkv.projections.q.weight,
        mtp["mtp.layers.0.self_attn.q_proj.weight"],
    )
    torch.testing.assert_close(
        model.mtp.embedding_norm.weight,
        mtp["mtp.pre_fc_norm_embedding.weight"],
    )


def test_a_verification_keeps_the_state_of_the_tokens_it_accepts(session):
    """A verification row of the next token and three drafts yields the
    reference's logits at each of its positions and leaves the request's
    slot as it was; accepting three of its tokens then advances the slot
    as decoding them would, writing the checkpoint at the boundary they
    reach: the next position decodes to the reference's logits, and a
    fresh request resumes from that checkpoint.
    """  # noqa: D205
    model, context, reference, units = session
    tokens = _prompt(11, PROMPT + 4)
    expected = _reference(reference, tokens)
    request = _Request(context.cache, 1, units)
    # The prompt ends three tokens before a page boundary, which the three
    # accepted tokens reach.
    boundary = PROMPT + 3
    assert boundary % BLOCK_SIZE == 0
    with torch.no_grad():
        # The prompt's tables already hold the page ending at the boundary.
        request.grow(PROMPT, {boundary})
        _forward(model, context, [(request, tokens[:PROMPT], 0)])
        request.grow(PROMPT + 4, set())
        verification = slots.Verification()
        (logits,) = _forward(
            model,
            context,
            [(request, tokens[PROMPT : PROMPT + 4], PROMPT)],
            verification=verification,
        )
        _close(logits, expected[PROMPT : PROMPT + 4])
        verification.accept(torch.tensor([3]))
        (logits,) = _forward(
            model,
            context,
            [(request, tokens[boundary : boundary + 1], boundary)],
        )
        _close(logits, expected[boundary : boundary + 1])

        fresh = _Request(context.cache, 2, units)
        fresh.share(request, boundary)
        fresh.grow(PROMPT + 4, set())
        (logits,) = _forward(
            model,
            context,
            [(fresh, tokens[boundary : PROMPT + 4], boundary)],
        )
        _close(logits, expected[boundary : PROMPT + 4])


def test_rows_writing_no_cache_keep_no_state(session):
    """A row evaluating a request's earlier tokens again from a checkpoint,
    writing no cache, yields their logits and writes the checkpoint its
    tokens reach, but leaves the request's slot at its latest token.
    """  # noqa: D205
    model, context, reference, units = session
    tokens = _prompt(12, PROMPT + 1)
    expected = _reference(reference, tokens)
    request = _Request(context.cache, 1, units)
    with torch.no_grad():
        request.grow(PROMPT, {12})
        _forward(model, context, [(request, tokens[:12], 0)])
        _forward(model, context, [(request, tokens[12:PROMPT], 12)])
        # The earlier tokens from the checkpoint at 12 on, again.
        (logits,) = _forward(
            model,
            context,
            [(request, tokens[12:24], 12)],
            writes=False,
        )
        _close(logits, expected[12:24])
        request.grow(PROMPT + 1, set())
        (logits,) = _forward(
            model, context, [(request, tokens[PROMPT:], PROMPT)]
        )
        _close(logits, expected[PROMPT:])


@pytest.fixture(scope="module")
def two_attention_layers(tmp_path_factory):
    """A checkpoint of eight layers: two full-attention layers."""
    root = tmp_path_factory.mktemp("qwen3_5_eight")
    qwen3_5_checkpoint(root, layers=8)
    return root


def test_the_draft_group_keeps_every_other_group_s_pages(two_attention_layers):
    """Planned with the drafter's layer, the pool keeps its columns, planes
    and every other group as they were; the draft group's page lays the
    layer's token blocks over every column of one unit.
    """  # noqa: D205
    model = _load(two_attention_layers)
    plain = UnitPool.allocate(
        model.cache_config, num_units=8, block_size=BLOCK_SIZE, device="cpu"
    )
    drafting = UnitPool.allocate(
        model.cache_config,
        num_units=8,
        block_size=BLOCK_SIZE,
        device="cpu",
        draft=model.mtp.cache_config,
    )
    before, after = plain.planes, drafting.planes
    assert after.columns == before.columns == 2
    assert after.plane_bytes == before.plane_bytes
    assert after.groups[: len(before.groups)] == before.groups
    draft = after.groups[-1]
    assert draft.draft and drafting.draft_group() == len(before.groups)
    assert (draft.span, draft.units_per_page) == (2, 1)
    assert draft.block_size == 2 * draft.block == 2 * BLOCK_SIZE


def test_the_drafter_reads_the_draft_cache_it_wrote(two_attention_layers):
    """The drafter's states over a sequence equal its states computed a
    chunk and a token at a time, each pass reading the draft cache entries
    the passes before it wrote: block ``j`` of a page lies in column ``j``
    of the page's unit.
    """  # noqa: D205
    model = _load(two_attention_layers)
    drafter = model.mtp
    pool = UnitPool.allocate(
        model.cache_config,
        num_units=16,
        block_size=BLOCK_SIZE,
        device="cpu",
        draft=drafter.cache_config,
    )
    group = pool.groups[pool.draft_group()]
    table = pool.table(group.layers[0])
    length = 3 * group.block_size + 1
    generator = torch.Generator().manual_seed(13)
    hidden = torch.randn(length, 32, generator=generator)
    tokens = torch.randint(0, 61, (length,), generator=generator)

    def run(units, rows):
        """The drafter's states of ``(prefix, query)`` rows on one table."""
        blocks = tuple(
            unit + column * pool.num_units
            for unit in units
            for column in range(group.span)
        )
        entry = PagedInput.from_blocks(
            blocks=(blocks,) * len(rows),
            query_lengths=tuple(query for _, query in rows),
            prefix_lengths=tuple(prefix for prefix, _ in rows),
            block_size=group.block,
            causal=True,
            device="cpu",
        )
        batch = AttentionBatch({table: entry}, entry.queries)
        context.bind_attention(batch)
        index = torch.cat(
            [torch.arange(prefix, prefix + query) for prefix, query in rows]
        )
        return drafter(
            model.embed_input_ids(tokens[index]),
            hidden[index],
            index,
            batch,
        )

    context = ExecutionContext(model, cache=pool, attention="torch")
    with context, torch.no_grad():
        context.prepare(TextSize(64, 3))
        pages = -(-length // group.block_size)
        whole = run(list(range(1, 1 + pages)), [(0, length)])
        split = torch.cat(
            [
                run(list(range(5, 5 + pages)), [(0, 5)]),
                run(list(range(5, 5 + pages)), [(5, 6)]),
                *(
                    run(list(range(5, 5 + pages)), [(position, 1)])
                    for position in range(11, length)
                ),
            ]
        )
    _close(split, whole)
