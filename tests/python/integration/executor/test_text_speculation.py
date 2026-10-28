"""Speculative decoding through the text executor's token batches.

The tests play the worker's half of a text computation that speculates,
over an eight-layer Qwen3.5 fixture on one GPU in FP32: two full-attention
layers, six gated delta layers whose recurrent state each request keeps in
its slot, and the multi-token prediction layer, the model's drafter. The
pool keeps the draft cache in its draft group, whose page lays the drafter
layer's token blocks over both columns of one unit. Each test owns the
block tables of its requests: units for every K/V page, none for the state
group's pages (no checkpoints), and units for the draft pages of a request
that keeps a draft cache, the padding sentinel for one that lacks it.

- A verification batch mixes rows of every draft length: all drafts
  accepted, a draft rejected, and a row of the next token alone. Each
  request commits the reference model's greedy continuation, and its next
  rows continue it: the recurrent states advanced over exactly the tokens
  each request accepted. Each committed token reports its logprobs, the
  accepted drafts' as well as the token after them, as decoding the same
  tokens one at a time reports them.
- A request whose draft cache is missing prefills and decodes the
  reference's greedy tokens. A refill then evaluates its tokens again, from
  its beginning, writing no cache of the target and keeping its slot's
  state: it proposes the drafts a request with the same tokens that wrote
  its draft cache while it prefilled proposes, and keeps decoding the
  reference's tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests.python.fixtures.checkpoints import qwen3_5_checkpoint
from uniserve.bindings.workspace import Scratch
from uniserve.executor.text import ComputationBinding, TextBinding, TextExecutor
from uniserve.executor.text.protocol import (
    CALL_FIELDS,
    CALL_WORDS,
    COMPLETION_FIELDS,
    HEADER_WORDS,
    RECORD_MAGIC,
    CallField,
    CallFlag,
    CallKind,
    ForwardField,
    HeaderField,
    Predicate,
    RecordField,
    RecordFlag,
    forward_fields,
    page_count,
    page_start,
    record_capacity,
)
from uniserve.loading import weights
from uniserve.loading.rank import Loaded
from uniserve.models import registry as models
from uniserve.prefix_cache.units import UnitPool
from uniserve.sampling.select import TOKEN_VALUE_MASK

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="the text computation runs on a GPU",
    ),
]

PAGE = 4
UNITS = 1 + 160
SLOTS = 8
MAX_CALLS = 4
MAX_TOKENS = 64
MAX_LEN = 64
DRAFTS = 3
GREEDY = {"temperature": 0.0}
# Greedy, reporting each generated token's logprob and its two most likely.
SCORED = {"temperature": 0.0, "return_logprobs": True, "n_logprobs": 2}
# The bytes a reported row takes on the wire: four, and twelve per entry
# (the scheduler's `logprob_result_bytes`), in four-byte words.
ROW_WORDS = (4 + 12 * (1 + SCORED["n_logprobs"])) // 4


@dataclass
class Progress:
    logical: int = 0
    visible: int = 0
    computed: int = 0
    rng: int = 0


@dataclass
class Outcome:
    """One call's result as its record and the output words state it."""

    token: int
    accepted: int
    progress: Progress
    proposals: list[int]
    # Per committed token: its own (token, logprob), then its top
    # candidates'.
    logprobs: list[list[tuple[int, float]]]


class Worker:
    """The worker's half of one speculating text computation."""

    def __init__(self, root) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.model = models.load_model(
            models.read_config(root),
            device=device,
            weights=weights.Config(dtype=torch.float32),
        ).model
        loaded = Loaded(
            module=self.model,
            description=None,
            root=self.model,
            attention="torch",
        )
        self.pool = UnitPool.allocate(
            self.model.cache_config,
            num_units=UNITS,
            block_size=PAGE,
            device=device,
            request_slots=SLOTS,
            draft=self.model.mtp.cache_config,
        )
        groups = self.pool.groups
        self.draft = self.pool.draft_group()
        assert self.draft is not None and groups[self.draft].span == 2
        self.state = next(
            index for index, group in enumerate(groups) if group.states
        )
        width = -(-MAX_LEN // PAGE)
        rows = SLOTS + 1
        tables = self.pool.tables
        self.unit_tables = torch.zeros(
            (len(tables), rows, width), dtype=torch.int32, device=device
        )
        self.start_pages = torch.zeros(
            (len(groups), rows), dtype=torch.int32, device=device
        )
        self.table_shapes = torch.tensor(
            [
                [
                    table.group,
                    groups[table.group].block_size,
                    1 if groups[table.group].states else -1,
                ]
                for table in tables
            ],
            dtype=torch.int32,
            device=device,
        )
        self.verified = torch.zeros(rows, dtype=torch.int32, device=device)
        self.free = list(range(1, UNITS))
        # Each slot's units of the K/V group's pages and of the draft
        # group's pages, which a slot without its draft cache lacks.
        self.units: dict[int, list[int]] = {}
        self.drafts: dict[int, list[int] | None] = {}
        self.stream = torch.cuda.Stream(device)
        self.executor = TextExecutor(loaded, self._settings())
        self.executor.manifest()
        self.executor.bind(
            TextBinding(
                slots=SLOTS,
                units=UNITS,
                unit_pool=self.pool,
                tables={
                    "unit_tables": self.unit_tables,
                    "start_pages": self.start_pages,
                    "table_shapes": self.table_shapes,
                    "verified_lengths": self.verified,
                },
                computations=(
                    ComputationBinding(
                        index=0,
                        component="text",
                        streams=(self.stream,),
                        scratches=(Scratch(),),
                        pools=None,
                        relay_tokens=torch.zeros(
                            MAX_CALLS, dtype=torch.int64, device=device
                        ),
                        relay_flags=torch.zeros(
                            (2, MAX_CALLS), dtype=torch.uint8, device=device
                        ),
                        collectives=(None,),
                    ),
                ),
            )
        )

    def _settings(self) -> dict:
        return {
            "graphs": False,
            "model_dtype": "float32",
            "components": [
                {
                    "name": "text",
                    "host": False,
                    "calls": [{"capability": "causal_lm", "path": ""}],
                    "kinds": ["prefill", "decode", "verify"],
                }
            ],
            "pages": {
                "max_batch_calls": MAX_CALLS,
                "max_batch_tokens": MAX_TOKENS,
                "max_model_len": MAX_LEN,
                "decode_graph_batch_sizes": [],
                "prefill_graph_token_sizes": [],
                "prefill_outputs": True,
                "draft_tokens": DRAFTS,
                "cache_groups": [
                    {
                        "block_size": group.block_size,
                        "units_per_page": group.units_per_page,
                        "window": 1 if group.states else None,
                    }
                    for group in self.pool.groups
                ],
            },
        }

    # Block tables.

    def install(self, slot: int, length: int, drafts: bool) -> None:
        """Hold every page of `length` tokens: K/V units, and draft units
        when the slot keeps its draft cache; the state group holds none.
        """  # noqa: D205
        groups = self.pool.groups
        units = self.units.setdefault(slot, [])
        while len(units) * PAGE < length:
            units.append(self.free.pop(0))
        draft_page = groups[self.draft].block_size
        if drafts:
            held = self.drafts.get(slot) or []
            while len(held) * draft_page < length:
                held.append(self.free.pop(0))
            self.drafts[slot] = held
        else:
            self.drafts.setdefault(slot, None)
        for index, table in enumerate(self.pool.tables):
            row = self.unit_tables[index, slot]
            row.zero_()
            if table.group == self.draft:
                held = self.drafts[slot] or [0] * -(-length // draft_page)
                row[: len(held)] = torch.tensor(held, dtype=torch.int32)
            elif not groups[table.group].states:
                row[: len(units)] = torch.tensor(units, dtype=torch.int32)

    def keep_drafts(self, slot: int, length: int) -> None:
        """Give the slot's draft pages units: its draft cache, empty."""
        self.drafts[slot] = []
        self.install(
            slot, max(length, len(self.units.get(slot, ())) * PAGE), True
        )

    def pages(self, slot: int, prefix: int, query: int) -> list[int]:
        """Each group's first selected page and page count of a row: a
        full-history group's every held page, the state group's window of
        one token before the row on.
        """  # noqa: D205
        result = []
        groups = self.pool.groups
        held = len(self.units[slot]) * PAGE
        for index, group in enumerate(groups):
            if group.states:
                first = max(prefix - 1, 0) // group.block_size
                end = -(-(prefix + query) // group.block_size)
                result += [first, end - first]
            else:
                result += [0, -(-held // group.block_size)]
        return result

    # Batches.

    def start(self, requests, sampling=GREEDY) -> None:
        refused = self.executor.start(
            [
                {
                    "request_pool_idx": slot,
                    "request_key": {
                        "engine_id": 1,
                        "request_id": slot,
                        "request_epoch": 0,
                    },
                    "generation": {
                        "sampling": sampling,
                        "finish_token_ids": [],
                    },
                    "input_images": 0,
                }
                for slot in requests
            ]
        )
        assert refused == []

    def run(
        self, kind: int, calls: list[dict], progress, scored=()
    ) -> list[Outcome]:
        """Execute and commit one batch of `calls`, one per slot.

        A call names its `slot`, its input `tokens` (a prefill's tokens, a
        verification's current token and drafts), its `proposals` and
        `lookahead`, and a refill's first position `refill`. The batch is
        lent the output words the worker leases its calls: each call's
        completion block, a word per draft it asks for and, for a call of a
        request reporting logprobs (`scored`), the words of a row per token
        it may commit.
        """
        count = len(calls)
        columns = np.zeros((CALL_FIELDS, count), dtype=np.int64)
        groups = len(self.pool.groups)
        forward = np.zeros((forward_fields(groups), count), dtype=np.int64)
        tokens: list[int] = []
        for index, call in enumerate(calls):
            slot = call["slot"]
            state = progress[slot]
            host = list(call["tokens"])
            refill = call.get("refill")
            prefix = state.visible if refill is None else refill
            query = len(host)
            if refill is None:
                # The tables hold the positions the drafter's chained steps
                # write past the call's tokens, as the scheduler grows them.
                self.install(
                    slot,
                    prefix + query + call.get("proposals", 0),
                    self.drafts.get(slot) is not None,
                )
            column = columns[:, index]
            column[CallField.SLOT] = slot
            column[CallField.KIND] = kind
            column[CallField.ACTIVE] = 1
            column[CallField.LOGICAL] = state.logical
            column[CallField.VISIBLE] = state.visible
            column[CallField.COMPUTED] = state.computed
            column[CallField.RNG] = state.rng
            column[CallField.ALLOCATED] = len(self.units[slot]) * PAGE
            column[CallField.PREDICATE] = Predicate.NONE
            column[CallField.ROW_START] = index
            column[CallField.ROW_COUNT] = 1
            column[CallField.TOKEN_START] = len(tokens)
            column[CallField.TOKEN_COUNT] = query
            column[CallField.MAX_TOKENS] = query
            column[CallField.FLAGS] = (
                CallFlag.DRAFT_REFILL
                if refill is not None
                else CallFlag.TOKEN_OUTPUT | CallFlag.COMPLETION_OUTPUT
            )
            column[CallField.CANVAS_BLOCK] = -1
            column[CallField.CANVAS_STEP] = -1
            column[CallField.REQUEST_ID] = slot
            column[CallField.PROPOSALS] = call.get("proposals", 0)
            column[CallField.LOOKAHEAD] = call.get("lookahead", -1)
            tokens.extend(host)
            forward[ForwardField.CALL_INDEX, index] = index
            forward[ForwardField.REQUEST_POOL_INDEX, index] = slot
            forward[ForwardField.SEQ_LEN, index] = prefix + query
            forward[ForwardField.QUERY_LEN, index] = query
            forward[ForwardField.WRITE_CACHE, index] = int(refill is None)
            selected = self.pages(slot, prefix, query)
            for group in range(groups):
                forward[page_start(group), index] = selected[2 * group]
                forward[page_count(group), index] = selected[2 * group + 1]
        records = torch.zeros(
            record_capacity(count, len(tokens)), dtype=torch.int64
        )
        lent = sum(
            COMPLETION_FIELDS
            + call.get("proposals", 0)
            + (
                ROW_WORDS
                * (len(call["tokens"]) if kind == CallKind.VERIFY else 1)
                if call["slot"] in scored and call.get("refill") is None
                else 0
            )
            for call in calls
        )
        words = torch.zeros(lent, dtype=torch.int64, pin_memory=True)
        used = self.executor.execute(
            0,
            json.dumps({"batch": 1, "computation": 0, "calls": {}}),
            torch.from_numpy(columns),
            torch.from_numpy(forward),
            torch.tensor(tokens, dtype=torch.int64),
            records,
            words,
        )
        torch.cuda.synchronize(self.device)
        self.executor.commit(0)
        torch.cuda.synchronize(self.device)
        outcomes = _parse(records.numpy()[:used], words.numpy(), count)
        for call, outcome in zip(calls, outcomes, strict=True):
            if kind == CallKind.VERIFY:
                # A verification reports the coordinates it started from;
                # the worker advances them over the tokens it commits, its
                # accepted drafts and the token after them.
                committed = outcome.accepted + 1
                outcome.progress.logical += committed
                outcome.progress.rng += committed
                outcome.progress.visible += committed
            progress[call["slot"]] = outcome.progress
        return outcomes

    def close(self) -> None:
        torch.cuda.synchronize(self.device)
        self.executor.close(False)


def _parse(record: np.ndarray, words: np.ndarray, count: int) -> list[Outcome]:
    assert record[HeaderField.MAGIC] == RECORD_MAGIC
    # Each logprob segment's words and row layout, field-major: the rows'
    # tokens, scores and ranks, then their top tokens, scores and ranks.
    segments = []
    cursor = HEADER_WORDS + CALL_WORDS * count
    for _ in range(int(record[HeaderField.SEGMENTS])):
        offset, size, rows = (
            int(value) for value in record[cursor : cursor + 3]
        )
        counts = record[cursor + 3 + rows : cursor + 3 + 2 * rows]
        segments.append((words[offset : offset + size], rows, counts))
        cursor += 3 + 2 * rows

    def score(word) -> float:
        return float(
            np.array([word], dtype=np.int64)
            .astype(np.uint32)
            .view(np.float32)[0]
        )

    def logprob_row(segment: int, row: int) -> list[tuple[int, float]]:
        column, rows, counts = segments[segment]
        total = int(counts.sum())
        first = int(counts[:row].sum())
        entries = [(int(column[row]), score(column[rows + row]))]
        for top in range(first, first + int(counts[row])):
            entries.append(
                (
                    int(column[3 * rows + top]),
                    score(column[3 * rows + total + top]),
                )
            )
        return entries

    outcomes = []
    for index in range(count):
        fields = record[HEADER_WORDS + CALL_WORDS * index :][:CALL_WORDS]
        flags = int(fields[RecordField.FLAGS])
        assert flags & RecordFlag.RAN and flags & RecordFlag.PROGRESS
        token = accepted = -1
        if flags & RecordFlag.SAMPLING:
            offset = int(fields[RecordField.SAMPLING_OFFSET])
            tasks = (
                int(fields[RecordField.SAMPLING_EXTENT]) // COMPLETION_FIELDS
            )
            row = int(fields[RecordField.SAMPLING_ROW])
            assert words[offset + row] and words[offset + 4 * tasks + row]
            token = int(words[offset + 2 * tasks + row]) & TOKEN_VALUE_MASK
            accepted = int(words[offset + 3 * tasks + row])
        logprobs = []
        if flags & RecordFlag.LOGPROB:
            # One row per committed token, from the call's first.
            segment = int(fields[RecordField.LOGPROB_SEGMENT])
            first = int(fields[RecordField.LOGPROB_ROW])
            logprobs = [
                logprob_row(segment, first + position)
                for position in range(accepted + 1)
            ]
        proposals = []
        if flags & RecordFlag.PROPOSALS:
            start = int(fields[RecordField.RANGE_OFFSET])
            proposed = int(fields[RecordField.RANGE_COUNT])
            proposals = [int(word) for word in words[start : start + proposed]]
        outcomes.append(
            Outcome(
                token=token,
                accepted=accepted,
                progress=Progress(
                    logical=int(fields[RecordField.LOGICAL]),
                    rng=int(fields[RecordField.RNG]),
                    visible=int(fields[RecordField.VISIBLE]),
                    computed=int(fields[RecordField.COMPUTED]),
                ),
                proposals=proposals,
                logprobs=logprobs,
            )
        )
    return outcomes


def _greedy(reference, prompt, count: int) -> list[int]:
    """The reference model's greedy continuation of `prompt`."""
    tokens = list(prompt)
    with torch.no_grad():
        for _ in range(count):
            logits = reference(input_ids=torch.tensor([tokens])).logits
            tokens.append(int(logits[0, -1].argmax()))
    return tokens[len(prompt) :]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen3_5_drafting")
    reference, _ = qwen3_5_checkpoint(root, layers=8)
    return root, reference


@pytest.fixture
def worker(checkpoint):
    worker = Worker(checkpoint[0])
    yield worker
    worker.close()


PROMPTS = {
    1: (5, 9, 12, 3, 7, 21, 40, 2, 17),
    2: (8, 2, 30, 14, 6, 6, 19, 4, 33, 1, 25),
    3: (17, 25, 3, 11, 50, 50, 8),
}


def test_a_verification_batch_mixes_rows_of_every_draft_length(
    checkpoint, worker
):
    _, reference = checkpoint
    expected = {
        slot: _greedy(reference, prompt, 8) for slot, prompt in PROMPTS.items()
    }
    worker.start(PROMPTS, SCORED)
    progress = {slot: Progress() for slot in PROMPTS}
    for slot, prompt in PROMPTS.items():
        worker.keep_drafts(slot, len(prompt))
    prefilled = worker.run(
        CallKind.PREFILL,
        [
            {"slot": slot, "tokens": prompt, "proposals": DRAFTS}
            for slot, prompt in PROMPTS.items()
        ],
        progress,
        scored=PROMPTS,
    )
    for (slot, _), outcome in zip(PROMPTS.items(), prefilled, strict=True):
        assert outcome.token == expected[slot][0]
        assert len(outcome.proposals) == DRAFTS

    # Request 1 verifies three correct drafts, the most a call carries,
    # request 2 a correct and a wrong one, request 3 none: rows of four,
    # three and one tokens. Each asks for another count of drafts after it.
    wrong = (expected[2][2] + 1) % int(reference.config.text_config.vocab_size)
    drafts = {1: expected[1][1:4], 2: [expected[2][1], wrong], 3: []}
    asked = {1: DRAFTS, 2: DRAFTS - 1, 3: 1}
    verified = worker.run(
        CallKind.VERIFY,
        [
            {
                "slot": slot,
                "tokens": [expected[slot][0], *drafts[slot]],
                "proposals": asked[slot],
            }
            for slot in PROMPTS
        ],
        progress,
        scored=PROMPTS,
    )
    assert [outcome.accepted for outcome in verified] == [3, 1, 0]
    assert [outcome.token for outcome in verified] == [
        expected[1][4],
        expected[2][2],
        expected[3][1],
    ]
    committed = {1: 5, 2: 3, 3: 2}
    for slot, outcome in zip(PROMPTS, verified, strict=True):
        assert len(outcome.proposals) == asked[slot]
        # A row per committed token, each reporting that token.
        assert [row[0][0] for row in outcome.logprobs] == list(
            expected[slot][1 : committed[slot]]
        )
        assert (
            int(worker.verified[slot])
            == len(PROMPTS[slot]) + committed[slot] - 1
        )

    # The next rows continue each request from the tokens it accepted:
    # its recurrent states hold exactly those.
    following = worker.run(
        CallKind.VERIFY,
        [
            {"slot": slot, "tokens": [expected[slot][committed[slot] - 1]]}
            for slot in PROMPTS
        ],
        progress,
        scored=PROMPTS,
    )
    assert [outcome.token for outcome in following] == [
        expected[slot][committed[slot]] for slot in PROMPTS
    ]

    # The same prompts decoded one token at a time, in slots of their own,
    # report each token's logprobs as the verification reported them.
    decoding = {slot + 4: prompt for slot, prompt in PROMPTS.items()}
    worker.start(decoding, SCORED)
    plain = {slot: Progress() for slot in decoding}
    worker.run(
        CallKind.PREFILL,
        [{"slot": slot, "tokens": prompt} for slot, prompt in decoding.items()],
        plain,
        scored=decoding,
    )
    decoded = {slot: [] for slot in PROMPTS}
    for step in range(max(committed.values()) - 1):
        steps = worker.run(
            CallKind.VERIFY,
            [
                {"slot": slot + 4, "tokens": [expected[slot][step]]}
                for slot in PROMPTS
            ],
            plain,
            scored=decoding,
        )
        for slot, outcome in zip(PROMPTS, steps, strict=True):
            decoded[slot] += outcome.logprobs
    for slot, outcome in zip(PROMPTS, verified, strict=True):
        speculated = outcome.logprobs
        reference = decoded[slot][: len(speculated)]
        assert [[token for token, _ in row] for row in speculated] == [
            [token for token, _ in row] for row in reference
        ]
        torch.testing.assert_close(
            torch.tensor([[value for _, value in row] for row in speculated]),
            torch.tensor([[value for _, value in row] for row in reference]),
            rtol=2e-5,
            atol=2e-5,
        )


def test_a_missing_draft_cache_decodes_and_a_refill_restores_it(
    checkpoint, worker
):
    _, reference = checkpoint
    prompt = PROMPTS[1]
    expected = _greedy(reference, prompt, 6)
    # Slot 1 writes its draft cache as it prefills; slot 2, of the same
    # tokens, lacks one.
    slots = (1, 2)
    worker.start(slots)
    progress = {slot: Progress() for slot in slots}
    worker.keep_drafts(1, len(prompt))
    prefilled = worker.run(
        CallKind.PREFILL,
        [
            {"slot": slot, "tokens": prompt, "proposals": DRAFTS}
            for slot in slots
        ],
        progress,
    )
    assert [outcome.token for outcome in prefilled] == [expected[0]] * 2
    decoded = worker.run(
        CallKind.VERIFY,
        [{"slot": slot, "tokens": [expected[0]]} for slot in slots],
        progress,
    )
    assert [outcome.token for outcome in decoded] == [expected[1]] * 2

    # Slot 2's draft pages take units and a refill evaluates its tokens
    # before its last again, its last reading the token it decodes next.
    visible = progress[2].visible
    worker.keep_drafts(2, visible + 1)
    tokens = [*prompt, expected[0]]
    refilled = worker.run(
        CallKind.PREFILL,
        [
            {
                "slot": 2,
                "tokens": tokens[:visible],
                "lookahead": expected[1],
                "refill": 0,
            }
        ],
        progress,
    )
    assert refilled[0].progress == progress[2]
    assert progress[2].visible == visible

    # Both decode on and propose the same drafts: slot 2's draft cache
    # holds what slot 1's prefill and decode wrote, and its slot kept the
    # state its decoding reached.
    proposed = worker.run(
        CallKind.VERIFY,
        [
            {"slot": slot, "tokens": [expected[1]], "proposals": DRAFTS}
            for slot in slots
        ],
        progress,
    )
    assert [outcome.token for outcome in proposed] == [expected[2]] * 2
    assert proposed[0].proposals == proposed[1].proposals
    assert len(proposed[0].proposals) == DRAFTS
