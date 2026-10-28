"""Token batches of a causal language model through the text executor.

The tests play the worker's half of a text computation over the two-layer
Qwen3 fixture on one GPU: they own the prefix cache's unit pool and block
tables, allocate each slot's pages, select the attention pages of each
forward row, build a batch's call and forward columns, lend page-locked
output words and parse the completion records the executor writes; after
each batch they call the computation's commit and carry the records'
progress into the next batch. Four requests run together, prefilled in one
batch, then decoded with relayed continuations:

- a greedy request (the decode forward's own greedy selection);
- a top-k sampled request (the decode forward's own top-k sampling);
- a nucleus sampled request (the general sampler after each forward);
- a greedy request reporting generated and prompt logprobs.

Without graphs, the greedy tokens equal the Hugging Face reference's
argmax continuation and the top-k tokens equal the library's top-k sampler
applied to the reference's logits with the request's Philox draws. With
graphs, after the phased warmup and the seal, every batch replays captured
graphs and selects the same tokens as the eager computation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from tests.python.fixtures.checkpoints import qwen_checkpoint
from uniserve.bindings.workspace import Scratch
from uniserve.executor.text import ComputationBinding, TextBinding, TextExecutor
from uniserve.executor.text.context import VisionBlock
from uniserve.executor.text.philox import request_sampling_key, sampling_uniform
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
    GraphMode,
    HeaderField,
    Predicate,
    RecordField,
    RecordFlag,
    forward_fields,
    page_count,
    page_start,
    record_capacity,
)
from uniserve.executor.text.sampling import SamplingParams
from uniserve.loading import weights
from uniserve.loading.rank import Loaded
from uniserve.models import registry as models
from uniserve.prefix_cache.units import UnitPool
from uniserve.processing import PositionLayout
from uniserve.sampling.select import TOKEN_VALUE_MASK
from uniserve.sampling.top_k import sample_top_k

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="the text computation runs on a GPU",
    ),
]

PAGE = 4
UNITS = 1 + 96
SLOTS = 8
MAX_CALLS = 4
MAX_TOKENS = 64
MAX_LEN = 48
DECODE_STEPS = 6
WORDS = 4096


@dataclass(frozen=True)
class Request:
    slot: int
    request_id: int
    prompt: tuple[int, ...]
    sampling: dict


REQUESTS = (
    Request(1, 11, (5, 9, 12, 3, 7, 21), {"temperature": 0.0}),
    Request(
        2,
        12,
        (8, 2, 30, 14, 6, 6, 19, 4),
        {"temperature": 0.8, "top_k": 5, "seed": 1234},
    ),
    Request(
        3, 13, (17, 25, 3, 11), {"temperature": 0.7, "top_p": 0.9, "seed": 77}
    ),
    Request(
        4,
        14,
        (29, 1, 33, 10, 22, 18, 2),
        {
            "temperature": 0.0,
            "return_logprobs": True,
            "n_logprobs": 2,
            "return_prompt_logprobs": True,
            "n_prompt_logprobs": 1,
        },
    ),
)


def _settings(graphs: bool, media: bool) -> dict:
    return {
        "graphs": graphs,
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
            "decode_graph_batch_sizes": [1, 2, 4],
            "prefill_graph_token_sizes": [16, 32],
            "prefill_outputs": True,
            "cache_groups": [
                {"block_size": PAGE, "units_per_page": 1, "window": None}
            ],
            "image_processor": media,
            "media_axes": 1 if media else None,
        },
    }


@dataclass
class Progress:
    logical: int = 0
    visible: int = 0
    computed: int = 0
    rng: int = 0
    ready: bool = False


@dataclass
class Outcome:
    """One call's result as its record and the output words state it."""

    flags: int
    token: int
    valid: bool
    progress: Progress
    active: bool = True
    accepted: int = 0
    finite: bool = True
    logprob_tokens: list[int] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    drafts: list[int] | None = None
    terminal: int | None = None


# Marker tokens of the framed vision blocks the image facts lay out.
START, END = 35, 36


class FramedTemporal:
    """Framed blocks placing every slot at the conditioning position."""

    positions = PositionLayout.TEMPORAL

    def block(self, feature_tokens, position, close_image, grid):
        tokens = np.array([START] + [1] * feature_tokens + [END])
        return VisionBlock(
            token_ids=tokens,
            features=range(1, 1 + feature_tokens),
            positions=np.full(len(tokens), position, dtype=np.int64),
        )


class Facts:
    """The image half's answers for a model placing framed vision blocks."""

    injection = FramedTemporal()
    rope_advance = 1

    def spatial_positions(self):
        return False

    def grid(self, height, width, close_image, input_images):
        raise AssertionError("temporal blocks need no patch grid")


class Worker:
    """The worker's half of one text computation, played over one stream."""

    def __init__(self, root, graphs: bool, media: bool = False) -> None:
        device = torch.device("cuda", 0)
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
        )
        tables = len(self.pool.tables)
        width = -(-MAX_LEN // PAGE)
        rows = SLOTS + 1
        self.unit_tables = torch.zeros(
            (tables, rows, width), dtype=torch.int32, device=device
        )
        self.start_pages = torch.zeros(
            (1, rows), dtype=torch.int32, device=device
        )
        self.table_shapes = torch.tensor(
            [[0, PAGE, -1]] * tables, dtype=torch.int32, device=device
        )
        self.verified = torch.zeros(rows, dtype=torch.int32, device=device)
        self.free = list(range(1, UNITS))
        self.units: dict[int, list[int]] = {}
        self.stream = torch.cuda.Stream(device)
        self.relay_tokens = torch.zeros(
            MAX_CALLS, dtype=torch.int64, device=device
        )
        self.relay_flags = torch.zeros(
            (2, MAX_CALLS), dtype=torch.uint8, device=device
        )

        self.executor = TextExecutor(loaded, _settings(graphs, media))
        self.manifest = self.executor.manifest()
        pool = torch.cuda.MemPool() if graphs else None
        self.executor.bind(
            TextBinding(
                slots=SLOTS,
                units=UNITS,
                warmup_units=UNITS - 1,
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
                        pools=None if pool is None else (pool,),
                        relay_tokens=self.relay_tokens,
                        relay_flags=self.relay_flags,
                        collectives=(None,),
                    ),
                ),
            )
        )
        if media:
            self.executor.attach_image(Facts(), None)
        self.records: list[np.ndarray] = []

    # Block tables.

    def install(self, slot: int, length: int) -> None:
        units = self.units.setdefault(slot, [])
        while len(units) * PAGE < length:
            units.append(self.free.pop(0))
        self.unit_tables[:, slot].zero_()
        self.unit_tables[:, slot, : len(units)] = torch.tensor(
            units, dtype=torch.int32, device=self.device
        )

    def allocated(self, slot: int) -> int:
        return len(self.units.get(slot, ())) * PAGE

    def pages(self, slots) -> np.ndarray:
        # Full attention selects every installed page from page zero.
        counts = [len(self.units[int(slot)]) for slot in slots]
        return np.array([[0] * len(counts), counts], dtype=np.int64)

    def release(self, slot: int) -> None:
        units = self.units.pop(slot, [])
        self.pool.zero_units(units)
        self.free.extend(units)
        self.unit_tables[:, slot].zero_()

    # Warmup.

    def warm(self) -> list[dict]:
        executor = self.executor
        described = executor.warmup_phases()
        for index, phase in enumerate(described):
            lengths = phase["lengths"]
            slots = range(1, len(lengths or ()) + 1)
            for slot, length in zip(slots, lengths or ()):
                self.install(slot, length)
            pages = [
                torch.from_numpy(self.pages([row[0] for row in step]))
                for step in phase["steps"]
            ]
            try:
                executor.warmup_phase(index, *pages)
            finally:
                torch.cuda.synchronize(self.device)
                for slot in slots:
                    self.release(slot)
        executor.warmup_finish()
        executor.seal()
        return described

    # Batches.

    def start(self, requests, input_images: int = 0) -> None:
        refused = self.executor.start(
            [
                {
                    "request_pool_idx": request.slot,
                    "request_key": {
                        "engine_id": 1,
                        "request_id": request.request_id,
                        "request_epoch": 0,
                    },
                    "generation": {
                        "sampling": request.sampling,
                        "finish_token_ids": [],
                    },
                    "input_images": input_images,
                }
                for request in requests
            ]
        )
        assert refused == []

    def run(
        self, kind: int, requests, progress, hosts=None, predicates=None
    ) -> list[Outcome]:
        """Execute and commit one batch of one call per request.

        ``hosts`` are each call's input tokens: a prefill's prompt, a
        verification's drafts. Decodes and verifications follow their
        slot's committed token. ``predicates`` gate each call on a leased
        ``uint8`` device predicate of that value, as the worker hands it
        (``None`` for none).
        """
        count = len(requests)
        entries: dict[str, dict] = {}
        features: list[torch.Tensor] = []
        calls = np.zeros((CALL_FIELDS, count), dtype=np.int64)
        forward = np.zeros((forward_fields(1), count), dtype=np.int64)
        tokens: list[int] = []
        for index, request in enumerate(requests):
            state = progress[request.slot]
            host = list(hosts[index]) if hosts is not None else []
            query = len(host) if kind == CallKind.PREFILL else len(host) + 1
            self.install(request.slot, state.visible + query)
            column = calls[:, index]
            column[CallField.SLOT] = request.slot
            column[CallField.KIND] = kind
            column[CallField.ACTIVE] = 1
            column[CallField.LOGICAL] = state.logical
            column[CallField.VISIBLE] = state.visible
            column[CallField.COMPUTED] = state.computed
            column[CallField.RNG] = state.rng
            column[CallField.PROMPT_READY] = int(state.ready)
            column[CallField.ALLOCATED] = self.allocated(request.slot)
            column[CallField.PREDICATE] = (
                Predicate.NONE if kind == CallKind.PREFILL else Predicate.RELAY
            )
            if predicates is not None and predicates[index] is not None:
                column[CallField.PREDICATE] = Predicate.DEVICE
                entries[str(index)] = {"predicate": {"tensor": len(features)}}
                features.append(
                    torch.tensor(
                        [predicates[index]],
                        dtype=torch.uint8,
                        device=self.device,
                    )
                )
            column[CallField.ROW_START] = index
            column[CallField.ROW_COUNT] = 1
            column[CallField.TOKEN_START] = len(tokens)
            column[CallField.TOKEN_COUNT] = len(host)
            column[CallField.MAX_TOKENS] = query
            column[CallField.FLAGS] = (
                CallFlag.TOKEN_OUTPUT | CallFlag.COMPLETION_OUTPUT
            )
            column[CallField.CANVAS_BLOCK] = -1
            column[CallField.CANVAS_STEP] = -1
            column[CallField.REQUEST_ID] = request.request_id
            tokens.extend(host)
            forward[ForwardField.CALL_INDEX, index] = index
            forward[ForwardField.REQUEST_POOL_INDEX, index] = request.slot
            forward[ForwardField.SEQ_LEN, index] = state.visible + query
            forward[ForwardField.QUERY_LEN, index] = query
            forward[ForwardField.WRITE_CACHE, index] = 1
        pages = self.pages([request.slot for request in requests])
        forward[page_start(0)] = pages[0]
        forward[page_count(0)] = pages[1]

        records = torch.zeros(
            record_capacity(count, len(tokens)), dtype=torch.int64
        )
        words = torch.zeros(WORDS, dtype=torch.int64, pin_memory=True)
        used = self.executor.execute(
            0,
            json.dumps(
                {
                    "batch": 1,
                    "computation": 0,
                    "collective_seq": 0,
                    "calls": entries,
                }
            ),
            torch.from_numpy(calls),
            torch.from_numpy(forward),
            torch.tensor(tokens, dtype=torch.int64),
            records,
            words,
            *features,
        )
        torch.cuda.synchronize(self.device)
        self.executor.commit(0)
        torch.cuda.synchronize(self.device)
        record = records.numpy()[:used].copy()
        self.records.append(record)
        outcomes = self._parse(record, words.numpy(), count)
        for request, outcome in zip(requests, outcomes):
            progress[request.slot] = outcome.progress
        return outcomes

    def context(self, request, tokens, offset, features, progress) -> Outcome:
        """Execute and commit a context prefill placing one vision block.

        The call's rows are its prompt run before the block, the block
        (the facts' framed layout) and the prompt run after it.
        """
        block = len(Facts.injection.block(len(features), 0, False, None))
        queries = [offset, block, len(tokens) - offset]
        total = sum(queries)
        self.install(request.slot, total)
        calls = np.zeros((CALL_FIELDS, 1), dtype=np.int64)
        column = calls[:, 0]
        column[CallField.SLOT] = request.slot
        column[CallField.KIND] = CallKind.PREFILL
        column[CallField.ACTIVE] = 1
        column[CallField.ALLOCATED] = self.allocated(request.slot)
        column[CallField.ROW_COUNT] = 3
        column[CallField.TOKEN_COUNT] = len(tokens)
        column[CallField.MAX_TOKENS] = total
        column[CallField.FLAGS] = (
            CallFlag.TOKEN_OUTPUT
            | CallFlag.COMPLETION_OUTPUT
            | CallFlag.WRITES_CONTEXT
            | CallFlag.FEATURES
        )
        column[CallField.CANVAS_BLOCK] = -1
        column[CallField.CANVAS_STEP] = -1
        column[CallField.REQUEST_ID] = request.request_id
        forward = np.zeros((forward_fields(1), 3), dtype=np.int64)
        forward[ForwardField.REQUEST_POOL_INDEX] = request.slot
        forward[ForwardField.QUERY_LEN] = queries
        forward[ForwardField.SEQ_LEN] = np.cumsum(queries)
        forward[ForwardField.WRITE_CACHE] = 1
        pages = self.pages([request.slot] * 3)
        forward[page_start(0)] = pages[0]
        forward[page_count(0)] = pages[1]
        vision = {"offset": offset, "tensor": 0, "height": 2, "width": 2}
        step = {
            "batch": 1,
            "computation": 0,
            "collective_seq": 0,
            "calls": {"0": {"vision": [vision]}},
        }
        records = torch.zeros(
            record_capacity(1, len(tokens)), dtype=torch.int64
        )
        words = torch.zeros(WORDS, dtype=torch.int64, pin_memory=True)
        used = self.executor.execute(
            0,
            json.dumps(step),
            torch.from_numpy(calls),
            torch.from_numpy(forward),
            torch.tensor(tokens, dtype=torch.int64),
            records,
            words,
            features,
        )
        torch.cuda.synchronize(self.device)
        self.executor.commit(0)
        torch.cuda.synchronize(self.device)
        record = records.numpy()[:used].copy()
        self.records.append(record)
        (outcome,) = self._parse(record, words.numpy(), 1)
        progress[request.slot] = outcome.progress
        return outcome

    @staticmethod
    def _parse(
        record: np.ndarray, words: np.ndarray, count: int
    ) -> list[Outcome]:
        assert record[HeaderField.MAGIC] == RECORD_MAGIC
        assert record[HeaderField.CALLS] == count
        written = int(record[HeaderField.WORDS_WRITTEN])
        segments = []
        cursor = HEADER_WORDS + CALL_WORDS * count
        for _ in range(int(record[HeaderField.SEGMENTS])):
            offset, size, rows = (
                int(value) for value in record[cursor : cursor + 3]
            )
            assert offset + size <= written
            segments.append((offset, rows))
            cursor += 3 + 2 * rows
        outcomes = []
        for index in range(count):
            fields = record[HEADER_WORDS + CALL_WORDS * index :][:CALL_WORDS]
            flags = int(fields[RecordField.FLAGS])
            assert flags & RecordFlag.RAN and flags & RecordFlag.PROGRESS
            assert flags & RecordFlag.SAMPLING
            offset = int(fields[RecordField.SAMPLING_OFFSET])
            tasks = (
                int(fields[RecordField.SAMPLING_EXTENT]) // COMPLETION_FIELDS
            )
            row = int(fields[RecordField.SAMPLING_ROW])
            outcome = Outcome(
                flags=flags,
                token=int(words[offset + 2 * tasks + row]) & TOKEN_VALUE_MASK,
                valid=bool(words[offset + row]),
                active=bool(words[offset + tasks + row]),
                accepted=int(words[offset + 3 * tasks + row]),
                finite=bool(words[offset + 4 * tasks + row]),
                progress=Progress(
                    logical=int(fields[RecordField.LOGICAL]),
                    rng=int(fields[RecordField.RNG]),
                    visible=int(fields[RecordField.VISIBLE]),
                    computed=int(fields[RecordField.COMPUTED]),
                    ready=bool(flags & RecordFlag.PROMPT_LOGITS_READY),
                ),
            )
            if flags & RecordFlag.LOGPROB:
                start, rows = segments[int(fields[RecordField.LOGPROB_SEGMENT])]
                # A packed row's selected token is its first word.
                packed = int(fields[RecordField.LOGPROB_ROW])
                outcome.logprob_tokens.append(int(words[start + packed]))
            tail = record[int(fields[RecordField.TAIL_OFFSET]) :]
            if flags & RecordFlag.PROMPT:
                for entry in range(int(tail[0])):
                    segment, packed = tail[1 + 2 * entry : 3 + 2 * entry]
                    start, rows = segments[int(segment)]
                    outcome.prompt_tokens.append(int(words[start + packed]))
            if flags & RecordFlag.SPECULATION:
                at = 1 + 2 * int(tail[0])
                drafts = int(tail[at])
                outcome.drafts = [
                    int(token) for token in tail[at + 1 :][:drafts]
                ]
                terminal = int(tail[at + 1 + drafts])
                outcome.terminal = None if terminal < 0 else terminal
            outcomes.append(outcome)
        return outcomes

    def serve(self) -> tuple[dict[int, list[int]], list[list[Outcome]]]:
        """Prefill every request in one batch, then decode them together."""
        self.start(REQUESTS)
        progress = {request.slot: Progress() for request in REQUESTS}
        generated = {request.slot: [] for request in REQUESTS}
        batches = [
            self.run(
                CallKind.PREFILL,
                REQUESTS,
                progress,
                hosts=[request.prompt for request in REQUESTS],
            )
        ]
        for _ in range(DECODE_STEPS):
            batches.append(self.run(CallKind.DECODE, REQUESTS, progress))
        for outcomes in batches:
            for request, outcome in zip(REQUESTS, outcomes):
                # A real checkpoint's logits on these prompts are finite.
                assert outcome.valid and outcome.finite
                generated[request.slot].append(outcome.token)
        return generated, batches

    def close(self) -> None:
        torch.cuda.synchronize(self.device)
        self.executor.close(False)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen")
    reference = qwen_checkpoint(root)
    return root, reference


@pytest.fixture(scope="module")
def eager(checkpoint):
    worker = Worker(checkpoint[0], graphs=False)
    generated, batches = worker.serve()
    yield worker, generated, batches
    worker.close()


def _reference_logits(reference, tokens) -> torch.Tensor:
    with torch.no_grad():
        ids = torch.tensor([tokens], dtype=torch.int64)
        return reference(input_ids=ids).logits[0, -1].float()


def test_eager_computation_selects_the_reference_continuation(
    checkpoint, eager
):
    _, reference = checkpoint
    worker, generated, batches = eager
    vocab = int(reference.config.vocab_size)
    for request in REQUESTS:
        sequence = list(request.prompt)
        params = SamplingParams.from_json(request.sampling)
        key = request_sampling_key(params.seed, 1, request.request_id, 0)
        expected = []
        for step in range(DECODE_STEPS + 1):
            logits = _reference_logits(reference, sequence)
            if request.slot == 3:
                # The nucleus sampler is compared with graphs below.
                token = generated[request.slot][step]
            elif params.temperature == 0.0:
                token = int(torch.argmax(logits))
            else:
                # The draw of the token at its semantic position.
                draw = sampling_uniform(key, len(sequence))
                tokens, valid = sample_top_k(
                    logits[None].cuda(),
                    torch.tensor([draw], dtype=torch.float32, device="cuda"),
                    torch.from_numpy(params.parameters(vocab)[None]).cuda(),
                    min(128, vocab),
                    torch.zeros((1, vocab), dtype=torch.int32, device="cuda"),
                    torch.zeros(1, dtype=torch.int64, device="cuda"),
                )
                assert bool(valid[0])
                token = int(tokens[0])
            expected.append(token)
            sequence.append(token)
        assert generated[request.slot] == expected, request

    # Each call advanced by its query, then by one token per step, with one
    # draw per sampled token; the cache holds every evaluated token.
    for request, outcome in zip(REQUESTS, batches[-1]):
        evaluated = len(request.prompt) + DECODE_STEPS
        assert outcome.progress.logical == evaluated
        assert outcome.progress.rng == DECODE_STEPS + 1
        assert outcome.progress.visible == evaluated
        assert outcome.progress.computed == evaluated


def test_logprob_words_report_the_selected_and_prompt_tokens(eager):
    _, generated, batches = eager
    reporting = REQUESTS[3]
    for step, outcomes in enumerate(batches):
        outcome = outcomes[3]
        assert outcome.logprob_tokens == [generated[reporting.slot][step]]
        if step == 0:
            # A first chunk scores its tokens after the first.
            assert outcome.flags & RecordFlag.PROMPT
            assert outcome.prompt_tokens == list(reporting.prompt[1:])
            assert outcome.progress.ready
        else:
            assert not outcome.flags & RecordFlag.PROMPT
    for outcomes in batches:
        for outcome in outcomes[:3]:
            assert not outcome.logprob_tokens


def test_captured_graphs_replay_the_eager_selections(checkpoint, eager):
    _, eager_generated, _ = eager
    worker = Worker(checkpoint[0], graphs=True)
    try:
        phases = worker.warm()
        computation = worker.executor.computations[0]
        runner = computation.runners[0]
        assert runner.graphs.sealed and len(runner.graphs) > 0
        # Prefill buckets first, largest footprint first, then decode
        # buckets largest first; every phase reads scratch tables.
        sizes = [len(phase["lengths"]) for phase in phases]
        decode = [phase for phase in phases if phase["lengths"][0] == 2]
        assert [len(phase["lengths"]) for phase in decode] == [4, 2, 1]
        assert sizes[-3:] == [4, 2, 1]

        generated, _ = worker.serve()
        assert generated == eager_generated
        modes = [int(record[HeaderField.MODE]) for record in worker.records]
        assert modes == [GraphMode.REPLAY] * len(modes)
        paddings = [
            int(record[HeaderField.PADDED_TOKENS]) for record in worker.records
        ]
        # The prefill of 25 tokens pads to the 32-token bucket; four decode
        # rows fill their bucket.
        assert paddings[0] == 32 - sum(len(r.prompt) for r in REQUESTS)
        assert paddings[1:] == [0] * DECODE_STEPS
    finally:
        worker.close()


def test_a_false_predicate_gates_a_prefill_and_the_decodes_behind_it(
    checkpoint, eager
):
    _, eager_generated, _ = eager
    worker = Worker(checkpoint[0], graphs=False)
    try:
        worker.start(REQUESTS)
        progress = {request.slot: Progress() for request in REQUESTS}
        # Each prefill waits on a device predicate, as one queued behind an
        # unresolved call does: false for the second request alone.
        gates = [1, 0, 1, 1]
        prefills = worker.run(
            CallKind.PREFILL,
            REQUESTS,
            progress,
            hosts=[request.prompt for request in REQUESTS],
            predicates=gates,
        )
        state = worker.executor.computations[0].state
        slots = [request.slot for request in REQUESTS]
        assert state.predicates[slots].tolist() == gates
        decodes = [
            worker.run(CallKind.DECODE, REQUESTS, progress)
            for _ in range(DECODE_STEPS)
        ]
        for index, request in enumerate(REQUESTS):
            outcomes = [prefills[index], *(batch[index] for batch in decodes)]
            if gates[index]:
                # A true predicate runs the prefill as an ungated one does.
                assert all(outcome.active for outcome in outcomes)
                assert [outcome.token for outcome in outcomes] == (
                    eager_generated[request.slot]
                )
            else:
                # The skipped prefill's selection does not continue its
                # slot, so no decode behind it continues either.
                assert not any(outcome.active for outcome in outcomes)
    finally:
        worker.close()


def _greedy(reference, tokens, count) -> list[int]:
    sequence = list(tokens)
    for _ in range(count):
        sequence.append(
            int(torch.argmax(_reference_logits(reference, sequence)))
        )
    return sequence[len(tokens) :]


def test_verification_accepts_drafts_up_to_the_first_mismatch(
    checkpoint, eager
):
    _, reference = checkpoint
    worker = eager[0]
    greedy = {"temperature": 0.0}
    requests = (
        Request(7, 17, (4, 8, 15, 16), greedy),
        Request(8, 18, (23, 4, 2, 31, 9), greedy),
    )
    worker.start(requests)
    progress = {request.slot: Progress() for request in requests}
    prefilled = worker.run(
        CallKind.PREFILL,
        requests,
        progress,
        hosts=[request.prompt for request in requests],
    )
    expected = [_greedy(reference, request.prompt, 4) for request in requests]
    assert [outcome.token for outcome in prefilled] == [
        tokens[0] for tokens in expected
    ]

    # The first request's drafts are the reference continuation; the
    # second's second draft is not.
    wrong = (expected[1][2] + 1) % int(reference.config.vocab_size)
    drafts = [expected[0][1:3], [expected[1][1], wrong]]
    verified = worker.run(CallKind.VERIFY, requests, progress, hosts=drafts)
    assert [outcome.accepted for outcome in verified] == [2, 1]
    assert [outcome.token for outcome in verified] == [
        expected[0][3],
        expected[1][2],
    ]
    for request, outcome, chain in zip(requests, verified, drafts):
        assert outcome.drafts == chain and outcome.terminal is None
        # The progress is the speculation base the worker resolves: the
        # prompt visible, the relayed token and drafts computed.
        prompt = len(request.prompt)
        assert outcome.progress.logical == prompt
        assert outcome.progress.visible == prompt
        assert outcome.progress.computed == prompt + 3
        assert outcome.progress.rng == 1
    # The commit verified each slot's cache through its emitted tokens.
    assert int(worker.verified[7]) == len(requests[0].prompt) + 3
    assert int(worker.verified[8]) == len(requests[1].prompt) + 2


def test_guidance_prefixes_write_the_cache_a_prefill_continues(
    checkpoint, eager
):
    _, reference = checkpoint
    worker = eager[0]
    prefix = (3, 14, 15, 9, 2, 6)
    request = Request(6, 16, (26,), {"temperature": 0.0})
    worker.start([request])
    worker.install(request.slot, len(prefix))
    computation = worker.executor.computations[0]
    executed = computation.prefill_cache(
        [(request.slot, prefix)], worker.pages([request.slot])
    )
    assert executed.mode == GraphMode.EAGER
    assert executed.live_tokens == len(prefix)

    progress = {request.slot: Progress(len(prefix), len(prefix), len(prefix))}
    (outcome,) = worker.run(
        CallKind.PREFILL, [request], progress, hosts=[request.prompt]
    )
    assert outcome.token == _greedy(reference, prefix + request.prompt, 1)[0]


def _media_reference(reference, tokens, offset, features, steps) -> list[int]:
    """The greedy continuation of a prompt holding one framed block.

    The block's slots attend to all of the block and what precedes it, sit
    at one rotary position, and take the features as their embeddings;
    every other token attends causally.
    """
    count = len(features)
    ids = list(tokens[:offset]) + [START] + [1] * count + [END]
    ids += list(tokens[offset:])
    first, last = offset, offset + count + 1
    positions = list(range(offset)) + [offset] * (count + 2)
    positions += list(range(offset + 1, offset + 1 + len(tokens) - offset))
    generated: list[int] = []
    for _ in range(steps):
        length = len(ids)
        with torch.no_grad():
            embeds = reference.model.embed_tokens(torch.tensor([ids]))
            embeds[0, first + 1 : first + 1 + count] = features.cpu()
            allowed = torch.ones(length, length).tril().bool()
            allowed[first : last + 1, first : last + 1] = True
            mask = torch.zeros(1, 1, length, length)
            mask.masked_fill_(~allowed, float("-inf"))
            logits = reference(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=torch.tensor([positions]),
            ).logits[0, -1]
        token = int(torch.argmax(logits))
        generated.append(token)
        ids.append(token)
        positions.append(positions[-1] + 1)
    return generated


def _serve_context(worker) -> tuple[list[int], torch.Tensor]:
    request = Request(5, 15, (5, 9, 12), {"temperature": 0.0})
    features = torch.randn(
        (3, 32), generator=torch.Generator().manual_seed(5)
    ).cuda()
    worker.start([request], input_images=1)
    progress = {request.slot: Progress()}
    outcome = worker.context(
        request, list(request.prompt), 2, features, progress
    )
    # The prompt's two runs and the block span eight cache tokens; the
    # block takes one position.
    assert outcome.progress.visible == 8 and outcome.progress.logical == 4
    generated = [outcome.token]
    for _ in range(3):
        (outcome,) = worker.run(CallKind.DECODE, [request], progress)
        generated.append(outcome.token)
    return generated, features


def test_context_prefill_places_vision_features_in_the_sequence(checkpoint):
    _, reference = checkpoint
    eager = Worker(checkpoint[0], graphs=False, media=True)
    try:
        generated, features = _serve_context(eager)
    finally:
        eager.close()
    assert generated == _media_reference(
        reference, (5, 9, 12), 2, features, len(generated)
    )

    graphed = Worker(checkpoint[0], graphs=True, media=True)
    try:
        phases = graphed.warm()
        # The context warmup follows every bucket phase.
        assert phases[-1]["steps"] == [[[1, 0, 1], [1, 1, 6]], [[1, 7, 6]]]
        replayed, _ = _serve_context(graphed)
        modes = [int(record[HeaderField.MODE]) for record in graphed.records]
        assert modes == [GraphMode.REPLAY] * len(modes)
    finally:
        graphed.close()
    assert replayed == generated
