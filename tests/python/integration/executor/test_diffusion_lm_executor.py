"""Token-denoising batches of a diffusion language model through the
diffusion LM executor.

The tests play the worker's half of a diffusion LM lane over the two-layer
DiffusionGemma fixture on one SM100 GPU: they prefill a prompt into the
prefix cache's unit pool, install its pages as the block tables of request
slots, select each forward row's attention pages as the worker does, build
a batch's call and forward columns, lend page-locked output words and parse
the completion records the lane writes. Each batch is compared with direct
calls of the lane's numerical runner over tensors the test owns, with block
tables written on the host:

- a readout of blocks of different lengths, two in one call, returns every
  candidate's log-probability in call order, and a call without candidates
  an empty range; an inactive call reports nothing;
- denoising steps of two calls around an inactive one return each row's
  outcome and block, commit the stepped state to their slots and relay each
  row's continuation into the worker's completion-flag staging;
- with graphs, after the phased warmup over scratch tables and the seal,
  every batch replays captured graphs: the denoising steps equal the direct
  calls bit for bit, and the readout's candidates agree with them within
  the rounding its padded bucket allows.
"""  # noqa: D205

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from tests.python.fixtures.checkpoints import diffusion_gemma_checkpoint
from uniserve.bindings.context import ExecutionContext
from uniserve.bindings.workspace import Scratch
from uniserve.executor.diffusion_lm import DiffusionLMExecutor
from uniserve.executor.diffusion_lm.calls import seed_bits
from uniserve.executor.text import LaneBinding, TextBinding
from uniserve.executor.text.protocol import (
    CALL_FIELDS,
    CALL_WORDS,
    HEADER_WORDS,
    RECORD_MAGIC,
    CallField,
    CallFlag,
    CallKind,
    ForwardField,
    GraphMode,
    HeaderField,
    RecordField,
    RecordFlag,
    forward_fields,
    page_count,
    page_start,
    record_capacity,
)
from uniserve.loading import weights
from uniserve.loading.rank import Loaded
from uniserve.models import TextInput, TextSize
from uniserve.models import registry as models
from uniserve.nn.attention import AttentionBatch, PagedInput
from uniserve.prefix_cache.units import UnitPool

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 0),
        reason="the sampler kernels and native attention require an SM100 GPU",
    ),
]

# Attention shapes the SM100 attention providers serve and a vocabulary and
# block the SM100 sampler kernels accept
# (``tests/python/integration/runner/test_diffusion_lm_runner.py``). The
# 60-token prompt exceeds the 40-token history window, so sliding tables
# start after a retired page.
TEXT = {
    "vocab_size": 4096,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_global_key_value_heads": 2,
    "sliding_window": 41,
    "top_k_experts": 6,
}
VISION = {"hidden_size": 144, "head_dim": 72}
VOCAB = TEXT["vocab_size"]
CANVAS = 32
PAGE, PAGES = 16, 8
PROMPT = 60
SLOTS = 4
MAX_CALLS = 4
MAX_TOKENS = 128
WORDS = 4096
# The served sampling; the stability threshold keeps one argmax block.
SAMPLING = {
    "canvas_length": CANVAS,
    "max_steps": 4,
    "entropy_bound": 0.5,
    "t_min": 0.4,
    "t_max": 0.8,
    "confidence_threshold": 0.2,
    "stability_threshold": 1,
}
BANKS = ("canvas", "history", "self_conditioning", "live")
# The direct calls' expert capacity: the fixture has no expert-parallel
# layer.
NO_EXPERT_STEP = 0


@dataclass(frozen=True)
class Call:
    """One call of a batch, as the worker plans it.

    ``rows`` are its forward rows' query tokens. A readout carries its
    input ``tokens``, its answer slots' indices among them and each slot's
    candidates; a step its canvas step.
    """

    slot: int
    rows: tuple[int, ...]
    active: bool = True
    tokens: tuple[int, ...] = ()
    slot_tokens: tuple[int, ...] = ()
    candidates: tuple[tuple[int, ...], ...] = ()
    step: int | None = None
    prompt_ready: bool = False


@dataclass
class Batch:
    """One executed batch: its records and the words it wrote."""

    records: np.ndarray
    words: np.ndarray
    calls: list[np.ndarray] = field(default_factory=list)

    def header(self, name: HeaderField) -> int:
        return int(self.records[name])

    def record(self, call: int) -> np.ndarray:
        start = HEADER_WORDS + CALL_WORDS * call
        return self.records[start : start + CALL_WORDS]

    def range(self, call: int) -> tuple[int, int]:
        record = self.record(call)
        return (
            int(record[RecordField.RANGE_OFFSET]),
            int(record[RecordField.RANGE_COUNT]),
        )


class Model:
    """The fixture's model and a unit pool holding one prefilled prompt.

    Table ``t`` gives the prompt the units ``1 + t * PAGES ...``; the last
    units of each table's range are scratch units, and unit zero is the
    padding sentinel.
    """

    def __init__(self, root) -> None:
        diffusion_gemma_checkpoint(
            root,
            canvas_length=CANVAS,
            text=TEXT,
            vision=VISION,
            unit_scores=True,
        )
        self.device = torch.device("cuda", 0)
        self.model = models.load_model(
            models.read_config(root),
            device=self.device,
            weights=weights.Config(dtype=torch.bfloat16),
        ).model
        self.loaded = Loaded(
            module=self.model,
            description=None,
            root=self.model,
            attention="auto",
        )
        tables = len(
            UnitPool.allocate(
                self.model.text.cache_config,
                num_units=1,
                block_size=PAGE,
                device="cpu",
            ).tables
        )
        self.units = 1 + tables * PAGES
        self.pool = UnitPool.allocate(
            self.model.text.cache_config,
            num_units=self.units,
            block_size=PAGE,
            device=self.device,
        )
        self.prompt = torch.randint(
            7, 58, (PROMPT,), generator=torch.Generator().manual_seed(3)
        ).to(self.device)
        self._prefill()

    def table_units(self, table: int) -> list[int]:
        return list(range(1 + table * PAGES, 1 + (table + 1) * PAGES))

    def _prefill(self) -> None:
        entries = {}
        for number, table in enumerate(self.pool.tables):
            entry = PagedInput.from_blocks(
                blocks=(tuple(self.table_units(number)),),
                query_lengths=(PROMPT,),
                prefix_lengths=(0,),
                block_size=self.pool.groups[table.group].block_size,
                causal=True,
                device=self.device,
            )
            if entries:
                entry = PagedInput(
                    entries[0].queries,
                    entries[0].prefixes,
                    entry.block_table,
                    entry.slot_mapping,
                    entry.causal,
                )
            entries[number] = entry
        batch = AttentionBatch(entries, entries[0].queries)
        context = ExecutionContext(
            self.model, cache=self.pool, attention="auto"
        )
        context.prepare(TextSize(MAX_TOKENS, 2 * MAX_CALLS))
        context.bind_attention(batch)
        with context.activate(), torch.no_grad():
            self.model.text(
                TextInput(
                    self.prompt,
                    torch.arange(PROMPT, device=self.device),
                    batch,
                )
            )
        torch.cuda.synchronize(self.device)
        context.close()


def _settings(model: Model, graphs: bool) -> dict:
    return {
        "device": "cuda:0",
        "graphs": graphs,
        "components": [
            {
                "name": "denoiser",
                "host": False,
                "calls": [{"capability": "diffusion_lm", "path": "denoiser"}],
                "kinds": ["token_denoising"],
            }
        ],
        "pages": {
            "max_batch_calls": MAX_CALLS,
            "max_batch_tokens": MAX_TOKENS,
            "max_model_len": MAX_TOKENS,
            "graph_policy": "full" if graphs else "off",
            "cache_groups": [
                {
                    "block_size": group.block_size,
                    "units_per_page": group.units_per_page,
                    "window": group.window,
                }
                for group in model.pool.groups
            ],
            "canvas_sampling": SAMPLING,
            "expert": {"exchanges_tokens": False, "attention_ranks": 0},
        },
    }


class Worker:
    """The worker's half of one diffusion LM lane, played over one stream.

    Slots ``1..3`` hold requests over the prompt; their block tables hold
    the prompt's pages from page zero.
    """

    def __init__(self, model: Model, graphs: bool) -> None:
        self.model = model
        device = model.device
        pool = model.pool
        self.tables = pool.tables
        self.groups = pool.groups
        self.unit_tables = torch.zeros(
            (len(self.tables), SLOTS + 1, PAGES),
            dtype=torch.int32,
            device=device,
        )
        self.start_pages = torch.zeros(
            (len(self.groups), SLOTS + 1), dtype=torch.int32, device=device
        )
        self.table_shapes = torch.tensor(
            [
                [
                    table.group,
                    self.groups[table.group].block_size,
                    -1
                    if self.groups[table.group].window is None
                    else self.groups[table.group].window,
                ]
                for table in self.tables
            ],
            dtype=torch.int32,
            device=device,
        )
        # Pages each slot's tables hold.
        self.installed = [0] * (SLOTS + 1)
        self.stream = torch.cuda.Stream(device)
        self.relay_flags = torch.zeros(
            (2, MAX_CALLS), dtype=torch.uint8, device=device
        )
        self.executor = DiffusionLMExecutor(
            model.loaded, _settings(model, graphs)
        )
        self.manifest = self.executor.manifest()
        pool_handle = torch.cuda.MemPool() if graphs else None
        self.executor.bind(
            TextBinding(
                slots=SLOTS,
                units=model.units,
                unit_pool=pool,
                tables={
                    "unit_tables": self.unit_tables,
                    "start_pages": self.start_pages,
                    "table_shapes": self.table_shapes,
                },
                lanes=(
                    LaneBinding(
                        index=0,
                        component="denoiser",
                        streams=(self.stream,),
                        scratches=(Scratch(),),
                        pools=None if pool_handle is None else (pool_handle,),
                        relay_tokens=torch.zeros(
                            MAX_CALLS, dtype=torch.int64, device=device
                        ),
                        relay_flags=self.relay_flags,
                        collectives=(None,),
                    ),
                ),
            )
        )
        self.lane = self.executor.lanes[0]

    # Block tables.

    def install(self, slot: int, units: list[list[int]]) -> None:
        """Install each table's ``units`` as slot ``slot``'s pages."""
        self.unit_tables[:, slot].zero_()
        for table, values in enumerate(units):
            self.unit_tables[table, slot, : len(values)] = torch.tensor(
                values, dtype=torch.int32
            )
        self.start_pages[:, slot] = 0
        self.installed[slot] = len(units[0])

    def install_prompt(self, slots) -> None:
        pages = math.ceil(PROMPT / PAGE)
        for slot in slots:
            self.install(
                slot,
                [
                    self.model.table_units(table)[:pages]
                    for table in range(len(self.tables))
                ],
            )

    def select(self, group, prefix: int, query: int, installed: int):
        """A read-only row's first page and page count in ``group``, as the
        worker's attention page selection makes them.
        """  # noqa: D205
        if group.window is None:
            return 0, installed
        first = max(prefix - group.window, 0) // group.block_size
        end = min(installed, math.ceil((prefix + query) / group.block_size))
        return first, max(end, first) - first

    def pages(self, rows) -> np.ndarray:
        """``int64 [2G, rows]``: each ``[slot, prefix, query]`` row's first
        page and page count per cache group.
        """  # noqa: D205
        columns = np.zeros((2 * len(self.groups), len(rows)), dtype=np.int64)
        for index, (slot, prefix, query) in enumerate(rows):
            for number, group in enumerate(self.groups):
                columns[2 * number : 2 * number + 2, index] = self.select(
                    group, prefix, query, self.installed[slot]
                )
        return columns

    # Warmup.

    def warm(self) -> list[dict]:
        """Run every warmup phase over one-token scratch tables, then seal."""
        executor = self.executor
        described = executor.warmup_phases()
        for index, phase in enumerate(described):
            slots = range(1, len(phase["lengths"]) + 1)
            # A one-token table selects one page of every table: a scratch
            # unit past the prompt's pages.
            for slot in slots:
                self.install(
                    slot,
                    [
                        [self.model.table_units(table)[PAGES - slot]]
                        for table in range(len(self.tables))
                    ],
                )
            pages = [
                torch.from_numpy(self.pages(step)) for step in phase["steps"]
            ]
            try:
                executor.warmup_phase(index, *pages)
            finally:
                torch.cuda.synchronize(self.model.device)
                for slot in slots:
                    self.unit_tables[:, slot].zero_()
                    self.installed[slot] = 0
        executor.warmup_finish()
        executor.seal()
        return described

    # Batches.

    def start(self, requests) -> None:
        """Start ``(slot, request id, seed)`` requests."""
        refused = self.executor.start(
            [
                {
                    "request_pool_idx": slot,
                    "request_key": {
                        "engine_id": 1,
                        "request_id": request,
                        "request_epoch": 0,
                    },
                    "generation": {
                        "sampling": {"temperature": 1.0, "seed": seed},
                        "finish_token_ids": [],
                    },
                }
                for slot, request, seed in requests
            ]
        )
        assert refused == []

    def run(self, calls: list[Call]) -> Batch:
        """Execute one batch of ``calls`` over the prompt's prefix."""
        count = len(calls)
        rows = sum(len(call.rows) for call in calls)
        columns = np.zeros((CALL_FIELDS, count), dtype=np.int64)
        forward = np.zeros(
            (forward_fields(len(self.groups)), rows), dtype=np.int64
        )
        tokens: list[int] = []
        entries: dict[str, dict] = {}
        row = 0
        for index, call in enumerate(calls):
            column = columns[:, index]
            column[CallField.SLOT] = call.slot
            column[CallField.KIND] = CallKind.TOKEN_DENOISING
            column[CallField.ACTIVE] = int(call.active)
            column[CallField.LOGICAL] = PROMPT
            column[CallField.VISIBLE] = PROMPT
            column[CallField.COMPUTED] = PROMPT
            column[CallField.PROMPT_READY] = int(call.prompt_ready)
            column[CallField.ALLOCATED] = self.installed[call.slot] * PAGE
            column[CallField.ROW_START] = row
            column[CallField.ROW_COUNT] = len(call.rows)
            column[CallField.TOKEN_START] = len(tokens)
            column[CallField.MAX_TOKENS] = sum(call.rows)
            column[CallField.CANVAS_BLOCK] = -1 if call.step is None else 0
            column[CallField.CANVAS_STEP] = (
                -1 if call.step is None else call.step
            )
            column[CallField.REQUEST_ID] = 20 + call.slot
            if call.step is None:
                column[CallField.FLAGS] = CallFlag.READOUT
            else:
                column[CallField.FLAGS] = CallFlag.COMPLETION_OUTPUT
            if call.active:
                tokens.extend(call.tokens)
                column[CallField.TOKEN_COUNT] = len(call.tokens)
            for query in call.rows:
                forward[ForwardField.CALL_INDEX, row] = index
                forward[ForwardField.REQUEST_POOL_INDEX, row] = call.slot
                forward[ForwardField.SEQ_LEN, row] = PROMPT + query
                forward[ForwardField.QUERY_LEN, row] = query
                if call.active:
                    pages = self.pages([[call.slot, PROMPT, query]])[:, 0]
                    for group in range(len(self.groups)):
                        forward[page_start(group), row] = pages[2 * group]
                        forward[page_count(group), row] = pages[2 * group + 1]
                row += 1
        # Each active readout's slots, offsets and candidates follow the
        # input tokens.
        for index, call in enumerate(calls):
            if not call.active or call.step is not None:
                continue
            offsets = np.cumsum([0, *(len(c) for c in call.candidates)])
            lists = {}
            for name, values in (
                ("slot_tokens", list(call.slot_tokens)),
                ("candidate_offsets", offsets.tolist()),
                (
                    "candidate_ids",
                    [token for slot in call.candidates for token in slot],
                ),
            ):
                lists[name] = [len(tokens), len(values)]
                tokens.extend(values)
            entries[str(index)] = {"readout": lists}

        records = torch.zeros(
            record_capacity(count, len(tokens)), dtype=torch.int64
        )
        words = torch.zeros(WORDS, dtype=torch.int64, pin_memory=True)
        step = {"batch": 1, "lane": 0, "collective_seq": 0, "calls": entries}
        used = self.executor.execute(
            0,
            json.dumps(step),
            torch.from_numpy(columns),
            torch.from_numpy(forward),
            torch.tensor(tokens, dtype=torch.int64),
            records,
            words,
        )
        torch.cuda.synchronize(self.model.device)
        record = records.numpy()[:used].copy()
        assert record[HeaderField.MAGIC] == RECORD_MAGIC
        assert record[HeaderField.CALLS] == count
        return Batch(record, words.numpy().copy())

    # Direct runner calls.

    def metadata(self, slots: list[int], queries: list[int]):
        """Host-written attention metadata of rows reading the prompt.

        Returns each table's page width, every row's first page
        (``[table][row]``) and the metadata tensors in the runner's
        parameter order.
        """
        device = self.model.device
        rows = len(slots)
        prefixes = [PROMPT] * rows

        def lengths(values) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        def offsets(values) -> torch.Tensor:
            return lengths([0, *np.cumsum(values).tolist()])

        tables = torch.zeros(
            (len(self.tables), rows, PAGES), dtype=torch.int32, device=device
        )
        starts = torch.zeros(
            (len(self.tables), rows), dtype=torch.int32, device=device
        )
        widths, host = [], []
        for number, table in enumerate(self.tables):
            group = self.groups[table.group]
            width = 1
            for row, (slot, query) in enumerate(zip(slots, queries)):
                first, count = self.select(
                    group, PROMPT, query, self.installed[slot]
                )
                units = self.unit_tables[number, slot, first : first + count]
                tables[number, row, :count] = units
                starts[number, row] = first
                host.append(first)
                width = max(width, count)
            widths.append(width)
        return (
            widths,
            host,
            [
                lengths(queries),
                offsets(queries),
                lengths(prefixes),
                offsets(prefixes),
                tables,
                starts,
            ],
        )

    def direct_readout(
        self,
        blocks: list[tuple[int, list[int]]],
        slot_rows: list[int],
        index: list[int],
    ) -> torch.Tensor:
        """Candidate values of direct ``attend``, ``finish`` and
        ``readout`` calls over ``(slot, tokens)`` blocks.
        """  # noqa: D205
        runner = self.lane.runner
        device = self.model.device
        queries = [len(tokens) for _, tokens in blocks]
        total = sum(queries)
        widths, starts, metadata = self.metadata(
            [slot for slot, _ in blocks], queries
        )
        hidden = self.lane.facts.hidden_size
        dtype = self.lane.facts.compute_dtype

        def longs(values) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64, device=device)

        input_ids = longs([token for _, tokens in blocks for token in tokens])
        positions = torch.cat(
            [torch.arange(PROMPT, PROMPT + count) for count in queries]
        ).to(device)
        state = [
            torch.zeros((total, width), dtype=dtype, device=device)
            for width in self.lane.facts.attend_widths
        ]
        finished = torch.zeros((total, hidden), dtype=dtype, device=device)
        values = torch.zeros(len(index), dtype=torch.float32, device=device)
        slots = len(slot_rows)
        torch.cuda.synchronize(device)
        with torch.cuda.stream(self.stream):
            runner.attend(
                len(blocks),
                total,
                max(queries),
                widths,
                queries,
                [PROMPT] * len(blocks),
                starts,
                NO_EXPERT_STEP,
                input_ids,
                positions,
                *metadata,
                *state,
            )
            runner.finish(
                slots, NO_EXPERT_STEP, longs(slot_rows), finished, *state
            )
            runner.readout(
                slots,
                len(index),
                torch.arange(slots, device=device),
                finished,
                longs(index),
                values,
            )
        torch.cuda.synchronize(device)
        return values

    def direct_denoise(
        self, slots: list[int], seeds: list[int], step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A direct ``denoise`` call stepping the blocks of ``slots``.

        Gathers the slots' state from the lane's banks, which the call
        commits to. Returns the rows' results and continuation flags.
        """
        runner = self.lane.runner
        banks = self.lane.banks
        device = self.model.device
        count = len(slots)
        index = torch.tensor(slots, device=device)
        canvas = banks.canvas[index].clone()
        history = banks.history[index].clone()
        soft = banks.self_conditioning[index].clone()
        columns = torch.tensor(
            [
                slots,
                [seed_bits(seed) for seed in seeds],
                [0] * count,
                [step] * count,
            ],
            dtype=torch.int64,
            device=device,
        )
        positions = (
            torch.arange(PROMPT, PROMPT + CANVAS).repeat(count).to(device)
        )
        widths, starts, metadata = self.metadata(slots, [CANVAS] * count)
        results = torch.zeros(
            (count, 1 + CANVAS), dtype=torch.int64, device=device
        )
        continued = torch.zeros(count, dtype=torch.uint8, device=device)
        torch.cuda.synchronize(device)
        with torch.cuda.stream(self.stream):
            runner.denoise(
                count,
                step == 0,
                widths,
                [CANVAS] * count,
                [PROMPT] * count,
                starts,
                NO_EXPERT_STEP,
                columns,
                positions,
                *metadata,
                canvas,
                history,
                soft,
                results,
                continued,
            )
        torch.cuda.synchronize(device)
        return results, continued

    def banks(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self.lane.banks, name).clone() for name in BANKS}

    def restore(self, banks: dict[str, torch.Tensor]) -> None:
        for name, value in banks.items():
            getattr(self.lane.banks, name).copy_(value)
        torch.cuda.synchronize(self.model.device)

    def close(self) -> None:
        torch.cuda.synchronize(self.model.device)
        self.executor.close(False)


# A readout of two calls around an inactive one: two blocks of 32 and 20
# tokens with four answer slots, and one block of 16 tokens whose two slots
# read no candidate.
_TOKENS = torch.randint(
    0, VOCAB, (68,), generator=torch.Generator().manual_seed(4)
).tolist()
READOUT = [
    Call(
        1,
        (32, 20),
        tokens=tuple(_TOKENS[:52]),
        slot_tokens=(3, 31, 32, 50),
        candidates=((5, 7), (4095,), (0, 1, 2), (9,)),
        prompt_ready=True,
    ),
    Call(2, (16,), active=False),
    Call(
        3,
        (16,),
        tokens=tuple(_TOKENS[52:]),
        slot_tokens=(0, 15),
        candidates=((), ()),
    ),
]
# The candidates in readout order, each by its slot's packed row.
CANDIDATES = [
    (row, token)
    for row, slot in enumerate([*READOUT[0].candidates, *READOUT[2].candidates])
    for token in slot
]
# Seeds of the slots' requests; the third is ``seed = -1``, the largest u64.
SEEDS = {1: 7, 2: 8, 3: (1 << 64) - 1}


def _readout_reference(worker: Worker) -> torch.Tensor:
    first, third = READOUT[0], READOUT[2]
    blocks = [
        (1, list(first.tokens[:32])),
        (1, list(first.tokens[32:])),
        (3, list(third.tokens)),
    ]
    slot_rows = [*first.slot_tokens, *(52 + slot for slot in third.slot_tokens)]
    index = [row * VOCAB + token for row, token in CANDIDATES]
    return worker.direct_readout(blocks, slot_rows, index)


def _candidate_values(batch: Batch) -> torch.Tensor:
    # FP32 values travel as their 32-bit patterns in each word's low half.
    words = batch.words[: len(CANDIDATES)].astype(np.uint32)
    return torch.from_numpy(words.view(np.float32).copy())


def _check_readout_records(batch: Batch, mode: GraphMode, padding: int) -> None:
    assert batch.header(HeaderField.WORDS_WRITTEN) == len(CANDIDATES)
    assert batch.header(HeaderField.MODE) == mode
    assert batch.header(HeaderField.LIVE_TOKENS) == 68
    assert batch.header(HeaderField.PADDED_TOKENS) == padding
    assert batch.header(HeaderField.ROWS) == 3
    assert batch.header(HeaderField.EXPERT_CAPACITY) == 0
    ran = RecordFlag.RAN | RecordFlag.PROGRESS | RecordFlag.CANDIDATES
    first = batch.record(0)
    assert first[RecordField.FLAGS] == ran | RecordFlag.PROMPT_LOGITS_READY
    # A readout leaves the progress as it was, computed at the visible
    # length.
    assert first[RecordField.LOGICAL : RecordField.COMPUTED + 1].tolist() == [
        PROMPT,
        0,
        PROMPT,
        PROMPT,
    ]
    assert batch.range(0) == (0, len(CANDIDATES))
    assert not batch.record(1).any()
    assert batch.record(2)[RecordField.FLAGS] == ran
    assert batch.range(2) == (len(CANDIDATES), 0)


def _steps(step: int) -> list[Call]:
    return [
        Call(1, (CANVAS,), step=step),
        Call(2, (CANVAS,), active=False, step=step),
        Call(3, (CANVAS,), step=step),
    ]


def _check_steps(worker: Worker, step: int, mode: GraphMode) -> None:
    """Run one step of slots 1 and 3 and compare it with direct calls."""
    before = worker.banks()
    results, continued = worker.direct_denoise(
        [1, 3], [SEEDS[1], SEEDS[3]], step
    )
    expected = worker.banks()
    worker.restore(before)
    # The inactive call's staged flag stays as the worker bound it.
    worker.relay_flags.fill_(7)

    batch = worker.run(_steps(step))
    width = 1 + CANVAS
    assert batch.header(HeaderField.MODE) == mode
    assert batch.header(HeaderField.WORDS_WRITTEN) == 2 * width
    assert batch.header(HeaderField.LIVE_TOKENS) == 2 * CANVAS
    assert batch.header(HeaderField.PADDED_TOKENS) == 0
    flags = RecordFlag.RAN | RecordFlag.PROGRESS | RecordFlag.CANVAS
    assert batch.record(0)[RecordField.FLAGS] == flags
    assert batch.record(2)[RecordField.FLAGS] == flags
    assert not batch.record(1).any()
    assert batch.range(0) == (0, width)
    assert batch.range(2) == (width, width)
    words = torch.from_numpy(batch.words[: 2 * width].reshape(2, width))
    assert torch.equal(words, results.cpu())
    for name in BANKS:
        assert torch.equal(getattr(worker.lane.banks, name), expected[name]), (
            name
        )
    relayed = worker.relay_flags[1].cpu()
    assert relayed[[0, 2]].tolist() == continued.cpu().tolist()
    assert int(relayed[1]) == 7


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return Model(tmp_path_factory.mktemp("diffusion_gemma"))


@pytest.fixture(scope="module")
def eager(model):
    worker = Worker(model, graphs=False)
    worker.start([(slot, 20 + slot, seed) for slot, seed in SEEDS.items()])
    worker.install_prompt(SEEDS)
    yield worker
    worker.close()


def test_the_manifest_states_one_lane_over_the_prefix_cache(eager):
    (lane,) = eager.manifest["lanes"]
    assert lane == {
        "component": "denoiser",
        "kinds": ["token_denoising"],
        "microbatches": 1,
        "prepares": False,
        "commits": False,
        "resources": ["pages"],
        "output_words": MAX_CALLS * (1 + CANVAS),
    }
    pages = eager.manifest["pages"]
    assert pages["fixed_bytes"] > 0 and pages["slot_bytes"] > 0
    # Four blocks fit a batch, one sampler chunk steps them all.
    assert eager.executor.step_rows() == 4


def test_an_eager_readout_reads_the_runners_candidate_values(eager):
    expected = _readout_reference(eager)
    batch = eager.run(READOUT)
    _check_readout_records(batch, GraphMode.EAGER, padding=0)
    torch.testing.assert_close(
        _candidate_values(batch), expected.cpu(), rtol=0, atol=0
    )


def test_eager_steps_commit_and_relay_what_the_runner_steps(eager):
    for step in (0, 1):
        _check_steps(eager, step, GraphMode.EAGER)


def test_captured_graphs_replay_the_runners_results(model):
    worker = Worker(model, graphs=True)
    try:
        (phase,) = worker.warm()
        # One-token scratch tables for the blocks a step holds; every row
        # reads one of them over a one-token prefix.
        assert phase["lengths"] == [1] * SLOTS
        assert all(
            slot in range(1, SLOTS + 1) and prefix == 1
            for rows in phase["steps"]
            for slot, prefix, _ in rows
        )
        assert worker.lane.graphs.sealed and len(worker.lane.graphs) > 0

        worker.start([(slot, 20 + slot, seed) for slot, seed in SEEDS.items()])
        worker.install_prompt(SEEDS)
        expected = _readout_reference(worker)
        batch = worker.run(READOUT)
        # Three blocks of 68 tokens pad to the bucket of three 32-token
        # blocks.
        _check_readout_records(batch, GraphMode.REPLAY, padding=96 - 68)
        torch.testing.assert_close(
            _candidate_values(batch), expected.cpu(), rtol=2e-2, atol=2e-2
        )
        for step in (0, 1):
            _check_steps(worker, step, GraphMode.REPLAY)
    finally:
        worker.close()
