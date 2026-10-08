"""Canvas readouts and steps through the numerical interface's binding.

The worker reaches token canvases only through ``canvas.describe`` and the
functions ``canvas.bind`` returns over buffers it owns. These tests play the
worker's part on the CPU over the two-layer DiffusionGemma fixture: they
prefill a prompt into a K/V unit pool, allocate the bound buffers with
torch, write the canvas tokens, positions and attention metadata, call the
binding, and compare its outputs with the public ``TokenDenoiser`` and the
block-diffusion sampler driven directly over the same cache:

- a readout returns every candidate's log-probability under the full
  vocabulary at its slot, in row, slot and candidate order, for canvas rows
  of different lengths read in one call;
- a canvas step returns each row's outcome and argmax canvas and commits the
  stepped state to its request slot, leaving the padding slot zero and the
  slot of a row whose block an earlier step stopped untouched; a row of a
  stopped block reports the skipped outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from tests.python.fixtures.checkpoints import (
    VOCAB,
    diffusion_gemma_checkpoint,
)
from uniserve_models import catalog as models
from uniserve_models.bindings.context import ExecutionContext
from uniserve_models.cache.units import UnitPool
from uniserve_models.diffusion import canvas as sampler
from uniserve_models.interface import binding, canvas, graphs
from uniserve_models.interface.loading import Loaded
from uniserve_models.loading import weights
from uniserve_models.model import CanvasInput, TextInput, TextSize
from uniserve_models.nn.attention import (
    AttentionBatch,
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)

pytestmark = pytest.mark.integration

PAGE, PAGES = 4, 12
CANVAS = 16
PROMPT = 12
SAMPLING = {
    "max_steps": 4,
    "entropy_bound": 0.5,
    "t_min": 0.4,
    "t_max": 0.8,
    "confidence_threshold": 0.2,
    "stability_threshold": 1,
}
# The constants of ``SAMPLING`` with the fixture's end of sequence 1 and 6
# and padding 0.
CONSTANTS = sampler.CanvasSampling(
    steps=4,
    entropy_bound=0.5,
    t_min=0.4,
    t_max=0.8,
    confidence=0.2,
    stability=1,
    eos_ids=(1, 6),
    pad_id=0,
)
SLOTS = 3
ROWS = 4
TOKENS = 64


@dataclass(frozen=True)
class _Setup:
    """The model, cache and device one session runs."""

    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    attention: str = "torch"
    text: dict | None = None
    vision: dict | None = None
    canvas: int = CANVAS
    vocab: int = VOCAB
    page: int = PAGE
    pages: int = PAGES
    prompt: int = PROMPT


class _Session:
    """The public model, a unit pool holding one prompt, and the binding.

    Table ``t`` of the pool gives the prompt the units
    ``1 + t * pages ...``; unit zero is the padding sentinel.
    """

    def __init__(self, root, *, sampling, setup: _Setup = _Setup()) -> None:
        self.setup = setup
        diffusion_gemma_checkpoint(
            root,
            canvas_length=setup.canvas,
            text={"vocab_size": setup.vocab, **(setup.text or {})},
            vision=setup.vision,
            unit_scores=setup.text is not None,
        )
        self.model = models.load_model(
            models.read_config(root),
            device=setup.device,
            weights=weights.Config(dtype=setup.dtype),
        ).model
        tables = len(
            UnitPool.allocate(
                self.model.text.cache_config,
                num_units=1,
                block_size=setup.page,
                device="cpu",
            ).tables
        )
        self.pool = UnitPool.allocate(
            self.model.text.cache_config,
            num_units=1 + tables * setup.pages,
            block_size=setup.page,
            device=setup.device,
        )
        cuda = torch.device(setup.device).type == "cuda"
        self.stream = torch.cuda.Stream(setup.device) if cuda else None
        self.context = ExecutionContext(
            self.model, cache=self.pool, attention=setup.attention
        )
        self.context.prepare(TextSize(TOKENS, ROWS))
        generator = torch.Generator().manual_seed(3)
        self.prompt = torch.randint(
            7, 58, (setup.prompt,), generator=generator
        ).to(setup.device)
        self._prefill()

        loaded = Loaded(
            module=self.model, description=None, attention=setup.attention
        )
        self.facts = canvas.describe(
            loaded,
            {"path": "denoiser", "exchange": False, "attention_ranks": 0},
        )
        prepared, requirements = binding.prepare(
            loaded,
            {
                "component": "diffusion_gemma",
                "path": "denoiser",
                "stream": 0 if self.stream is None else self.stream.cuda_stream,
                "attention": setup.attention,
                "vsa": "auto",
                "matmul": "auto",
                "moe": "auto",
                "derive_host_lengths": True,
                "cache": self.pool,
                "size": {"tokens": TOKENS, "rows": 2 * ROWS},
            },
        )
        workspaces = {
            item["name"]: torch.zeros(
                item["shape"],
                dtype=getattr(torch, item["dtype"]),
                device=setup.device if item["device"] >= 0 else "cpu",
            )
            for item in requirements["workspaces"]
        }
        binding.bind(prepared, {"workspaces": workspaces})
        self.buffers = self._buffers(sampling)
        options = {
            "tables": [
                {
                    "page_tokens": self.pool.groups[table.group].page_tokens,
                    "windowed": self.pool.groups[table.group].window
                    is not None,
                }
                for table in self.pool.tables
            ],
            "device": torch.device(setup.device).index or 0 if cuda else -1,
            "step_rows": 2,
        }
        if sampling:
            options["sampling"] = SAMPLING
        self.functions = canvas.bind(prepared, self.buffers, options)

    def _units(self, table: int) -> list[int]:
        pages = self.setup.pages
        return list(range(1 + table * pages, 1 + (table + 1) * pages))

    def _prefill(self) -> None:
        entries = {}
        for number, table in enumerate(self.pool.tables):
            entry = PagedInput.from_blocks(
                blocks=(tuple(self._units(number)),),
                query_lengths=(self.setup.prompt,),
                prefix_lengths=(0,),
                block_size=self.pool.groups[table.group].page_tokens,
                causal=True,
                device=self.setup.device,
            )
            if entries:
                entry = PagedInput(
                    entries[0].queries,
                    entries[0].prefixes,
                    entry.block_table,
                    entry.write_indices,
                    entry.causal,
                )
            entries[number] = entry
        batch = AttentionBatch(entries, entries[0].queries)
        self.context.bind_attention(batch)
        with self.context.activate(), torch.no_grad():
            self.model.text(
                TextInput(
                    self.prompt,
                    torch.arange(self.setup.prompt, device=self.setup.device),
                    batch,
                )
            )

    def _buffers(self, sampling: bool) -> dict[str, torch.Tensor]:
        """Allocate every bound buffer, as the worker does."""
        tables, tokens, rows = len(self.pool.tables), TOKENS, 2 * ROWS
        setup = self.setup
        hidden, dtype = self.facts["hidden_size"], setup.dtype
        canvas, device = setup.canvas, setup.device
        long = {"dtype": torch.int64, "device": device}
        ints = {"dtype": torch.int32, "device": device}
        buffers = {
            "input_ids": torch.zeros(tokens, **long),
            "positions": torch.zeros(tokens, **long),
            "attention.query_lengths": torch.zeros(rows, **ints),
            "attention.query_offsets": torch.zeros(rows + 1, **ints),
            "attention.prefix_lengths": torch.zeros(rows, **ints),
            "attention.prefix_offsets": torch.zeros(rows + 1, **ints),
            "attention.block_tables": torch.zeros(
                (tables, rows, setup.pages), **ints
            ),
            "attention.start_pages": torch.zeros((tables, rows), **ints),
            "slot_tokens": torch.zeros(tokens, **long),
            "tail_rows": torch.zeros(64, **long),
            "candidate_index": torch.zeros(256, **long),
            "candidate_logprobs": torch.zeros(
                256, dtype=torch.float32, device=device
            ),
        }
        for index, width in enumerate(self.facts["attend_widths"]):
            buffers[f"attend_state.{index}"] = torch.zeros(
                (tokens, width), dtype=dtype, device=device
            )
        if sampling:
            depth = SAMPLING["stability_threshold"]
            buffers |= {
                "step.columns": torch.zeros((4, ROWS), **long),
                "step.canvas": torch.zeros((ROWS, canvas), **long),
                "step.history": torch.zeros((ROWS, depth, canvas), **long),
                "step.self_conditioning": torch.zeros(
                    (ROWS, canvas, hidden), dtype=dtype, device=device
                ),
                "step.results": torch.zeros((ROWS, 1 + canvas), **long),
                "step.continued": torch.zeros(
                    ROWS, dtype=torch.uint8, device=device
                ),
                "bank.canvas": torch.zeros((SLOTS + 1, canvas), **long),
                "bank.history": torch.zeros((SLOTS + 1, depth, canvas), **long),
                "bank.self_conditioning": torch.zeros(
                    (SLOTS + 1, canvas, hidden), dtype=dtype, device=device
                ),
                "bank.live": torch.zeros(
                    SLOTS + 1, dtype=torch.uint8, device=device
                ),
            }
            workspace = sampler.CanvasWorkspace.empty(
                2, canvas, setup.vocab, hidden, dtype=dtype, device=device
            )
            buffers |= {
                "workspace.weights": workspace.weights,
                "workspace.normalizer": workspace.normalizer,
                "workspace.product": workspace.product,
                "workspace.scratch": workspace.scratch,
            }
        return buffers

    def write_attention(
        self, queries: list[int], padding: tuple[int, ...] = ()
    ) -> tuple[list, list]:
        """Write the metadata of canvases that read the whole prompt.

        ``padding`` sequences follow them, reading no prefix and unit zero.
        Returns each table's host page width and every row's first page.
        """
        prompt, live = self.setup.prompt, len(queries)
        lengths = [*queries, *padding]
        rows = len(lengths)
        prefixes = [prompt] * live + [0] * len(padding)
        buffers = self.buffers
        buffers["attention.query_lengths"][:rows] = torch.tensor(lengths)
        buffers["attention.query_offsets"][: rows + 1] = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()]
        )
        buffers["attention.prefix_lengths"][:rows] = torch.tensor(prefixes)
        buffers["attention.prefix_offsets"][: rows + 1] = torch.tensor(
            [0, *torch.tensor(prefixes).cumsum(0).tolist()]
        )
        widths, starts = [], []
        for number, table in enumerate(self.pool.tables):
            group = self.pool.groups[table.group]
            first = 0
            if group.window is not None:
                first = max(prompt - group.window, 0) // group.page_tokens
            pages = math.ceil(prompt / group.page_tokens) - first
            units = self._units(number)[first : first + pages]
            tables = buffers["attention.block_tables"][number]
            tables[:rows].zero_()
            tables[:live, :pages] = torch.tensor(units, dtype=torch.int32)
            buffers["attention.start_pages"][number, :rows] = torch.tensor(
                [first] * live + [0] * len(padding)
            )
            widths.append(pages)
            starts.extend([first] * live + [0] * len(padding))
        return widths, starts

    def reference_hidden(self, canvases, soft=None) -> torch.Tensor:
        """The public denoiser's final hidden rows of packed canvases."""
        queries = tuple(int(row.numel()) for row in canvases)
        widths, starts = self.write_attention(list(queries))
        entries = {}
        device, prompt = self.setup.device, self.setup.prompt
        shared = SequenceLengths.from_lengths(queries, device=device)
        prefixes = SequenceLengths.from_lengths(
            (prompt,) * len(queries), device=device
        )
        for number, table in enumerate(self.pool.tables):
            group = self.pool.groups[table.group]
            windowed = group.window is not None
            rows = len(queries)
            first = starts[number * rows : (number + 1) * rows]
            entries[number] = SegmentedInput(
                shared,
                prefixes,
                BlockTable(
                    self.buffers["attention.block_tables"][
                        number, : len(queries), : widths[number]
                    ].clone(),
                    group.page_tokens,
                    torch.tensor(first, dtype=torch.int32, device=device)
                    if windowed
                    else None,
                    tuple(first) if windowed else None,
                ),
                None,
                shared.values[:, None].expand(-1, max(queries)),
                True,
            )
        batch = AttentionBatch(entries, shared)
        self.context.bind_attention(batch)
        positions = torch.cat(
            [
                torch.arange(prompt, prompt + count, device=device)
                for count in queries
            ]
        )
        with self.context.activate(), torch.no_grad():
            return self.model.denoiser(
                CanvasInput(torch.cat(canvases), positions, batch, soft)
            )


def test_a_readout_reads_every_candidate_of_rows_of_different_lengths(
    tmp_path,
):
    session = _Session(tmp_path, sampling=False)
    assert session.facts["canvas_length"] == CANVAS
    assert session.facts["vocab_size"] == VOCAB
    assert session.facts["eos_token_ids"] == [1, 6]
    assert session.facts["generates"] and session.facts["local_tail"]

    generator = torch.Generator().manual_seed(9)
    canvases = [
        torch.randint(0, VOCAB, (CANVAS,), generator=generator),
        torch.randint(0, VOCAB, (9,), generator=generator),
    ]
    # Row-relative slots and each slot's candidates, in readout order.
    slots = [(2, (5, 7, 11)), (15, (1,)), (0, (63, 0)), (8, (4, 4, 30, 2))]
    rows_of = [0, 0, 1, 1]
    expected_hidden = session.reference_hidden(canvases)
    logits = session.model.denoiser.compute_logits(
        expected_hidden, token_indices=torch.arange(expected_hidden.shape[0])
    ).gather()
    normalized = torch.log_softmax(logits.float(), dim=-1)

    tokens = torch.cat(canvases)
    count = tokens.numel()
    session.buffers["input_ids"][:count] = tokens
    session.buffers["positions"][:count] = torch.cat(
        [torch.arange(PROMPT, PROMPT + row.numel()) for row in canvases]
    )
    widths, starts = session.write_attention([row.numel() for row in canvases])
    packed, index, expected = [], [], []
    for number, ((slot, candidates), row) in enumerate(
        zip(slots, rows_of, strict=True)
    ):
        token = slot + (0 if row == 0 else canvases[0].numel())
        packed.append(token)
        for candidate in candidates:
            index.append(number * VOCAB + candidate)
            expected.append(normalized[token, candidate])
    session.buffers["slot_tokens"][: len(packed)] = torch.tensor(packed)
    session.buffers["candidate_index"][: len(index)] = torch.tensor(index)

    functions = session.functions
    functions["readout_attend"](
        2,
        count,
        CANVAS,
        [row.numel() for row in canvases],
        [PROMPT, PROMPT],
        widths,
        starts,
    )
    tail = functions["readout_finish"](len(packed))
    functions["readout_candidates"](tail, 0, len(index))

    torch.testing.assert_close(
        session.buffers["candidate_logprobs"][: len(index)],
        torch.stack(expected),
        rtol=1e-5,
        atol=1e-6,
    )


def _gather(session, slots: list[int], rows: int) -> None:
    """Gather the rows of ``slots`` into the step views; pad with zeros."""
    for name in ("canvas", "history", "self_conditioning"):
        view = session.buffers[f"step.{name}"]
        view.zero_()
        view[: len(slots)] = session.buffers[f"bank.{name}"][slots]


def _reference_step(session, state: sampler.CanvasState, first: bool):
    """One step of the public denoiser and sampler over ``state``."""
    rows = state.canvas.shape[0]
    sampler.start_canvas(state, vocab_size=VOCAB)
    hidden = session.reference_hidden(
        list(state.canvas.unbind(0)),
        None if first else state.self_conditioning,
    )
    logits = session.model.denoiser.compute_logits(
        hidden, token_indices=torch.arange(rows * CANVAS)
    ).gather()
    decision = sampler.CanvasDecision.empty(rows, CANVAS)
    embedding = session.model.denoiser.backbone
    sampler.denoise_canvas(
        logits.view(rows, CANVAS, VOCAB),
        embedding.embedding.weight,
        embedding.embedding_scale,
        state,
        CONSTANTS,
        scores=sampler.CanvasScores.empty(rows, CANVAS),
        decision=decision,
        workspace=sampler.CanvasWorkspace.empty(
            rows,
            CANVAS,
            VOCAB,
            state.self_conditioning.shape[1],
            dtype=torch.float32,
        ),
    )
    return decision


def test_canvas_steps_follow_the_sampler_and_commit_their_slots(tmp_path):
    session = _Session(tmp_path, sampling=True)
    functions, buffers = session.functions, session.buffers
    slots, seeds = [2, 3], [11, 12]

    def run(steps: list[int], live: int) -> None:
        rows = live + 1  # One padding row: slot zero starting a canvas.
        _gather(session, slots[:live], rows)
        columns = buffers["step.columns"]
        columns.zero_()
        columns[0, :live] = torch.tensor(slots[:live])
        columns[1, :live] = torch.tensor(seeds[:live])
        columns[3, :live] = torch.tensor(steps)
        buffers["positions"][: rows * CANVAS] = torch.cat(
            [torch.arange(PROMPT, PROMPT + CANVAS)] * live
            + [torch.zeros(CANVAS, dtype=torch.int64)]
        )
        widths, starts = session.write_attention([CANVAS] * rows)
        functions["step"](
            rows,
            all(step == 0 for step in steps),
            [CANVAS] * rows,
            [PROMPT] * rows,
            widths,
            starts,
        )

    for step in range(2):
        before = {
            name: buffers[f"bank.{name}"].clone()
            for name in ("canvas", "history", "self_conditioning", "live")
        }
        # The reference steps the same three rows: the padding row gathers
        # the zero state of slot zero and starts a canvas of seed zero.
        rows = [*slots, 0]
        state = sampler.CanvasState(
            seed=torch.tensor([*seeds, 0]),
            block=torch.zeros(3, dtype=torch.int64),
            step=torch.tensor([step, step, 0]),
            canvas=before["canvas"][rows].clone(),
            history=before["history"][rows].clone(),
            self_conditioning=before["self_conditioning"][rows]
            .clone()
            .view(3 * CANVAS, -1),
        )
        decision = _reference_step(session, state, step == 0)
        run([step, step], 2)

        results = buffers["step.results"][:2]
        assert torch.equal(results[:, 1:], decision.tokens[:2])
        assert torch.equal(results[:, 0], decision.finished[:2, 0].long())
        assert torch.equal(buffers["bank.canvas"][slots], state.canvas[:2])
        assert torch.equal(buffers["bank.history"][slots], state.history[:2])
        torch.testing.assert_close(
            buffers["bank.self_conditioning"][slots].view(2 * CANVAS, -1),
            state.self_conditioning[: 2 * CANVAS],
            rtol=1e-5,
            atol=1e-6,
        )
        continued = (~decision.finished[:2, 0]).to(torch.uint8)
        assert torch.equal(buffers["bank.live"][slots], continued)
        assert torch.equal(buffers["step.continued"][:2], continued)
        # The padding row started a canvas in slot zero but never commits;
        # the unused slot stays as it was.
        for name in ("canvas", "history", "self_conditioning", "live"):
            assert torch.equal(
                buffers[f"bank.{name}"][[0, 1]], before[name][[0, 1]]
            )

    # A step queued behind a stopped block is skipped and commits nothing.
    buffers["bank.live"][slots[0]] = 0
    before = {
        name: buffers[f"bank.{name}"].clone()
        for name in ("canvas", "history", "self_conditioning")
    }
    run([2], 1)
    assert buffers["step.results"][0, 0] == canvas.STEP_SKIPPED
    for name, value in before.items():
        assert torch.equal(buffers[f"bank.{name}"], value)


# Attention shapes of the released checkpoints, which the SM100 attention
# providers serve (``test_diffusion_gemma.NATIVE_TEXT``), and a vocabulary
# and canvas the SM100 sampler kernels accept. The 60-token prompt exceeds
# the 40-token history window, so sliding tables start after retired pages.
GPU = _Setup(
    device="cuda:0",
    dtype=torch.bfloat16,
    attention="auto",
    text={
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_global_key_value_heads": 2,
        "sliding_window": 41,
        "top_k_experts": 6,
    },
    vision={"hidden_size": 144, "head_dim": 72},
    canvas=32,
    vocab=4096,
    page=16,
    pages=8,
    prompt=60,
)


def _synchronized(function, *arguments):
    """Call a binding function on its stream and wait for it."""
    torch.cuda.synchronize()
    result = function(*arguments)
    torch.cuda.synchronize()
    return result


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0),
    reason="the canvas kernels and native attention require an SM100 GPU",
)
def test_captured_readouts_and_steps_replay_their_eager_results(tmp_path):
    """Graph replays of readouts and steps equal the eager functions.

    The readout graph holds two canvases of 32 and 20 tokens padded to a
    bucket of two 32-token canvases (a 12-token and an empty padding
    sequence that read no prefix); its tail holds 64 slot rows. Its
    candidates agree with the eager pass within the rounding the different
    shapes allow. A captured step replays the eager step bit for bit, the
    committed bank rows included, for a first step and the step after it.
    """
    session = _Session(tmp_path, sampling=True, setup=GPU)
    functions, buffers = session.functions, session.buffers
    setup, vocab = GPU, GPU.vocab
    pool = graphs.pool(0)[0]
    generator = torch.Generator().manual_seed(4)

    lengths = [32, 20]
    tokens = torch.randint(0, vocab, (sum(lengths),), generator=generator)
    buffers["input_ids"][: tokens.numel()] = tokens.cuda()
    buffers["positions"][: tokens.numel()] = torch.cat(
        [torch.arange(setup.prompt, setup.prompt + count) for count in lengths]
    ).cuda()
    slots = [3, 31, 32, 50]
    candidates = [(5, 7), (4095,), (0, 1, 2), (9,)]
    buffers["slot_tokens"][: len(slots)] = torch.tensor(slots).cuda()
    flat = [
        number * vocab + token
        for number, options in enumerate(candidates)
        for token in options
    ]
    count = len(flat)
    buffers["candidate_index"][:count] = torch.tensor(flat).cuda()
    queries, prefixes = lengths, [setup.prompt] * 2

    widths, starts = session.write_attention(lengths)
    _synchronized(
        functions["readout_attend"],
        2,
        tokens.numel(),
        32,
        queries,
        prefixes,
        widths,
        starts,
    )
    tail = _synchronized(functions["readout_finish"], len(slots))
    _synchronized(functions["readout_candidates"], tail, 0, count)
    eager = buffers["candidate_logprobs"][:count].clone()

    # The bucket of two 32-token canvases: four sequences, 64 tokens.
    padding = (12, 0)
    buffers["input_ids"][tokens.numel() : 64] = 0
    buffers["positions"][tokens.numel() : 64] = 0
    widths, starts = session.write_attention(lengths, padding)
    padded = [*queries, *padding], [*prefixes, 0, 0]
    attend = _synchronized(
        functions["capture"],
        "readout_attend",
        [4, 64, 32, *padded, widths, starts],
        pool,
    )
    buffers["candidate_logprobs"].zero_()
    buffers["tail_rows"].zero_()
    buffers["tail_rows"][: len(slots)] = torch.tensor(slots).cuda()
    tail = _synchronized(functions["capture"], "readout_tail", [64], pool)
    buffers["attend_state.0"].zero_()
    buffers["attend_state.1"].zero_()
    _synchronized(functions["replay"], attend, *padded, starts)
    _synchronized(functions["replay"], tail, None, None, None)
    _synchronized(functions["readout_candidates"], tail, 0, count)
    torch.testing.assert_close(
        buffers["candidate_logprobs"][:count], eager, rtol=2e-2, atol=2e-2
    )

    rows, seeds = [1, 2], [7, 8]
    canvas = setup.canvas
    buffers["positions"][: 2 * canvas] = torch.cat(
        [torch.arange(setup.prompt, setup.prompt + canvas)] * 2
    ).cuda()
    widths, starts = session.write_attention([canvas] * 2)
    metadata = ([canvas] * 2, [setup.prompt] * 2, widths, starts)
    names = ("canvas", "history", "self_conditioning", "live")
    for step in (0, 1):
        columns = buffers["step.columns"]
        columns.zero_()
        columns[0, :2] = torch.tensor(rows)
        columns[1, :2] = torch.tensor(seeds)
        columns[3, :2] = step
        before = {name: buffers[f"bank.{name}"].clone() for name in names}

        def restore() -> None:
            for name in names:
                buffers[f"bank.{name}"].copy_(before[name])
            _gather(session, rows, 2)

        restore()
        _synchronized(functions["step"], 2, step == 0, *metadata)
        expected = {name: buffers[f"bank.{name}"].clone() for name in names}
        results = buffers["step.results"][:2].clone()

        restore()
        graph = _synchronized(
            functions["capture"], "step", [2, step == 0, *metadata], pool
        )
        restore()
        buffers["step.results"].zero_()
        _synchronized(
            functions["replay"], graph, metadata[0], metadata[1], starts
        )
        assert torch.equal(buffers["step.results"][:2], results)
        for name in names:
            assert torch.equal(buffers[f"bank.{name}"], expected[name]), name
        functions["release"](graph)
    functions["release"](attend)
    functions["release"](tail)
