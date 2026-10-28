"""The forwards of a causal language model on one computation binding.

A :class:`TextRunner` owns one binding's columns
(:class:`~.buffers.InputBuffers`), its graph catalogs and captured graphs,
and the numerical runner (:class:`~uniserve.runner.causal_lm.CausalLMRunner`)
whose methods compute on the binding's stream. A computation has one runner,
or one per expert microbatch. Per forward the runner:

1. plans the forward of an :class:`Invocation` (:meth:`TextRunner.plan`):
   the smallest captured bucket holding its rows (or its live shape when
   graphs are off), the page widths, the padded host lengths and the host
   columns;
2. stages the host columns into the binding's columns on its stream, then
   the device copies the forward reads besides them
   (:meth:`TextRunner.stage`);
3. replays the bucket's graph, runs the method eagerly, or, before the
   computation is sealed, runs it eagerly and returns the capture of its
   graph, which the caller performs behind the forward's selection
   (:meth:`TextRunner.forward`);
4. selects the sampled tasks' tokens, reports their logprobs and scores
   prompt chunks with the captured selection graphs
   (:meth:`TextRunner.select`).

A decode forward evaluates one token per row from the slot state: its
inputs are each row's slot, forced-finish flag and finish tokens, and a
row whose token comes from the host has it written to its slot first. Its
graph gathers, runs the model and selects: greedily, or, in the bucket's
sampling graph, each row from its top-k candidates with the row's draw and
parameters. A decode bucket pads with rows of slot 0.

A prefill forward evaluates rows of host tokens (prompt chunks, vision
blocks and verification rows): its inputs are the rows' packed columns,
tokens and positions. Its graph gathers, runs the model and, unless the
forward only writes the prefix cache, keeps the hidden states and projects
each row's last position. A prefill bucket adds one sequence holding every
padding token, then empty sequences. Padding rows never write K/V.

A runner whose pool keeps the draft cache runs the drafter after the
forward and its selection (:meth:`TextRunner.draft`): rows of the positions
the forward evaluated whose next token is known write their draft cache
entries, and the rows that propose select their draft tokens. A prefill row
that refills its request's draft cache writes no K/V of the target and
keeps no per-request state.

After the worker sealed a batch, :meth:`TextRunner.commit` installs the
batch's commit rows into the slot state (the ``commit`` graph of their
bucket) and copies scored chunks' retained logits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import torch

from uniserve import cuda_graph
from uniserve.runner.causal_lm import CausalLMRunner
from uniserve.sampling.state import COMMIT_FIELDS

from .buffers import (
    DECODE_ROW_BYTES,
    SELECT_ROW_BYTES,
    InputBuffers,
    Relay,
    SlotWrite,
    Stage,
    Staging,
    TableShape,
    TextBounds,
    decode_layout,
    host_region,
    select_layout,
)
from .commit import CommitWork, commit_bucket
from .graphs import (
    CommitKey,
    DecodeKey,
    GraphKey,
    GraphTable,
    PrefillKey,
    PromptKey,
    ScoreKey,
    SelectKey,
)
from .prompt import (
    PROMPT_FIELDS,
    PromptChunks,
    prompt_columns,
    prompt_padding,
    prompt_rows,
)
from .protocol import CallKind, GraphMode
from .rows import (
    PaddedRows,
    Rows,
    RowSource,
    graph_widths,
    logit_positions,
    pack_rows,
    pad_rows,
    page_width,
    text_forward,
)
from .sampling import (
    FINISH_IDS,
    PaddedSelection,
    RowSampling,
    SamplingMetadata,
    candidates,
    pad_selection,
    padding_selection,
    task_bucket,
)

__all__ = [
    "DraftRows",
    "Executed",
    "Invocation",
    "Plan",
    "Relays",
    "TextRunner",
]


@dataclass(slots=True)
class Relays:
    """The selections an invocation relays into the relay staging.

    Each pair names a call (its index in the batch) and the task column of
    its selection; a transition relay of a call that does not sample names
    column -1 and relays no transition. ``consecutive`` states whether the
    token relays name consecutive calls and columns, so they copy as one
    slice.
    """

    token_calls: np.ndarray
    token_tasks: np.ndarray
    transition_calls: np.ndarray
    transition_tasks: np.ndarray
    consecutive: bool = field(init=False)

    def __post_init__(self) -> None:
        def runs(values: np.ndarray) -> bool:
            return int(values[0]) >= 0 and bool((np.diff(values) == 1).all())

        calls, tasks = self.token_calls, self.token_tasks
        self.consecutive = bool(len(calls)) and runs(calls) and runs(tasks)


@dataclass(slots=True)
class Invocation:
    """One forward a runner evaluates: its rows and what follows them.

    Attributes:
        kind: The forward's :class:`~.protocol.CallKind`: a prefill, a
            decode or a verification.
        rows: The forward's live rows.
        starts: ``int64 [tables, rows]`` each table's first selected page.
        counts: ``int64 [tables, rows]`` each table's selected page count.
        finish_ids: ``int64 [rows, FINISH_IDS]`` each decode row's finish
            tokens when the decode forward's own selections are final;
            ``None`` otherwise.
        row_sampling: The rows' draws and parameters when the decode
            forward samples its rows itself.
        sampling: The tasks the runner's selection selects; ``None`` when
            no row samples or the decode forward's selections are final.
        prompt: The prompt chunks the forward scores.
        embeddings: ``(token, features)`` of each vision block, its first
            feature token counted from the forward's first token.
        relays: The selections the computation relays.
        refill: ``bool [rows]`` the rows refilling their request's draft
            cache, which write no cache of the target and keep no
            per-request state; ``None`` for none.
    """

    kind: int
    rows: Rows
    starts: np.ndarray
    counts: np.ndarray
    finish_ids: np.ndarray | None = None
    row_sampling: RowSampling | None = None
    sampling: SamplingMetadata | None = None
    prompt: PromptChunks | None = None
    embeddings: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    relays: Relays | None = None
    refill: np.ndarray | None = None


@dataclass(slots=True)
class DraftRows:
    """The rows of a forward's draft pass and the drafts they propose.

    Each row evaluates consecutive positions its forward evaluated, from
    ``prefixes[r]`` on, over its slot's draft table; each position reads
    the target's hidden state there and the token after it.

    Attributes:
        slots: ``int64 [n]`` each row's request slot.
        prefixes: ``int64 [n]`` each row's first position, its draft cache
            length.
        queries: ``int64 [n]`` each row's positions.
        pages: ``int64 [tables, n]`` each table's selected page count.
        tokens: ``int64 [sum queries]`` each position's next token; the
            entries ``pairs`` names are the forward's selections.
        hidden_rows: ``int64 [sum queries]`` each position's row of the
            forward's hidden states.
        pairs: ``int64 [2, s]`` (token index, task): positions whose next
            token is the forward's selection of a task, each advanced by
            the task's accepted drafts on the device: a verification's
            selection follows the drafts it accepts.
        proposers: ``int64 [p]`` the rows that propose drafts.
        bases: ``int64 [p]`` each proposer's position, relative to its
            row's first, whose drafter state proposes its first draft
            before its accepted drafts advance it: its last for a prefill,
            its first for a verification.
        tasks: ``int64 [p]`` each proposer's task.
        proposals: Drafts each proposer proposes.
    """

    slots: np.ndarray
    prefixes: np.ndarray
    queries: np.ndarray
    pages: np.ndarray
    tokens: np.ndarray
    hidden_rows: np.ndarray
    pairs: np.ndarray
    proposers: np.ndarray
    bases: np.ndarray
    tasks: np.ndarray
    proposals: int


@dataclass(slots=True)
class Plan:
    """A planned forward: its bucket, host lengths and host columns.

    Attributes:
        kind: The forward's call kind.
        rows: Rows evaluated, padding included.
        live_tokens: The forward's own tokens.
        tokens: Tokens evaluated, padding included.
        widths: Pages each table holds per row.
        padded: The padded row lengths.
        queries, prefixes, starts, causal: Host mirrors of the padded
            lengths, first pages (``[table][row]`` flat) and causality.
        device_causality: The forward reads each row's causality from the
            device column: its graph serves rows of either causality, or
            its rows differ.
        embeddings: The forward reads the embedding columns.
        axes: Rotary position axes the rows carry.
        outputs: A prefill keeps hidden states and last-position logits.
        verification: The forward is a verification whose rows hold
            drafts: its per-request states advance once its selection
            accepts each row's tokens.
        key: The graph serving the forward; ``None`` runs it eagerly.
        row_columns: A prefill's row-gather column words.
        token_ids: ``int64 [tokens]`` a prefill's tokens.
        positions: ``int64 [axes, tokens]`` a prefill's positions.
        token_relays: ``int64 [2, n]`` a prefill's relayed tokens: token
            offset and slot.
        finish_ids: ``int64 [rows, FINISH_IDS]`` a decode's finish tokens.
        candidates: Candidates each decode row samples among, 0 greedy.
        draws, params: A sampling decode's rows' draws and parameters.
        host_tokens: ``int64 [2, n]`` slots and tokens of decode rows whose
            token comes from the host.
        host_positions: ``int64 [2, n]`` slots and logical positions of
            decode rows whose position comes from the host.
        logit_rows: Token positions projected to candidate rows.
        prompt: ``int64 [n, PROMPT_FIELDS]`` the prompt rows scored.
        prompt_top: The top bucket the prompt rows are scored at.
    """

    kind: int
    rows: int
    live_tokens: int
    tokens: int
    widths: list[int]
    padded: PaddedRows
    queries: list[int]
    prefixes: list[int]
    starts: list[int]
    causal: list[int]
    device_causality: bool
    embeddings: bool
    axes: int
    outputs: bool
    verification: bool
    key: GraphKey | None
    row_columns: np.ndarray | None
    token_ids: np.ndarray | None
    positions: np.ndarray | None
    token_relays: np.ndarray
    finish_ids: np.ndarray | None
    candidates: int
    draws: np.ndarray | None
    params: np.ndarray | None
    host_tokens: np.ndarray
    host_positions: np.ndarray
    logit_rows: np.ndarray
    prompt: np.ndarray
    prompt_top: int

    @property
    def logit_count(self) -> int:
        """Return the candidate logits rows the forward projects."""
        return len(self.logit_rows)


@dataclass(frozen=True, slots=True)
class Executed:
    """How a forward ran: its mode and its live and padding tokens."""

    mode: GraphMode
    live_tokens: int
    padded_tokens: int


class TextRunner:
    """One computation binding (or expert microbatch) of a causal LM.

    Args:
        owner: Names the runner in errors.
        kernel: The numerical runner on the binding's stream.
        bounds: The computation's bounds.
        buffers: The binding's columns.
        future_input_tokens: ``int64 [S, 1]`` the slot state's next tokens.
        logical_lengths: ``int64 [S]`` the slot state's logical positions.
        prompt_logits: ``[S, vocab]`` the slot state's retained logits.
        stream: The binding's stream; ``None`` for a computation on the host.
        pool: The binding's graph pool; ``None`` serves every forward
            eagerly at its live shape.
    """

    def __init__(
        self,
        *,
        owner: str,
        kernel: CausalLMRunner,
        bounds: TextBounds,
        buffers: InputBuffers,
        future_input_tokens: torch.Tensor,
        logical_lengths: torch.Tensor,
        prompt_logits: torch.Tensor,
        stream: torch.cuda.Stream | None,
        pool: Any | None,
    ) -> None:
        self.owner = owner
        self.kernel = kernel
        self.bounds = bounds
        self.buffers = buffers
        self.dims = buffers.dimensions
        self.shapes = bounds.shapes()
        self.tables: tuple[TableShape, ...] = bounds.tables
        self.stream = stream
        self.pool = pool
        self.graphs = GraphTable(owner)
        self.staging = Staging(buffers.arena.device)
        self.plans_attention = bool(kernel.plans_attention())
        self._future_tokens = future_input_tokens[:, 0]
        self._logical_lengths = logical_lengths
        self._prompt_logits = prompt_logits
        # The commit the computation's last executed batch left for this
        # runner.
        self.pending: CommitWork | None = None
        # The graph pool an eager runner set aside (:meth:`run_eager`).
        self._eager_pool: Any | None = None

    def run_eager(self, eager: bool) -> None:
        """Serve every forward eagerly at its live shape (``eager``), or by
        the graphs captured at startup again.

        The graphs stay captured: the worker gives their memory back while
        the runner is eager and restores it at the same addresses before the
        runner replays them again.
        """  # noqa: D205
        if eager and self.pool is not None:
            self._eager_pool, self.pool = self.pool, None
        elif not eager and self._eager_pool is not None:
            self.pool, self._eager_pool = self._eager_pool, None

    # Planning.

    def plan(self, invocation: Invocation) -> Plan:
        """Resolve the graph, the padded lengths and the host columns.

        Raises:
            ValueError: The rows exceed the binding, a decode row evaluates
                more than one token, a source contradicts the forward's
                kind, no captured graph holds the forward under graphs, or
                a table selects more pages than its width.
        """
        rows = invocation.rows
        kind = int(invocation.kind)
        decode = kind == CallKind.DECODE
        dims = self.dims
        live = len(rows)
        if live == 0 or live > dims.rows:
            raise ValueError(
                f"text invocation has {live} rows; the binding holds "
                f"{dims.rows}"
            )
        rows.validate()
        live_tokens = int(np.sum(rows.query))
        if live_tokens > dims.tokens:
            raise ValueError(
                f"text invocation has {live_tokens} tokens; the binding "
                f"holds {dims.tokens}"
            )
        if decode and np.any(rows.query != 1):
            raise ValueError("decode rows evaluate exactly one token")
        starts = np.asarray(invocation.starts, dtype=np.int64)
        counts = np.asarray(invocation.counts, dtype=np.int64)
        if (
            starts.shape != (len(self.tables), live)
            or counts.shape != starts.shape
        ):
            raise AssertionError(
                "attention pages disagree with the binding's tables"
            )

        request = text_forward(rows, decode, bool(invocation.embeddings))
        variant = request.variant
        key: GraphKey | None = None
        padded_rows, padded_tokens = live, live_tokens
        if self.pool is not None:
            # Graph selection depends only on call metadata, so every rank
            # of a group selects the same graph; the catalogs hold every
            # batch the worker admits.
            selected = self.shapes.select(request)
            if selected is None:
                name = CallKind(kind).name.lower()
                raise ValueError(
                    f"no captured text graph holds a {name} forward of "
                    f"{live} rows and {live_tokens} tokens ({variant})"
                )
            if selected.decode:
                key = DecodeKey(
                    selected.rows, invocation.row_sampling is not None
                )
            else:
                key = PrefillKey(
                    selected.rows, selected.tokens, selected.outputs, variant
                )
            padded_rows, padded_tokens = selected.rows, selected.tokens
        graphed = key is not None
        if padded_rows > dims.rows or padded_tokens > dims.tokens:
            raise AssertionError("graph bucket exceeds the binding's columns")

        widths = graph_widths(
            [page_width(table) for table in counts],
            list(self.bounds.widths) if graphed else None,
            decode,
            dims.width,
        )
        padded = pad_rows(rows, starts, decode, padded_rows, padded_tokens)
        if invocation.refill is not None:
            # A refill row writes no cache of the target.
            padded.writes[:live] &= ~np.asarray(invocation.refill, dtype=bool)

        finish_ids = None
        sampled_candidates = 0
        draws = params = None
        host_tokens = np.zeros((2, 0), dtype=np.int64)
        host_positions = np.zeros((2, 0), dtype=np.int64)
        if decode:
            finish_ids = np.full((padded_rows, FINISH_IDS), -1, dtype=np.int64)
            if invocation.finish_ids is not None:
                finish_ids[: len(invocation.finish_ids)] = invocation.finish_ids
            if invocation.row_sampling is not None:
                sampled_candidates = candidates(dims.vocab)
                draws, params = invocation.row_sampling.padded(padded_rows)
            source = rows.source
            if np.any(source == RowSource.PLACED):
                raise ValueError("a decode row places no positions")
            # A row whose token or position comes from the host has it
            # written to its slot before the forward gathers it.
            host = np.flatnonzero(source == RowSource.HOST)
            written = np.flatnonzero(
                (source == RowSource.HOST) | (source == RowSource.RELAY)
            )
            host_tokens = np.stack(
                [rows.slot[host], rows.tokens[rows.token_start[host]]]
            ).astype(np.int64)
            host_positions = np.stack(
                [rows.slot[written], rows.position[written]]
            ).astype(np.int64)

        row_columns = token_ids = positions = None
        token_relays = np.zeros((2, 0), dtype=np.int64)
        axes = variant.axes
        if not decode:
            row_columns = pack_rows(
                padded.slots,
                padded.prefixes,
                padded.queries,
                padded.writes,
                list(counts),
                dims.rows + 1,
                padded_tokens,
            )
            token_ids, positions, token_relays = self._tokens(
                rows, axes, padded_tokens
            )

        # The token positions the selection projects into candidate rows:
        # every position of a verification row, and of a prefill row that
        # scores its prompt.
        prompt = invocation.prompt
        if kind == CallKind.VERIFY or (prompt is not None and len(prompt)):
            logit_rows = logit_positions(rows.query, rows.logit_rows())
        else:
            logit_rows = np.zeros(0, dtype=np.int64)
        prompt_table = (
            prompt_rows(prompt, logit_rows, dims.rows)
            if prompt is not None and len(prompt)
            else np.zeros((0, PROMPT_FIELDS), dtype=np.int64)
        )

        # A forward that only writes the cache keeps no outputs, unless the
        # graph serving it computes them or the drafter reads its hidden
        # states.
        outputs = (
            not request.cache_only
            or (isinstance(key, PrefillKey) and key.outputs)
            or self.drafts()
        )
        # A graph of the media variant reads the device columns whatever
        # its rows hold; an eager forward reads them for rows that differ in
        # causality, and for vision features.
        media_graph = graphed and variant.embeddings
        causal = padded.causal.astype(np.int64)
        device_causality = media_graph or bool(np.any(causal != causal[0]))
        return Plan(
            kind=kind,
            rows=padded_rows,
            live_tokens=live_tokens,
            tokens=padded_tokens,
            widths=widths,
            padded=padded,
            queries=padded.queries.tolist(),
            prefixes=padded.prefixes.tolist(),
            starts=padded.starts.reshape(-1).tolist(),
            causal=causal.tolist(),
            device_causality=device_causality,
            embeddings=media_graph or bool(invocation.embeddings),
            axes=axes,
            outputs=bool(outputs),
            verification=kind == CallKind.VERIFY
            and bool(np.any(rows.query > 1)),
            key=key,
            row_columns=row_columns,
            token_ids=token_ids,
            positions=positions,
            token_relays=token_relays,
            finish_ids=finish_ids,
            candidates=sampled_candidates,
            draws=draws,
            params=params,
            host_tokens=host_tokens,
            host_positions=host_positions,
            logit_rows=logit_rows,
            prompt=prompt_table,
            prompt_top=prompt.top if prompt is not None else 0,
        )

    @staticmethod
    def _tokens(
        rows: Rows, axes: int, padded_tokens: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a prefill's tokens, axis-major positions and token relays.

        A host row's tokens follow each other from its position; a relay
        row's first token is its slot's committed next token, which the
        device copies in, and its drafts follow; a placed row carries its
        positions on every axis.
        """
        live_tokens = int(np.sum(rows.query))
        token_ids = np.zeros(padded_tokens, dtype=np.int64)
        positions = np.zeros((axes, padded_tokens), dtype=np.int64)
        source = rows.source
        if np.any(source == RowSource.INDEXED):
            raise ValueError(
                "only a decode row reads its position from its slot"
            )
        # Each live token's row and its index within the row.
        offsets = np.cumsum(rows.query) - rows.query
        owner = np.repeat(np.arange(len(rows), dtype=np.int64), rows.query)
        within = np.arange(live_tokens, dtype=np.int64) - offsets[owner]
        relay = source[owner] == RowSource.RELAY
        # A relay row's host tokens are its drafts, from its second token.
        host = ~relay | (within > 0)
        index = rows.token_start[owner] + within - relay.astype(np.int64)
        token_ids[:live_tokens][host] = rows.tokens[index[host]]
        positions[0, :live_tokens] = rows.position[owner] + within
        for row, placed in rows.placed.items():
            start = int(offsets[row])
            count = int(rows.query[row])
            positions[:, start : start + count] = placed[:axes]
        relayed = np.flatnonzero(source == RowSource.RELAY)
        token_relays = np.stack([offsets[relayed], rows.slot[relayed]]).astype(
            np.int64
        )
        return token_ids, positions, token_relays

    # Staging.

    def stage(self, plan: Plan, invocation: Invocation) -> None:
        """Write ``plan``'s inputs into the binding's columns on its stream.

        Order on the stream: the token sets are cleared, the host columns
        are copied, then the nonzero set words, the relayed tokens and the
        decode rows' host slot state are written, the embedding mask is
        set and the vision features copied in.

        Raises:
            ValueError: Vision features disagree with the embedding column.
        """
        buffers = self.buffers
        stage = Stage()
        if plan.kind == CallKind.DECODE:
            inputs = buffers.decode(plan.rows)
            region, views = host_region(
                decode_layout(plan.rows), plan.rows * DECODE_ROW_BYTES
            )
            views["slots"][:] = plan.padded.slots
            views["force"][:] = plan.padded.force
            views["finish_ids"][:] = plan.finish_ids
            if plan.candidates:
                views["draws"][:] = plan.draws
                views["params"][:] = plan.params
            stage.put(inputs.region, region)
            self._put_pairs(
                stage, "slot_writes", SlotWrite.TOKENS, plan.host_tokens
            )
            self._put_pairs(
                stage, "slot_writes", SlotWrite.POSITIONS, plan.host_positions
            )
        else:
            assert plan.row_columns is not None and plan.token_ids is not None
            assert plan.positions is not None
            stage.put(
                buffers["row_columns"][: len(plan.row_columns)],
                plan.row_columns,
            )
            stage.put(buffers["input_ids"][: plan.tokens], plan.token_ids)
            for axis in range(plan.axes):
                stage.put(
                    buffers["positions"][axis, : plan.tokens],
                    plan.positions[axis],
                )
            if plan.device_causality:
                stage.put(
                    buffers["causal_values"][: plan.rows],
                    np.asarray(plan.causal, dtype=np.int32),
                )
            self._put_pairs(
                stage, "relays", Relay.PREFILL_TOKENS, plan.token_relays
            )
        if plan.logit_count:
            stage.put(
                buffers["logit_rows"][: plan.logit_count], plan.logit_rows
            )

        scatter = 0
        if invocation.sampling is not None:
            scatter = self._stage_selection(
                stage,
                invocation.sampling,
                verification=plan.kind == CallKind.VERIFY,
            )
        if len(plan.prompt):
            block = self.dims.prompt_block(plan.prompt_top)
            fields = prompt_columns(plan.prompt, block, self.dims.rows)
            for index, values in enumerate(fields):
                stage.put(buffers["prompt_rows"][index, : len(values)], values)
        relays = invocation.relays
        if relays is not None:
            # Consecutive token relays copy a slice; others index pairs.
            if not relays.consecutive:
                self._put_pairs(
                    stage,
                    "relays",
                    Relay.TOKENS,
                    np.stack([relays.token_calls, relays.token_tasks]),
                )
            self._put_pairs(
                stage,
                "relays",
                Relay.TRANSITIONS,
                np.stack([relays.transition_calls, relays.transition_tasks]),
            )

        with torch.cuda.stream(self.stream):
            if invocation.sampling is not None:
                # Token sets start empty; the set words follow.
                buffers["token_sets"].zero_()
            self.staging.submit(stage, self.stream)
            if scatter:
                buffers["token_sets"].view(-1).index_copy_(
                    0,
                    buffers["set_indices"][:scatter],
                    buffers["set_values"][:scatter],
                )
            relayed = plan.token_relays.shape[1]
            if relayed:
                offsets, slots = self.pairs(
                    "relays", Relay.PREFILL_TOKENS, relayed
                )
                buffers["input_ids"].index_copy_(
                    0, offsets, self._future_tokens.index_select(0, slots)
                )
            written = plan.host_tokens.shape[1]
            if written:
                slots, values = self.pairs(
                    "slot_writes", SlotWrite.TOKENS, written
                )
                self._future_tokens.index_copy_(0, slots, values)
            written = plan.host_positions.shape[1]
            if written:
                slots, values = self.pairs(
                    "slot_writes", SlotWrite.POSITIONS, written
                )
                self._logical_lengths.index_copy_(0, slots, values)
            if plan.embeddings:
                self._place_features(plan, invocation.embeddings)

    def _put_pairs(
        self, stage: Stage, name: str, block: int, pairs: np.ndarray
    ) -> None:
        # Stage ``[2, n]`` index pairs into block ``block`` of a pair column
        # as one contiguous copy (:meth:`pairs`).
        count = pairs.shape[1]
        if count:
            stage.put(self.pairs(name, block, count), pairs.astype(np.int64))

    def pairs(self, name: str, block: int, count: int) -> torch.Tensor:
        """Return the first ``count`` index pairs of a pair column's block.

        A pair column (``slot_writes``, :class:`~.buffers.SlotWrite`, or
        ``relays``, :class:`~.buffers.Relay`) is ``[2 * blocks, R]``: block
        ``block`` holds ``[2, n]`` pairs contiguously from word
        ``2 * R * block``.
        """
        base = 2 * self.dims.rows * block
        return (
            self.buffers[name].view(-1)[base : base + 2 * count].view(2, count)
        )

    def _stage_selection(
        self,
        stage: Stage,
        metadata: SamplingMetadata,
        verification: bool = False,
    ) -> int:
        """Stage a selection's inputs; return the set words to scatter.

        The nonzero token-set words are scattered after the copies; 0 when
        the sets are copied whole. A ``verification`` selects from its
        candidate columns, even with no task holding drafts.
        """
        buffers = self.buffers
        padded = pad_selection(metadata, self.dims.rows)
        self._stage_tasks(stage, padded, width=metadata.width)
        if metadata.width > 1 or verification:
            # A verification's candidate rows, tasks times width.
            count = len(padded.rows)
            stage.put(buffers["candidate_rows"][:count], padded.rows)
            stage.put(buffers["candidate_draws"][:count], padded.draws)
            stage.put(buffers["candidate_params"][:count], padded.params)
            stage.put(buffers["candidate_mask_rows"][:count], padded.mask_rows)
            stage.put(
                buffers["select_drafts"][: len(padded.drafts)], padded.drafts
            )
        if verification:
            # Each reported row's task, candidate row and top count: a row
            # per position each reporting task may commit.
            reported = len(metadata.logprob_tasks)
            if reported:
                assert metadata.logprobs is not None
                stage.put(
                    buffers["verified_logprobs"][:, :reported],
                    np.stack(
                        [
                            metadata.logprob_tasks,
                            metadata.logprob_offsets,
                            metadata.logprobs.counts,
                        ]
                    ),
                )
        else:
            reporting = len(padded.logprob_tasks)
            stage.put(
                buffers["logprob_tasks"][:reporting], padded.logprob_tasks
            )
            stage.put(
                buffers["logprob_counts"][:reporting], padded.logprob_counts
            )
        if len(metadata.masks):
            stage.put(
                buffers["select_masks"][: len(metadata.masks)], metadata.masks
            )

        # Each task's finish and transition sets: their nonzero words are
        # scattered into the cleared column, or copied whole when the
        # scatter column does not hold them.
        tasks = metadata.tasks
        words = self.dims.set_words()
        finish = metadata.finish.reshape(-1)
        transitions = metadata.transitions.reshape(-1)
        listed = np.flatnonzero(finish)
        changing = np.flatnonzero(transitions)
        total = len(listed) + len(changing)
        if total <= self.dims.scatter_words():
            if total:
                second = self.dims.rows * words
                stage.put(
                    buffers["set_indices"][:total],
                    np.concatenate([listed, second + changing]).astype(
                        np.int64
                    ),
                )
                stage.put(
                    buffers["set_values"][:total],
                    np.concatenate(
                        [finish[listed], transitions[changing]]
                    ).astype(np.int32),
                )
            return total
        stage.put(buffers["token_sets"][0, :tasks], metadata.finish)
        stage.put(buffers["token_sets"][1, :tasks], metadata.transitions)
        return 0

    def _stage_tasks(
        self, stage: Stage, padded: PaddedSelection, width: int
    ) -> None:
        # The per-task fields of a selection in the select region of its
        # task bucket; a verification's per-candidate fields go to the
        # candidate columns instead.
        tasks = padded.tasks
        region, views = host_region(
            select_layout(tasks), tasks * SELECT_ROW_BYTES
        )
        views["tasks"][0] = padded.slots
        views["tasks"][1] = padded.gates
        views["paths"][:] = padded.paths
        views["terminal"][:] = padded.terminal
        views["biased"][:] = padded.biased
        views["force"][:] = padded.force
        if width == 1:
            views["rows"][:] = padded.rows
            views["mask_rows"][:] = padded.mask_rows
            views["params"][:] = padded.params
            views["draws"][:] = padded.draws
        stage.put(self.buffers.select(tasks).region, region)

    def _place_features(
        self, plan: Plan, embeddings: Sequence[tuple[int, torch.Tensor]]
    ) -> None:
        # Vision features replace their tokens' embeddings: the mask is
        # cleared over the forward, then set over each block.
        buffers = self.buffers
        values = buffers["input_embeddings"]
        mask = buffers["embedding_mask"]
        mask[: plan.tokens].zero_()
        for token, features in embeddings:
            count = int(features.shape[0]) if features.ndim else 0
            if (
                features.dtype != values.dtype
                or features.ndim != 2
                or int(features.shape[1]) != int(values.shape[1])
                or token + count > plan.live_tokens
            ):
                raise ValueError(
                    "vision features disagree with the embedding column"
                )
            values[token : token + count].copy_(features)
            mask[token : token + count].fill_(1)

    # Forwards.

    def _method(self, plan: Plan, capacity: int | None) -> Callable[[], None]:
        """The runner's forward method of ``plan`` at ``capacity``."""
        buffers = self.buffers
        capacity = int(capacity or 0)
        if plan.kind == CallKind.DECODE:
            inputs = buffers.decode(plan.rows)
            return partial(
                self.kernel.decode,
                plan.rows,
                plan.widths,
                plan.prefixes,
                plan.starts,
                capacity,
                plan.candidates,
                inputs.slots,
                inputs.force,
                inputs.finish_ids,
                buffers["logits"],
                inputs.draws,
                inputs.params,
                buffers["completion"],
                buffers["continuation"],
                buffers["tagged"],
                buffers["points"],
                buffers["transition"],
            )
        embeddings = plan.embeddings
        return partial(
            self.kernel.prefill,
            plan.rows,
            plan.tokens,
            plan.widths,
            plan.queries,
            plan.prefixes,
            plan.starts,
            plan.causal,
            plan.axes,
            int(plan.outputs),
            capacity,
            buffers["row_columns"],
            buffers["input_ids"],
            buffers["positions"],
            buffers["causal_values"] if plan.device_causality else None,
            buffers["input_embeddings"] if embeddings else None,
            buffers["embedding_mask"] if embeddings else None,
            buffers["hidden"],
            buffers["logits"],
            int(plan.verification),
        )

    def _plan_attention(self, plan: Plan) -> None:
        """Build the host attention plans a captured forward reads."""
        buffers = self.buffers
        if plan.kind == CallKind.DECODE:
            self.kernel.plan_decode(
                plan.rows,
                plan.widths,
                plan.prefixes,
                plan.starts,
                buffers.decode(plan.rows).slots,
            )
            return
        self.kernel.plan_prefill(
            plan.rows,
            plan.tokens,
            plan.widths,
            plan.queries,
            plan.prefixes,
            plan.starts,
            plan.causal,
            buffers["row_columns"],
            buffers["causal_values"] if plan.device_causality else None,
        )

    def forward(
        self, plan: Plan, capacity: int | None = None, launch: bool = True
    ) -> tuple[GraphMode, Callable[[], None] | None]:
        """Run ``plan``'s forward at expert step ``capacity``.

        Replays the resident graph of the plan's bucket (after building the
        host attention plans when an operator plans on the host), or runs
        the method eagerly: without a graph pool, without ``launch`` (an
        expert microbatch that may not launch graphs yet), or as the warmup
        of a graph not yet captured. In that last case the capture is
        returned: the caller performs it behind the forward's selection,
        and it records without executing.

        Raises:
            ValueError: The computation is sealed and the bucket has no graph.
        """
        method = self._method(plan, capacity)
        key = plan.key
        with torch.cuda.stream(self.stream):
            if key is None:
                method()
                return GraphMode.EAGER, None
            graph = self.graphs.lookup(key, capacity)
            if graph is not None:
                if not launch:
                    method()
                    return GraphMode.EAGER, None
                if self.plans_attention:
                    self._plan_attention(plan)
                graph.replay()
                return GraphMode.REPLAY, None
            method()

        def capture() -> None:
            with torch.cuda.stream(self.stream):
                # The plans a capture reads are built outside it.
                self._plan_attention(plan)
            self.graphs.insert(key, capacity, self._capture(method))

        return GraphMode.CAPTURE, capture

    # Selection.

    def _alone(self, key: GraphKey, call: Callable[[], None]) -> None:
        """Launch ``key``'s resident graph, or run ``call`` without a pool.

        Selections are captured at startup, so a missing graph is an error.
        """
        if self.pool is None:
            call()
            return
        graph = self.graphs.lookup(key)
        if graph is None:
            raise AssertionError(
                f"{self.owner}: the {key!r} graph is captured at startup"
            )
        graph.replay()

    def _select_call(self, key: SelectKey) -> Callable[[], None]:
        buffers = self.buffers
        inputs = buffers.select(key.tasks)
        return partial(
            self.kernel.select,
            key.tasks,
            candidates(self.dims.vocab),
            int(key.top_k),
            int(key.categorical is not None),
            int(bool(key.categorical)),
            buffers["logits"],
            inputs.rows,
            inputs.paths,
            inputs.draws,
            inputs.params,
            inputs.tasks,
            buffers["select_masks"],
            inputs.mask_rows,
            inputs.biased,
            inputs.force,
            buffers["token_sets"][0],
            buffers["token_sets"][1],
            buffers["completion"],
            buffers["continuation"],
            buffers["tagged"],
            buffers["points"],
            buffers["transition"],
            buffers["scored"],
        )

    def _score_call(self, key: ScoreKey) -> Callable[[], None]:
        buffers = self.buffers
        return partial(
            self.kernel.score,
            key.rows,
            key.top,
            self.dims.logprob_words(),
            buffers["logits"],
            buffers["logprob_tasks"],
            buffers["logprob_counts"],
            buffers["scored"],
            buffers["completion"],
            buffers["logprobs"],
        )

    def _prompt_call(self, key: PromptKey) -> Callable[[], None]:
        buffers = self.buffers
        return partial(
            self.kernel.score_prompt,
            key.rows,
            key.top,
            self.dims.logprob_words(),
            buffers["hidden"],
            buffers["prompt_rows"],
            buffers["prompt_block"],
            buffers["input_ids"],
            buffers["logprobs"],
            buffers["prompt_tail"],
        )

    # Drafts.

    def drafts(self) -> bool:
        """Whether the runner runs the drafter after its forwards."""
        return bool(self.dims.drafts)

    def draft(self, plan: Plan, draft: DraftRows) -> None:
        """Run ``plan``'s forward's draft pass over ``draft``'s rows.

        Stages the rows' packed columns, next tokens, positions and hidden
        rows, and the windows its proposers' chained steps evaluate, then
        runs the drafter (:meth:`CausalLMRunner.draft`) on the runner's
        stream. A proposer's window starts after its base position and
        holds every position a chained step takes whatever the drafts its
        verification accepts: the base's drafts after it, then a position
        per further proposal. The proposers' drafts are left in the
        ``proposals`` column, in proposer order.
        """
        buffers = self.buffers
        dims = self.dims
        stride = dims.rows + 1
        count = len(draft.queries)
        tokens = int(draft.queries.sum())
        if count == 0 or tokens == 0:
            return
        proposers = len(draft.proposers) if draft.proposals else 0
        steps = draft.proposals - 1 if proposers else 0
        if draft.proposals > dims.drafts:
            raise AssertionError("a call proposes more drafts than the runner")
        positions = np.repeat(draft.prefixes, draft.queries) + (
            np.arange(tokens)
            - np.repeat(np.cumsum(draft.queries) - draft.queries, draft.queries)
        )
        stage = Stage()
        stage.put(
            buffers["draft_row_columns"],
            pack_rows(
                draft.slots,
                draft.prefixes,
                draft.queries,
                np.ones(count, dtype=bool),
                list(draft.pages),
                stride,
                dims.tokens,
            ),
        )
        stage.put(buffers["draft_input_ids"][:tokens], draft.tokens)
        stage.put(buffers["draft_positions"][:tokens], positions)
        stage.put(buffers["draft_hidden_rows"][:tokens], draft.hidden_rows)
        sampled = draft.pairs.shape[1]
        if sampled:
            stage.put(buffers["draft_pairs"][:, :sampled], draft.pairs)
        window_queries: list[int] = []
        window_prefixes: list[int] = []
        windows = 0
        if proposers:
            index = draft.proposers
            starts = np.cumsum(draft.queries) - draft.queries
            stage.put(
                buffers["draft_proposing"][:, :proposers],
                np.stack([starts[index] + draft.bases, draft.tasks]),
            )
            # A window starts after the proposer's base position and spans
            # its row's positions after the base and a position per
            # further proposal.
            prefixes = draft.prefixes[index] + draft.bases + 1
            queries = draft.queries[index] - 1 - draft.bases + steps
            windows = int(queries.sum())
            if steps and windows:
                if windows > dims.window_tokens():
                    raise AssertionError(
                        "the drafter's windows exceed their columns"
                    )
                window_queries = [int(value) for value in queries]
                window_prefixes = [int(value) for value in prefixes]
                first = np.cumsum(queries) - queries
                stage.put(
                    buffers["draft_window_columns"],
                    pack_rows(
                        draft.slots[index],
                        prefixes,
                        queries,
                        np.ones(proposers, dtype=bool),
                        list(draft.pages[:, index]),
                        stride,
                        dims.tokens,
                    ),
                )
                stage.put(buffers["draft_window_tokens"][:proposers], first)
                stage.put(
                    buffers["draft_window_owners"][:windows],
                    np.repeat(np.arange(proposers, dtype=np.int64), queries),
                )
                stage.put(
                    buffers["draft_window_positions"][:windows],
                    np.repeat(prefixes, queries)
                    + np.arange(windows)
                    - np.repeat(first, queries),
                )
        with torch.cuda.stream(self.stream):
            self.staging.submit(stage, self.stream)
            self.kernel.draft(
                count,
                tokens,
                plan.widths,
                draft.queries.tolist(),
                draft.prefixes.tolist(),
                sampled,
                proposers,
                steps,
                windows,
                window_queries,
                window_prefixes,
                buffers["draft_row_columns"],
                buffers["draft_window_columns"],
                buffers["draft_input_ids"],
                buffers["draft_positions"],
                buffers["draft_hidden_rows"],
                buffers["draft_pairs"],
                buffers["draft_proposing"],
                buffers["draft_window_tokens"],
                buffers["draft_window_owners"],
                buffers["draft_window_positions"],
                buffers["hidden"],
                buffers["completion"],
                buffers["proposals"],
            )

    def _verify(self, plan: Plan, metadata: SamplingMetadata) -> None:
        buffers = self.buffers
        inputs = buffers.select(metadata.tasks)
        reported = len(metadata.logprob_tasks)
        _, top = metadata.logprob_bucket
        self.kernel.verify(
            plan.logit_count,
            metadata.tasks,
            metadata.width,
            candidates(self.dims.vocab),
            int(metadata.sorts),
            top,
            reported,
            self.dims.logprob_words(),
            buffers["hidden"],
            buffers["logit_rows"],
            buffers["candidate_rows"],
            buffers["candidate_draws"],
            buffers["candidate_params"],
            inputs.tasks,
            buffers["select_masks"],
            buffers["candidate_mask_rows"],
            inputs.biased,
            inputs.force,
            buffers["token_sets"][0],
            buffers["token_sets"][1],
            buffers["select_drafts"],
            inputs.terminal,
            buffers["verified_logprobs"],
            buffers["completion"],
            buffers["continuation"],
            buffers["tagged"],
            buffers["points"],
            buffers["transition"],
            buffers["scored"],
            buffers["logprobs"],
        )

    def select(self, plan: Plan, invocation: Invocation) -> None:
        """Select, score and score prompts behind ``plan``'s forward.

        The sampled tasks select with the ``select`` graph of their task
        bucket and samplers, reporting tasks score with the ``score`` graph
        of their logprob bucket, and the prompt rows score one block per
        ``score_prompt`` launch, the block's first row written to the block
        column before it. A verification projects and selects eagerly,
        which graphs refuse.

        Raises:
            NotImplementedError: A verification under graphs.
            AssertionError: A selection graph was not captured at startup.
        """
        metadata = invocation.sampling
        with torch.cuda.stream(self.stream):
            if metadata is not None:
                if plan.kind == CallKind.VERIFY:
                    if self.pool is not None:
                        raise NotImplementedError(
                            "a verification's selection has no captured graph"
                        )
                    self._verify(plan, metadata)
                else:
                    key = SelectKey(
                        task_bucket(metadata.tasks, self.dims.rows),
                        metadata.top_k,
                        metadata.sorts if metadata.categorical else None,
                    )
                    self._alone(key, self._select_call(key))
                    if metadata.logprobs is not None:
                        rows, top = metadata.logprob_bucket
                        score = ScoreKey(rows, top)
                        self._alone(score, self._score_call(score))
            if len(plan.prompt):
                top = plan.prompt_top
                rows = self.dims.prompt_block(top)
                prompt = PromptKey(rows, top)
                block = self.buffers["prompt_block"]
                for first in range(0, len(plan.prompt), rows):
                    block.fill_(first)
                    self._alone(prompt, self._prompt_call(prompt))

    # Commit.

    def commit(self, work: CommitWork) -> None:
        """Install ``work``'s commit rows and prompt tails into the slot state.

        The rows pad to their commit bucket and replay its ``commit`` graph;
        before the computation is sealed a bucket without one commits
        eagerly and is captured. Without a pool the rows commit eagerly.
        Then each scored chunk's retained logits are copied into its slot.
        Everything here is past the batch's commit point.
        """
        rows = len(work.rows)
        tails = len(work.tails)
        if rows == 0 and tails == 0:
            return
        capacity = self.dims.rows
        if rows > capacity:
            raise AssertionError("commit rows exceed the binding")
        key = (
            CommitKey(commit_bucket(rows, self.shapes.decode, capacity))
            if self.pool is not None
            else None
        )
        count = key.rows if key is not None else rows
        stage = Stage()
        if rows:
            # The column's rows are field-major at a stride of the binding's
            # rows; the bucket's columns are written whole.
            columns = np.empty((COMMIT_FIELDS, capacity), dtype=np.int64)
            columns[:, :count] = work.columns(count)
            columns[:, count:] = 0
            stage.put(self.buffers["commit"], columns)
        if tails:
            self._put_pairs(stage, "relays", Relay.PROMPT_TAILS, work.tails.T)
        with torch.cuda.stream(self.stream):
            self.staging.submit(stage, self.stream)
            if rows:
                method = partial(
                    self.kernel.commit,
                    count,
                    self.buffers["commit"],
                    self.buffers["completion"],
                    self.buffers["continuation"],
                    self.buffers["points"],
                )
                self._commit(key, method)
            if tails:
                tail_rows, slots = self.pairs(
                    "relays", Relay.PROMPT_TAILS, tails
                )
                self._prompt_logits.index_copy_(
                    0,
                    slots,
                    self.buffers["prompt_tail"]
                    .index_select(0, tail_rows)
                    .to(self._prompt_logits.dtype),
                )

    def _commit(
        self, key: CommitKey | None, method: Callable[[], None]
    ) -> None:
        if key is None:
            method()
            return
        graph = self.graphs.lookup(key)
        if graph is not None:
            graph.replay()
            return
        # The eager commit installs these rows and compiles the kernel; the
        # capture records without executing.
        method()
        self.graphs.insert(key, None, self._capture(method))

    def _capture(self, method: Callable[[], None]):
        """Capture ``method`` into the binding's graph pool on its stream.

        Raises:
            AssertionError: The computation has no graph pool: it serves
                eagerly, as a computation on the host always does.
        """
        if self.stream is None or self.pool is None:
            raise AssertionError(
                "only a device computation with a graph pool captures"
            )
        return cuda_graph.capture(method, stream=self.stream, pool=self.pool)

    # Startup.

    def capture_alone(self, key: GraphKey) -> None:
        """Run a selection, scoring or prompt scoring eagerly, then capture it.

        Startup captures them over the inputs of padding tasks and prompt
        rows (:meth:`reset_selection`): their values do not choose kernels.
        """
        if isinstance(key, SelectKey):
            method = self._select_call(key)
            self._stage_padding(key.tasks)
        elif isinstance(key, ScoreKey):
            method = self._score_call(key)
        elif isinstance(key, PromptKey):
            method = self._prompt_call(key)
        else:
            raise AssertionError(f"{key!r} is no selection graph")
        with torch.cuda.stream(self.stream):
            method()
        self.graphs.insert(key, None, self._capture(method))

    def _stage_padding(self, tasks: int) -> None:
        # Padding tasks in the select region of their bucket.
        stage = Stage()
        self._stage_tasks(stage, padding_selection(tasks), width=1)
        self.staging.submit(stage, self.stream)

    def reset_selection(self) -> None:
        """Clear the selection's outputs and write padding inputs.

        The logits, token sets, scored rows, hidden states and the block
        column are zeroed; the logprob tasks report nothing and every
        prompt row pads.
        """
        buffers = self.buffers
        dims = self.dims
        padding = padding_selection(dims.rows)
        stage = Stage()
        stage.put(buffers["logprob_tasks"], padding.logprob_tasks)
        stage.put(buffers["logprob_counts"], padding.logprob_counts)
        stage.put(
            buffers["prompt_rows"],
            np.tile(
                prompt_padding(dims.rows)[:, None], (1, dims.prompt_rows())
            ),
        )
        with torch.cuda.stream(self.stream):
            for name in (
                "logits",
                "select_masks",
                "token_sets",
                "scored",
                "hidden",
                "prompt_block",
            ):
                buffers[name].zero_()
            self.staging.submit(stage, self.stream)

    def close(self, aborted: bool) -> None:
        """Release the graphs, then the execution context; the stream drained.

        An aborted close keeps both: device work may still address them.
        """
        try:
            self.graphs.close(aborted)
        finally:
            if not aborted:
                self.kernel.close()
                self.staging.close()
