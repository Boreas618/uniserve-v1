"""A text computation: one causal language model component's token batches.

The worker hands a computation each token batch it carries
(``Executor.execute``): host columns of its calls and of the scheduler's
forward rows with their attention pages, the calls' flat tokens, a small
JSON of rare per-call entries (sampling states, vision and latent
features), the record column and the output words it lends. The
computation (:meth:`TextComputation.execute`):

1. checks the calls against the requests their slots hold and builds every
   active call's rows: one row per text call (:func:`~.rows.prepare_calls`),
   a context prefill's prompt runs and vision blocks, a visual-state row
   (:mod:`.context`); the rows must be the scheduler's forward rows, whose
   pages the worker selected;
2. splits the calls with rows into one invocation per expert microbatch
   and lays out each invocation's sampled tasks, the decode forward's own
   selection or the sampling inputs of a selection, and its prompt chunks;
3. has the image denoiser write latent conditionings' K/V, plans, stages,
   runs and selects every invocation (as one expert step with
   microbatches), and captures the graphs a warmup forward left to capture;
4. copies each invocation's completion block, logprob words and prompt
   scores into the lent words, relays sampled tokens and transition
   decisions into the computation's staging columns, writes the calls'
   completion records and keeps each microbatch's commit rows for
   :meth:`TextComputation.commit`.

Nothing here advances slot state before the worker sealed the batch:
:meth:`TextComputation.commit` does, past the batch's commit point.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .buffers import Relay, TextBounds
from .commit import CommitWork, finish_rows
from .context import (
    ConditionRow,
    ImageDenoiser,
    ImageFacts,
    MediaCall,
    VisionInput,
    concat_rows,
    context,
    host_rows,
    placed_rows,
    take_rows,
    visual,
)
from .philox import DrawWindow
from .prompt import PromptChunk, PromptChunks, media_chunks, prompt_chunk
from .protocol import (
    CALL_FIELDS,
    COMPLETION_FIELDS,
    CallField,
    CallFlag,
    CallKind,
    ForwardField,
    GraphMode,
    Predicate,
    RecordWriter,
    combine_modes,
    forward_fields,
    page_count,
    page_start,
)
from .rows import (
    Prepared,
    Rows,
    TokenSelection,
    logit_layout,
    prepare_calls,
)
from .runner import Executed, Invocation, Plan, Relays, TextRunner
from .sampling import (
    FINISH_IDS,
    OwnSelection,
    RowSampling,
    SampleTask,
    SamplingMetadata,
    SamplingState,
    controls_for_call,
    own_selection,
    sampling_paths,
    task_draws,
)
from .state import RequestTable, SlotState

if TYPE_CHECKING:
    from uniserve.executor.experts import ExpertUnion

__all__ = ["TextComputation"]

_MEDIA = CallFlag.WRITES_CONTEXT | CallFlag.WRITES_VISUAL_STATE


@dataclass(slots=True)
class _Batch:
    """One batch's active calls and their rows, before invocations."""

    calls: np.ndarray
    forward: np.ndarray
    tokens: np.ndarray
    kind: int
    active: np.ndarray
    states: dict[int, SamplingState]
    # Text calls: their prepared rows, each call's position among them (-1
    # for others) and their pages, ``[groups, n]``.
    prepared: Prepared | None
    text_position: np.ndarray
    text_starts: np.ndarray
    text_counts: np.ndarray
    # Media calls: their rows, pages and whether they score their prompt.
    media: dict[int, MediaCall]
    media_pages: dict[int, tuple[np.ndarray, np.ndarray]]
    media_scores: dict[int, bool]
    # Latent conditionings and their pages, ``[2 * groups, n]``.
    conditions: list[ConditionRow]
    condition_pages: list[np.ndarray]
    # The slot and ``uint8 [1]`` device predicate of each call a predicate
    # gates (:attr:`Predicate.DEVICE`).
    predicates: list[tuple[int, torch.Tensor]]
    # Per call of the batch: its task column and microbatch.
    task: np.ndarray = field(init=False)
    microbatch: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        count = self.calls.shape[1]
        self.task = np.full(count, -1, dtype=np.int64)
        self.microbatch = np.zeros(count, dtype=np.int64)


@dataclass(slots=True)
class _Planned:
    """One microbatch's invocation and what its records need."""

    microbatch: int
    runner: TextRunner
    calls: np.ndarray
    invocation: Invocation
    # Each task's call, in task order.
    task_calls: np.ndarray
    # A verification's drafts and terminal prefix, by call.
    speculation: dict[int, tuple[tuple[int, ...], int | None]]
    plan: Plan | None = None
    mode: GraphMode = GraphMode.EAGER
    capture: Callable[[], None] | None = None


class TextComputation:
    """The token batches of one causal language model component on a rank.

    Args:
        index: The computation's index among the executor's computations.
        group: The computation's index among the rank's text computations:
            the kind its expert steps agree under.
        label: The kernel-table label of the computation's runners.
        runners: One runner per expert microbatch, the first on the primary
            stream.
        bounds: The computation's bounds.
        state: The rank's slot state.
        table: The rank's request constants.
        window: The rank's cached draws.
        relay_tokens: ``int64 [calls]`` the worker's staging of sampled
            tokens, by call.
        relay_flags: ``uint8 [2, calls]`` the worker's staging of
            transition decisions (row 0), by call.
        union: The rank's expert union member, or ``None``.
        prefill_outputs: The computation's prefills keep their outputs.
    """

    def __init__(
        self,
        *,
        index: int,
        group: int,
        label: str,
        runners: Sequence[TextRunner],
        bounds: TextBounds,
        state: SlotState,
        table: RequestTable,
        window: DrawWindow,
        relay_tokens: torch.Tensor,
        relay_flags: torch.Tensor,
        union: ExpertUnion | None,
        prefill_outputs: bool,
    ) -> None:
        if not runners:
            raise ValueError(
                "a text computation runs on one runner per microbatch"
            )
        self.index = index
        self.group = group
        self.label = label
        self.runners = tuple(runners)
        self.bounds = bounds
        self.state = state
        self.table = table
        self.window = window
        self.relay_tokens = relay_tokens
        self.relay_flags = relay_flags
        self.union = union
        self.prefill_outputs = bool(prefill_outputs)
        self.facts: ImageFacts | None = None
        self.denoiser: ImageDenoiser | None = None
        # Each table's cache group.
        self._table_groups = np.array(
            [table.group for table in bounds.tables], dtype=np.int64
        )

    @property
    def stream(self) -> torch.cuda.Stream | None:
        """The primary stream, which the first runner computes on; ``None`` on
        the host.
        """  # noqa: D205
        return self.runners[0].stream

    def tables_of(self, groups: np.ndarray) -> np.ndarray:
        """Expand per-group page columns ``[groups, n]`` to every table."""
        return np.asarray(groups, dtype=np.int64)[self._table_groups]

    # A batch.

    def execute(
        self,
        step: Mapping[str, Any],
        calls: torch.Tensor,
        forward: torch.Tensor,
        tokens: torch.Tensor,
        records: torch.Tensor,
        words: torch.Tensor,
        features: Sequence[torch.Tensor],
    ) -> int:
        """Run one token batch; return the record words written.

        Args:
            step: The batch's JSON: ``calls`` holds, by call index, the rare
                entries (``sampling_state``, ``vision``, ``latent``).
            calls: ``int64 [CALL_FIELDS, C]`` host call columns.
            forward: ``int64 [FORWARD_FIELDS + 2G, R]`` host forward rows
                with pages.
            tokens: ``int64 [N]`` host flat tokens.
            records: ``int64 [K]`` host record column, written.
            words: ``int64 [W]`` page-locked output words, written on the
                primary stream.
            features: The feature tensors the JSON names by position.

        Raises:
            ValueError: A call or row is malformed or disagrees with the
                requests, the forward rows or the binding.
            AssertionError: Records or output words exceed their columns.
        """
        call_columns = calls.numpy()
        forward_columns = forward.numpy()
        if call_columns.shape[0] != CALL_FIELDS:
            raise ValueError(
                "a token batch's call columns have the wrong fields"
            )
        if forward_columns.shape[0] != forward_fields(len(self.bounds.groups)):
            raise ValueError(
                "a token batch's forward columns have the wrong groups"
            )
        writer = RecordWriter(records.numpy(), call_columns.shape[1])
        lent = int(words.numel())
        # Each execute replaces what an earlier, failed batch left.
        for runner in self.runners:
            runner.pending = None
        active = np.flatnonzero(call_columns[CallField.ACTIVE] != 0)
        if len(active) == 0:
            return writer.finish(
                words_written=0,
                lent=lent,
                mode=GraphMode.NONE,
                live_tokens=0,
                padded_tokens=0,
                rows=0,
                draft_tokens=0,
            )

        batch = self._batch(
            step,
            call_columns,
            forward_columns,
            tokens.numpy(),
            active,
            features,
        )
        # Calls with rows, in call order, split into contiguous microbatches.
        if batch.media:
            inputs = np.array(
                [
                    call
                    for call in active.tolist()
                    if batch.text_position[call] >= 0
                    or (call in batch.media and batch.media[call].rows)
                ],
                dtype=np.int64,
            )
        else:
            inputs = active[batch.text_position[active] >= 0]
        planned: list[_Planned] = []
        count = len(self.runners)
        total = len(inputs)
        for microbatch in range(count):
            span = inputs[
                microbatch * total // count : (microbatch + 1) * total // count
            ]
            if len(span):
                planned.append(self._invocation(batch, microbatch, span))
        for item in planned:
            item.plan = item.runner.plan(item.invocation)

        # The image denoiser writes latent conditionings' K/V before the
        # forwards that read them.
        if batch.conditions:
            if self.denoiser is None:
                raise NotImplementedError(
                    "latent conditioning requires an image denoiser"
                )
            self.denoiser.condition_latents(
                batch.conditions,
                np.concatenate(batch.condition_pages, axis=1),
                self.stream,
            )
        # A predicated call's predicate becomes its slot's continuation
        # predicate, which its gated rows read on the device; nothing waits
        # for it on the host. A false one leaves the call's selection
        # stopped: its slot state does not continue.
        if batch.predicates:
            with torch.cuda.stream(self.stream):
                for slot, predicate in batch.predicates:
                    self.state.predicates[slot : slot + 1].copy_(predicate)
        for item in planned:
            assert item.plan is not None
            item.runner.stage(item.plan, item.invocation)
        capacity = self._run(planned)

        written = self._outputs(planned, batch, writer, words, lent)
        self._finish(batch, planned, writer)

        # Every planned invocation ran a forward; a batch of latent
        # conditionings alone ran none and reports no mode.
        plans = [item.plan for item in planned if item.plan is not None]
        live_tokens = sum(plan.live_tokens for plan in plans)
        rows = sum(len(item.invocation.rows) for item in planned)
        # A verification row's query tokens after its first are its drafts;
        # the worker reads the count of verification batches only.
        drafts = sum(
            int(np.sum(np.maximum(item.invocation.rows.query - 1, 0)))
            for item in planned
        )
        return writer.finish(
            words_written=written,
            lent=lent,
            mode=combine_modes(int(item.mode) for item in planned),
            live_tokens=live_tokens,
            padded_tokens=sum(plan.tokens for plan in plans) - live_tokens,
            rows=rows,
            draft_tokens=drafts,
            expert_capacity=capacity,
        )

    def _batch(
        self,
        step: Mapping[str, Any],
        calls: np.ndarray,
        forward: np.ndarray,
        tokens: np.ndarray,
        active: np.ndarray,
        features: Sequence[torch.Tensor],
    ) -> _Batch:
        """Check the active calls and build their rows and pages."""
        kinds = calls[CallField.KIND, active]
        kind = int(kinds[0])
        if np.any(kinds != kind):
            raise ValueError("a token batch serves one call kind")
        if kind not in (CallKind.PREFILL, CallKind.DECODE, CallKind.VERIFY):
            raise ValueError("only prefill, decode and verify are text calls")
        slots = calls[CallField.SLOT, active]
        self.table.check(
            slots,
            calls[CallField.REQUEST_ID, active],
            calls[CallField.EPOCH, active],
        )
        entries: Mapping[str, Any] = step.get("calls", {})
        flags = calls[CallField.FLAGS, active]
        states: dict[int, SamplingState] = {}
        for call in active[(flags & CallFlag.SAMPLING_STATE) != 0]:
            states[int(call)] = SamplingState.from_json(
                entries[str(int(call))]["sampling_state"]
            )

        groups = len(self.bounds.groups)
        media = (flags & _MEDIA) != 0
        text = active[~media]
        position = np.full(calls.shape[1], -1, dtype=np.int64)
        prepared = None
        starts = counts = np.zeros((groups, 0), dtype=np.int64)
        if len(text):
            text_slots = calls[CallField.SLOT, text]
            # A call's grammar state may force it to finish.
            force = None
            if states:
                force = np.array(
                    [
                        int(call) in states and states[int(call)].force_finish
                        for call in text
                    ],
                    dtype=bool,
                )
            prepared = prepare_calls(
                calls,
                tokens,
                text,
                self.table.scores_prompt[text_slots],
                force,
            )
            position[text] = np.arange(len(text))
            # A text call has exactly one forward row, which must be the
            # row Python built: the worker selected its pages.
            if np.any(calls[CallField.ROW_COUNT, text] != 1):
                raise ValueError("a token call has exactly one forward row")
            row = calls[CallField.ROW_START, text]
            columns = forward[:, row]
            rows = prepared.rows
            query = columns[ForwardField.QUERY_LEN]
            if (
                np.any(columns[ForwardField.REQUEST_POOL_INDEX] != rows.slot)
                or np.any(query != rows.query)
                or np.any(columns[ForwardField.SEQ_LEN] - query != rows.prefix)
            ):
                raise ValueError(
                    "text rows disagree with the batch's forward rows"
                )
            starts = np.stack([columns[page_start(g)] for g in range(groups)])
            counts = np.stack([columns[page_count(g)] for g in range(groups)])

        batch = _Batch(
            calls=calls,
            forward=forward,
            tokens=tokens,
            kind=kind,
            active=active,
            states=states,
            prepared=prepared,
            text_position=position,
            text_starts=starts,
            text_counts=counts,
            media={},
            media_pages={},
            media_scores={},
            conditions=[],
            condition_pages=[],
            predicates=[],
        )
        for call in active[media]:
            self._media(
                batch, int(call), entries.get(str(int(call)), {}), features
            )
        gated = active[calls[CallField.PREDICATE, active] == Predicate.DEVICE]
        for call in map(int, gated):
            entry = entries.get(str(call), {}).get("predicate")
            if entry is None:
                raise ValueError("a predicated call names no predicate tensor")
            batch.predicates.append(
                (
                    int(calls[CallField.SLOT, call]),
                    features[int(entry["tensor"])],
                )
            )
        return batch

    def _media(
        self,
        batch: _Batch,
        call: int,
        entry: Mapping[str, Any],
        features: Sequence[torch.Tensor],
    ) -> None:
        """Build a media call's rows and check them against its forward rows."""
        column = batch.calls[:, call]
        slot = int(column[CallField.SLOT])
        flags = int(column[CallField.FLAGS])
        start = int(column[CallField.TOKEN_START])
        tokens = batch.tokens[
            start : start + int(column[CallField.TOKEN_COUNT])
        ]
        vision = [
            VisionInput.from_json(value, features)
            for value in entry.get("vision", ())
        ]
        latent = entry.get("latent")
        common = {
            "slot": slot,
            "logical": int(column[CallField.LOGICAL]),
            "visible": int(column[CallField.VISIBLE]),
            "allocated": int(column[CallField.ALLOCATED]),
            "input_images": int(self.table.input_images[slot]),
        }
        token_output = bool(flags & CallFlag.TOKEN_OUTPUT)
        if flags & CallFlag.WRITES_CONTEXT:
            scores = bool(self.table.scores_prompt[slot])
            built = context(
                self.facts,
                tokens=np.asarray(tokens, dtype=np.int64),
                vision=sorted(vision, key=lambda block: block.offset),
                scores=scores,
                token_output=token_output,
                **common,
            )
        else:
            scores = False
            built = visual(
                self.facts,
                self.denoiser,
                vision=vision[0] if vision else None,
                latent=None
                if latent is None
                else VisionInput(
                    offset=0,
                    features=features[int(latent["tensor"])],
                    height=int(latent["height"]),
                    width=int(latent["width"]),
                ),
                max_tokens=int(column[CallField.MAX_TOKENS]),
                token_output=token_output,
                completion_output=bool(flags & CallFlag.COMPLETION_OUTPUT),
                **common,
            )

        first = int(column[CallField.ROW_START])
        described = int(column[CallField.ROW_COUNT])
        rows = batch.forward[:, first : first + described]
        groups = len(self.bounds.groups)
        starts = np.stack([rows[page_start(g)] for g in range(groups)])
        counts = np.stack([rows[page_count(g)] for g in range(groups)])
        if built.latent is not None:
            if described < 1:
                raise ValueError("a latent conditioning has no forward row")
            pages = np.empty((2 * groups, 1), dtype=np.int64)
            pages[0::2, 0] = starts[:, 0]
            pages[1::2, 0] = counts[:, 0]
            batch.conditions.append(built.latent)
            batch.condition_pages.append(pages)
        else:
            # The scheduler's forward rows of the call describe the same
            # rows: one per prompt run and vision block, on the call's slot,
            # each with its query.
            queries = np.array(
                [row.query for row in built.rows], dtype=np.int64
            )
            if (
                described != len(built.rows)
                or np.any(rows[ForwardField.REQUEST_POOL_INDEX] != slot)
                or np.any(rows[ForwardField.QUERY_LEN] != queries)
                or int(queries.sum()) > int(column[CallField.MAX_TOKENS])
            ):
                raise ValueError(
                    "context rows disagree with the call's forward rows"
                )
            batch.media_pages[call] = (starts, counts)
        batch.media[call] = built
        batch.media_scores[call] = scores

    def _invocation(
        self, batch: _Batch, microbatch: int, calls: np.ndarray
    ) -> _Planned:
        """Lay out one microbatch's invocation of ``calls``."""
        runner = self.runners[microbatch]
        batch.microbatch[calls] = microbatch
        prepared = batch.prepared
        media = bool(batch.media) and any(
            int(call) in batch.media for call in calls
        )
        embeddings: list[tuple[int, torch.Tensor]] = []
        if not media:
            assert prepared is not None
            index = batch.text_position[calls]
            rows = (
                prepared.rows
                if len(index) == len(prepared.rows)
                and np.array_equal(index, np.arange(len(index)))
                else take_rows(prepared.rows, index)
            )
            starts = batch.text_starts[:, index]
            counts = batch.text_counts[:, index]
            first_row = np.arange(len(calls), dtype=np.int64)
            row_count = np.ones(len(calls), dtype=np.int64)
        else:
            parts: list[Rows] = []
            start_parts: list[np.ndarray] = []
            count_parts: list[np.ndarray] = []
            spans: list[int] = []
            tokens = 0
            for call in map(int, calls):
                if call in batch.media:
                    built = batch.media[call]
                    part = placed_rows(built.rows)
                    # A predicated media call's rows are gated as a
                    # predicated text call's row is.
                    predicate = batch.calls[CallField.PREDICATE, call]
                    if predicate == Predicate.DEVICE:
                        part.gated = np.ones(len(part), dtype=bool)
                    page_starts, page_counts = batch.media_pages[call]
                    embeddings.extend(
                        (tokens + token, values)
                        for token, values in built.embeddings
                    )
                else:
                    assert prepared is not None
                    index = [int(batch.text_position[call])]
                    part = take_rows(prepared.rows, index)
                    # The row keeps only its own host tokens.
                    part.tokens = part.host_tokens(0).copy()
                    part.token_start = np.zeros(1, dtype=np.int64)
                    page_starts = batch.text_starts[:, index]
                    page_counts = batch.text_counts[:, index]
                parts.append(part)
                start_parts.append(page_starts)
                count_parts.append(page_counts)
                spans.append(len(part))
                tokens += int(np.sum(part.query))
            rows = concat_rows(parts)
            starts = np.concatenate(start_parts, axis=1)
            counts = np.concatenate(count_parts, axis=1)
            row_count = np.array(spans, dtype=np.int64)
            first_row = np.cumsum(row_count) - row_count

        offsets = np.cumsum(rows.query) - rows.query
        first_token = offsets[first_row]
        planned = _Planned(
            microbatch=microbatch,
            runner=runner,
            calls=calls,
            invocation=Invocation(
                kind=batch.kind,
                rows=rows,
                starts=self.tables_of(starts),
                counts=self.tables_of(counts),
                embeddings=embeddings,
            ),
            task_calls=np.zeros(0, dtype=np.int64),
            speculation={},
        )
        rare = bool(batch.states) and any(
            int(call) in batch.states for call in calls
        )
        if not media and not rare and batch.kind != CallKind.VERIFY:
            self._columnar_tasks(batch, planned, first_token)
        else:
            self._call_tasks(batch, planned, first_row, row_count, first_token)
        return planned

    def _columnar_tasks(
        self, batch: _Batch, planned: _Planned, first_token: np.ndarray
    ) -> None:
        """Lay out the tasks of one-row calls without a sampling state.

        Every column is an array expression over the calls' slots against
        the request constants: the decode hot path.
        """
        invocation = planned.invocation
        runner = planned.runner
        table = self.table
        rows = invocation.rows
        columns = batch.calls[:, planned.calls]
        slots = columns[CallField.SLOT]
        flags = columns[CallField.FLAGS]
        token_output = (flags & CallFlag.TOKEN_OUTPUT) != 0
        transition_output = (flags & CallFlag.TRANSITION_OUTPUT) != 0
        decode = batch.kind == CallKind.DECODE
        count = len(slots)
        # Prefill and decode sample after their last token; a prefill
        # without a token output only extends the cache.
        sampled = np.ones(count, dtype=bool) if decode else token_output
        tasks = np.flatnonzero(sampled)
        task_slots = slots[tasks]
        positions = columns[CallField.LOGICAL, tasks] + rows.query[tasks]
        draws = task_draws(table, self.window, task_slots, positions)

        own = None
        if decode:
            nothing = np.zeros(count, dtype=bool)
            own = own_selection(
                sampled=sampled,
                decode=np.ones(count, dtype=bool),
                tagged=rows.gated,
                token_output=token_output,
                transition_output=transition_output,
                transitions=nothing,
                finish_count=table.finish_count[slots],
                paths=sampling_paths(table, slots),
                suppressed=nothing,
            )
        if own is not None:
            invocation.finish_ids = table.finish_ids[slots]
            if own == OwnSelection.TOP_K:
                invocation.row_sampling = RowSampling(
                    draws=draws, params=table.params[task_slots]
                )
        elif len(tasks):
            # A call's task row is its row: the forward projects each row's
            # last logits.
            invocation.sampling = SamplingMetadata.gather(
                table,
                task_slots,
                tasks,
                rows.gated[tasks],
                draws,
                runner.dims.rows,
                runner.dims.logprob_words(),
            )

        chunks: list[PromptChunk] = []
        if batch.kind == CallKind.PREFILL:
            first, rows_of = logit_layout(rows.selection, rows.query)
            scored = np.flatnonzero(sampled & table.scores_prompt[slots])
            for position in scored:
                slot = int(slots[position])
                chunks.append(
                    prompt_chunk(
                        int(planned.calls[position]),
                        slot,
                        int(first[position]),
                        int(rows_of[position]),
                        int(first_token[position]),
                        int(columns[CallField.LOGICAL, position]),
                        bool(columns[CallField.PROMPT_READY, position]),
                        int(table.n_prompt_logprobs[slot]),
                    )
                )
        self._finish_tasks(
            batch,
            planned,
            tasks,
            token_output,
            transition_output,
            sampled,
            chunks,
        )

    def _call_tasks(
        self,
        batch: _Batch,
        planned: _Planned,
        first_row: np.ndarray,
        row_count: np.ndarray,
        first_token: np.ndarray,
    ) -> None:
        """Lay out the tasks of calls resolved one by one.

        Calls with a sampling state, verifications and media calls.
        """
        invocation = planned.invocation
        runner = planned.runner
        table = self.table
        vocab = table.vocab
        rows = invocation.rows
        columns = batch.calls[:, planned.calls]
        count = len(planned.calls)
        kind = batch.kind
        flags = columns[CallField.FLAGS]
        token_output = (flags & CallFlag.TOKEN_OUTPUT) != 0
        transition_output = (flags & CallFlag.TRANSITION_OUTPUT) != 0
        first, rows_of = logit_layout(rows.selection, rows.query)
        prepared = batch.prepared

        sampled = np.zeros(count, dtype=bool)
        paths = np.zeros(count, dtype=np.int64)
        suppressed = np.zeros(count, dtype=bool)
        transitions = np.zeros(count, dtype=bool)
        finish_count = np.zeros(count, dtype=np.int64)
        finish_ids = np.full((len(rows), FINISH_IDS), -1, dtype=np.int64)
        tasks: list[SampleTask] = []
        task_positions: list[int] = []
        chunks: list[PromptChunk] = []
        for position, call in enumerate(map(int, planned.calls)):
            slot = int(columns[CallField.SLOT, position])
            logical = int(columns[CallField.LOGICAL, position])
            last = int(first_row[position] + row_count[position] - 1)
            built = batch.media.get(call)
            drafts: tuple[int, ...] = ()
            if built is not None:
                points = [built.sample] if token_output[position] else None
            elif kind == CallKind.PREFILL:
                points = (
                    [logical + int(rows.query[last])]
                    if token_output[position]
                    else None
                )
            elif kind == CallKind.DECODE:
                points = [logical + int(rows.query[last])]
            else:
                assert prepared is not None
                index = int(batch.text_position[call])
                begin = int(prepared.draft_start[index])
                drafts = tuple(
                    int(token)
                    for token in batch.tokens[
                        begin : begin + int(prepared.draft_count[index])
                    ]
                )
                points = list(range(logical + 1, logical + len(drafts) + 2))
            if points is None:
                continue
            sampled[position] = True
            params = table.sampling[slot]
            state = batch.states.get(call)
            controls, draws = controls_for_call(
                params,
                state,
                table.finish_tokens.get(slot, ()),
                points,
                drafts,
                int(table.key[slot]),
                bool(transition_output[position]),
            )
            paths[position] = controls.path(params, vocab)
            suppressed[position] = bool(controls.suppress)
            transitions[position] = state is not None and bool(
                state.transition_token_ids
            )
            finish_count[position] = len(controls.finish_token_ids)
            listed = controls.finish_token_ids[:FINISH_IDS]
            finish_ids[last, : len(listed)] = listed
            # A verification samples its candidate rows, one per position;
            # the others the last-position logits of their row.
            row = int(first[last]) if kind == CallKind.VERIFY else last
            if (
                kind == CallKind.PREFILL
                and built is None
                and bool(table.scores_prompt[slot])
            ):
                chunks.append(
                    prompt_chunk(
                        call,
                        slot,
                        int(first[last]),
                        int(rows_of[last]),
                        int(first_token[position]),
                        logical,
                        bool(columns[CallField.PROMPT_READY, position]),
                        int(table.n_prompt_logprobs[slot]),
                    )
                )
            if kind == CallKind.VERIFY:
                planned.speculation[call] = (
                    controls.draft_token_ids,
                    controls.terminal_draft_prefix,
                )
            tasks.append(
                SampleTask(
                    params=params,
                    controls=controls,
                    draws=draws,
                    slot=slot,
                    row=row,
                    gated=bool(rows.gated[last]),
                )
            )
            task_positions.append(position)

        # A prompt placing vision blocks scores each prompt run; a run after
        # a block continues from the block's last logits.
        for position, call in enumerate(map(int, planned.calls)):
            if not batch.media_scores.get(call, False):
                continue
            slot = int(columns[CallField.SLOT, position])
            span = slice(
                int(first_row[position]),
                int(first_row[position] + row_count[position]),
            )
            chunks.extend(
                media_chunks(
                    call,
                    slot,
                    first[span],
                    rows_of[span],
                    rows.causal[span],
                    rows.query[span],
                    int(first_token[position]),
                    int(columns[CallField.LOGICAL, position]),
                    bool(columns[CallField.PROMPT_READY, position]),
                    int(table.n_prompt_logprobs[slot]),
                )
            )

        own = None
        if kind == CallKind.DECODE:
            own = own_selection(
                sampled=sampled,
                decode=np.ones(count, dtype=bool),
                tagged=rows.gated,
                token_output=token_output,
                transition_output=transition_output,
                transitions=transitions,
                finish_count=finish_count,
                paths=paths,
                suppressed=suppressed,
            )
        if own is not None:
            invocation.finish_ids = finish_ids
            if own == OwnSelection.TOP_K:
                invocation.row_sampling = RowSampling.build(tasks, vocab)
        elif tasks:
            invocation.sampling = SamplingMetadata.build(
                tasks, vocab, runner.dims.rows, runner.dims.logprob_words()
            )
        self._finish_tasks(
            batch,
            planned,
            np.asarray(task_positions, dtype=np.int64),
            token_output,
            transition_output,
            sampled,
            chunks,
        )

    def _finish_tasks(
        self,
        batch: _Batch,
        planned: _Planned,
        tasks: np.ndarray,
        token_output: np.ndarray,
        transition_output: np.ndarray,
        sampled: np.ndarray,
        chunks: list[PromptChunk],
    ) -> None:
        """Record each call's task, the prompt chunks and the relays."""
        invocation = planned.invocation
        calls = planned.calls
        task_column = np.full(len(calls), -1, dtype=np.int64)
        task_column[tasks] = np.arange(len(tasks), dtype=np.int64)
        batch.task[calls] = task_column
        planned.task_calls = calls[tasks]
        if chunks:
            sampling = invocation.sampling
            invocation.prompt = PromptChunks.plan(
                chunks,
                sampling.logprob_end() if sampling is not None else 0,
                self.table.vocab,
                planned.runner.dims.logprob_words(),
            )
        # Every call with a token output samples; a call with a transition
        # output that does not sample relays no transition.
        relayed = token_output & sampled
        changing = transition_output
        invocation.relays = Relays(
            token_calls=calls[relayed],
            token_tasks=task_column[relayed],
            transition_calls=calls[changing],
            transition_tasks=task_column[changing],
        )

    def _run(self, planned: Sequence[_Planned]) -> int:
        """Run every invocation's forward and selection; captures follow.

        Returns the expert step's agreed capacity, 0 without experts. A
        batch of latent conditionings alone runs no forward.
        """
        if not planned:
            return 0
        if self.union is None:
            (item,) = planned
            assert item.plan is not None
            item.mode, item.capture = item.runner.forward(item.plan)
            item.runner.select(item.plan, item.invocation)
            if item.capture is not None:
                item.capture()
            return 0

        forwards: list[Callable[[int], None] | None] = [None] * len(
            self.runners
        )
        for item in planned:
            forwards[item.microbatch] = self._microbatch(item, self.union)
        tokens = max(
            item.plan.tokens for item in planned if item.plan is not None
        )
        capacity = self.union.step(forwards, tokens, self.group)
        for item in planned:
            if item.capture is not None:
                item.capture()
        return int(capacity)

    def _microbatch(
        self, item: _Planned, union: ExpertUnion
    ) -> Callable[[int], None]:
        # The expert step's forward of one microbatch; it keeps how the
        # forward ran, and the capture it left, on the item.
        def forward(capacity: int) -> None:
            assert item.plan is not None
            item.mode, item.capture = item.runner.forward(
                item.plan, capacity, launch=union.replays()
            )
            item.runner.select(item.plan, item.invocation)

        return forward

    def _outputs(
        self,
        planned: Sequence[_Planned],
        batch: _Batch,
        writer: RecordWriter,
        words: torch.Tensor,
        lent: int,
    ) -> int:
        """Copy the invocations' results into the lent words and relays.

        Returns the words written. Every copy is enqueued on the primary
        stream, which waited for every microbatch's work.
        """
        cursor = 0
        prompt: dict[int, list[tuple[int, int]]] = {}

        def reserve(count: int) -> int:
            nonlocal cursor
            if cursor + count > lent:
                raise AssertionError(
                    "a batch's outputs exceed its output words"
                )
            start = cursor
            cursor += count
            return start

        with torch.cuda.stream(self.stream):
            for item in planned:
                buffers = item.runner.buffers
                invocation = item.invocation
                tasks = len(item.task_calls)
                if tasks:
                    # The tasks' completion block, field-major.
                    extent = COMPLETION_FIELDS * tasks
                    start = reserve(extent)
                    words[start : start + extent].view(
                        COMPLETION_FIELDS, tasks
                    ).copy_(buffers["completion"][:, :tasks], non_blocking=True)
                    writer.sampling(
                        item.task_calls,
                        start,
                        extent,
                        np.arange(tasks, dtype=np.int64),
                    )
                sampling = invocation.sampling
                chunks = invocation.prompt
                end = (
                    chunks.end
                    if chunks is not None
                    else sampling.logprob_end()
                    if sampling is not None
                    else 0
                )
                # The sampled tasks' compact words, then each chunk's.
                base = reserve(end)
                if end:
                    words[base : base + end].copy_(
                        buffers["logprobs"][:end], non_blocking=True
                    )
                    if sampling is not None and sampling.logprobs is not None:
                        layout = sampling.logprobs
                        segment = writer.segment(
                            base, layout.words, layout.rows, layout.counts
                        )
                        reporting = sampling.logprob_tasks
                        writer.logprobs(
                            item.task_calls[reporting],
                            segment,
                            np.arange(len(reporting), dtype=np.int64),
                        )
                if chunks is not None:
                    for chunk, call in enumerate(chunks.calls.tolist()):
                        ranges = prompt.setdefault(call, [])
                        scored = chunks.layout(chunk)
                        if scored is None:
                            continue
                        segment = writer.segment(
                            base + int(chunks.offsets[chunk]),
                            int(chunks.sizes[chunk]),
                            scored.rows,
                            scored.counts,
                        )
                        ranges.extend(
                            (segment, row) for row in range(len(scored.rows))
                        )
                self._relay(item)
        # Tails follow every segment: prompt ranges and speculation.
        speculation = {
            call: value
            for item in planned
            for call, value in item.speculation.items()
        }
        for call in sorted(set(prompt) | set(speculation)):
            drafts, terminal = speculation.get(call, (None, None))
            writer.tail(
                call,
                prompt=prompt.get(call),
                drafts=drafts,
                terminal=terminal,
            )
        return cursor

    def _relay(self, item: _Planned) -> None:
        """Write an invocation's relayed values into the relay staging."""
        relays = item.invocation.relays
        if relays is None:
            return
        buffers = item.runner.buffers
        count = len(relays.token_calls)
        if count:
            if relays.consecutive:
                first, task = (
                    int(relays.token_calls[0]),
                    int(relays.token_tasks[0]),
                )
                self.relay_tokens[first : first + count].copy_(
                    buffers["tagged"][task : task + count]
                )
            else:
                calls, tasks = item.runner.pairs("relays", Relay.TOKENS, count)
                self.relay_tokens.index_copy_(
                    0, calls, buffers["tagged"].index_select(0, tasks)
                )
        count = len(relays.transition_calls)
        if count:
            calls, tasks = item.runner.pairs("relays", Relay.TRANSITIONS, count)
            values = buffers["transition"].index_select(0, tasks.clamp_min(0))
            if np.any(relays.transition_tasks < 0):
                values = values * (tasks >= 0)
            self.relay_flags[0].index_copy_(0, calls, values.to(torch.uint8))

    def _finish(
        self, batch: _Batch, planned: Sequence[_Planned], writer: RecordWriter
    ) -> None:
        """Record every active call's progress; keep the commit rows.

        Each microbatch's runner keeps its calls' commit rows until the
        computation commits.
        """
        calls = batch.calls
        active = batch.active
        count = len(active)
        prefix = np.zeros(count, dtype=np.int64)
        query = np.zeros(count, dtype=np.int64)
        media = np.zeros(count, dtype=bool)
        media_position = np.zeros(count, dtype=np.int64)
        media_continued = np.zeros(count, dtype=np.int64)
        position = batch.text_position[active]
        text = position >= 0
        if batch.prepared is not None:
            prefix[text] = batch.prepared.rows.prefix[position[text]]
            query[text] = batch.prepared.rows.query[position[text]]
        for row in np.flatnonzero(~text).tolist():
            built = batch.media[int(active[row])]
            prefix[row] = built.last.prefix
            query[row] = built.last.query
            media[row] = True
            media_position[row] = built.position
            media_continued[row] = built.continued

        # Each call's chunks, and the tail row its last chunk retains.
        chunked = np.zeros(calls.shape[1], dtype=bool)
        tails: list[list[tuple[int, int]]] = [[] for _ in self.runners]
        for item in planned:
            chunks = item.invocation.prompt
            if chunks is None:
                continue
            last: dict[int, int] = {}
            for chunk, call in enumerate(chunks.calls.tolist()):
                last[call] = chunk
            for call, chunk in last.items():
                chunked[call] = True
                tails[item.microbatch].append(
                    (int(chunks.tails[chunk]), int(calls[CallField.SLOT, call]))
                )

        slots = calls[CallField.SLOT, active]
        finished = finish_rows(
            kind=calls[CallField.KIND, active],
            slot=slots,
            prefix=prefix,
            query=query,
            logical=calls[CallField.LOGICAL, active],
            rng=calls[CallField.RNG, active],
            allocated=calls[CallField.ALLOCATED, active],
            task=batch.task[active],
            penalties=self.table.penalties[slots],
            prompt_ready=calls[CallField.PROMPT_READY, active] != 0,
            chunks=chunked[active],
            media=media,
            media_position=media_position,
            media_continued=media_continued,
        )
        writer.ran(active)
        writer.progress(
            active,
            finished.logical,
            finished.rng,
            finished.visible,
            finished.computed,
            finished.prompt_ready,
        )
        microbatch = batch.microbatch[active]
        for index, runner in enumerate(self.runners):
            mine = microbatch == index
            runner.pending = CommitWork(
                rows=finished.rows[mine],
                tails=np.asarray(tails[index], dtype=np.int64).reshape(-1, 2),
            )

    # After the seal.

    def commit(self) -> None:
        """Install the last executed batch's commit rows and prompt tails.

        Each microbatch's runner commits on its stream, which first orders
        after the primary stream; the primary stream then orders after it.
        """
        primary = self.stream
        for index, runner in enumerate(self.runners):
            work, runner.pending = runner.pending, None
            if work is None:
                continue
            # Microbatches beyond the first run on streams of their own,
            # which only a device computation has.
            other = runner.stream if index else None
            if other is not None and primary is not None:
                other.wait_stream(primary)
            runner.commit(work)
            if other is not None and primary is not None:
                primary.wait_stream(other)

    # Couplings and startup.

    def prefill_cache(
        self, prefixes: Sequence[tuple[int, Sequence[int]]], pages: np.ndarray
    ) -> Executed:
        """Write the K/V of token prefixes, each from position zero of its slot.

        The image executor's guidance prefixes: the prefixes are evaluated
        causally and only their K/V is kept; slot state and the computation's
        pending commit are unchanged. The caller installed the slots'
        tables, retains their extents and orders the primary stream around
        the call.

        Args:
            prefixes: ``(slot, tokens)`` of each prefix.
            pages: ``int64 [2 * G, len(prefixes)]`` each prefix row's first
                selected page and page count per cache group.

        Raises:
            NotImplementedError: The rank is a member of an expert union.
        """
        if self.union is not None:
            raise NotImplementedError(
                "guidance prefixes are not evaluated on an expert union rank"
            )
        rows = host_rows(
            [slot for slot, _ in prefixes],
            [np.asarray(tokens, dtype=np.int64) for _, tokens in prefixes],
            prefix=0,
            selection=TokenSelection.CACHE,
        )
        return self.warm_step(0, rows, CallKind.PREFILL, pages)

    def warm_step(
        self,
        microbatch: int,
        rows: Rows,
        kind: int,
        pages: np.ndarray,
        *,
        row_sampling: RowSampling | None = None,
        embeddings: Sequence[tuple[int, torch.Tensor]] = (),
        capacity: int | None = None,
    ) -> Executed:
        """Run one forward of ``rows`` without sampling on ``microbatch``.

        The forward of a warmup step or of guidance prefixes: it selects
        nothing; a decode with ``row_sampling`` samples its rows, else
        selects greedily. Its graph is captured when it has none yet. With
        experts the forward is an expert step forced to ``capacity`` that
        the other microbatches join.

        Args:
            pages: ``int64 [2 * G, rows]`` each row's first page and page
                count per cache group.
        """
        runner = self.runners[microbatch]
        pages = np.asarray(pages, dtype=np.int64)
        invocation = Invocation(
            kind=kind,
            rows=rows,
            starts=self.tables_of(pages[0::2]),
            counts=self.tables_of(pages[1::2]),
            row_sampling=row_sampling,
            embeddings=list(embeddings),
        )
        plan = runner.plan(invocation)
        runner.stage(plan, invocation)
        item = _Planned(
            microbatch=microbatch,
            runner=runner,
            calls=np.zeros(0, dtype=np.int64),
            invocation=invocation,
            task_calls=np.zeros(0, dtype=np.int64),
            speculation={},
            plan=plan,
        )
        if self.union is None:
            item.mode, item.capture = runner.forward(plan, capacity)
            if item.capture is not None:
                item.capture()
        else:
            forwards: list[Callable[[int], None] | None] = [None] * len(
                self.runners
            )
            forwards[microbatch] = self._microbatch(item, self.union)
            self.union.step(forwards, plan.tokens, self.group, capacity)
            if item.capture is not None:
                item.capture()
        return Executed(
            item.mode, plan.live_tokens, plan.tokens - plan.live_tokens
        )

    def kernels(self) -> list[tuple[str, list[str]]]:
        """Each runner's kernel-table label and selected kernels."""
        return [
            (self.label, runner.kernel.kernels()) for runner in self.runners
        ]

    def seal(self) -> None:
        """Refuse further captures on every runner."""
        for runner in self.runners:
            runner.graphs.seal()

    def run_eager(self, eager: bool) -> None:
        """Serve eagerly (``eager``) or by the captured graphs again, on
        every runner (:meth:`TextRunner.run_eager`).
        """  # noqa: D205
        for runner in self.runners:
            runner.run_eager(eager)

    def reopen(self) -> None:
        """Release every runner's graphs and accept captures again.

        A promoted standby whose expert workspaces were rebuilt captures its
        forwards anew: the released graphs addressed the freed workspaces.
        The caller drained the computation's streams.
        """
        for runner in self.runners:
            runner.graphs.reopen()

    def close(self, aborted: bool) -> None:
        """Release every runner, newest first; its streams are drained.

        Every release is attempted before the first failure is raised.
        """
        failure: BaseException | None = None
        for runner in reversed(self.runners):
            try:
                runner.close(aborted)
            except Exception as error:
                failure = failure or error
        if failure is not None:
            raise failure
