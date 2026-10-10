"""Autoregressive text steps over Rust-owned buffers: the ``text.bind`` entry.

The worker binds one text step function set per lane binding of a causal
language model. Every tensor is a Rust allocation bound once, here; per
step the worker passes only integers and small host arrays (``Shape``).

A step evaluates one homogeneous invocation (prefill, decode or
verification rows of one runner):

1. gather attention metadata on the device from the request slots'
   installed block tables, into the fixed input columns;
2. bind attention plans for the live host lengths;
3. run the model eagerly, or replay a captured graph;
4. project the requested rows' logits, score prompt chunks and sample the
   call's tasks into the bound selection columns.

A commit then installs a batch's selections into the per-slot continuation
state (:mod:`uniserve_models.sampling.state`). The worker owns every
decision about rows, padding, graphs, sampling groups and output layout;
this module computes on what it is given and keeps no request state.

Graph capture records the forward of a step whose inputs an eager step on
the same shapes just prepared; ``capture`` returns the
:class:`~uniserve_models.interface.graphs.Graph` as an opaque handle that
the worker passes back to ``step`` and finally to ``release``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from uniserve_models.model import (
    EmbeddingReplacement,
    Logits,
    TextInput,
    VocabShard,
)
from uniserve_models.nn.attention import (
    AttentionBatch,
    AttentionInput,
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve_models.nn.attention.gather import (
    gather_request_decode_inputs,
    gather_request_rows,
)
from uniserve_models.sampling.greedy import greedy
from uniserve_models.sampling.logprobs import (
    LogprobRequest,
    packed_words,
    prompt_logprobs,
)
from uniserve_models.sampling.select import (
    TOKEN_CONTINUATION_BIT,
    Selection,
    Task,
    decisions,
    select_categorical,
    select_greedy,
    select_top_k,
)
from uniserve_models.sampling.state import commit as commit_state

from . import graphs
from .errors import binding

__all__ = ["bind", "functions"]

# Sampling paths of a task group (``uniserve_worker::SamplingPath``).
GREEDY, GREEDY_SUPPRESSED, TOP_K, CATEGORICAL = 0, 1, 2, 3

# Task flag bits of the sampling descriptor.
FORCE_FINISH = 1 << 0
RETURN_TRANSITION = 1 << 1
GATED = 1 << 2
PENALTIES = 1 << 3
LOGPROBS = 1 << 4
TOP_P = 1 << 5
MIN_P = 1 << 6
TYPICAL_P = 1 << 7

# Fields of one group in the sampling descriptor: path, top-k width, first
# task, task count, logprob word offset, logprob words and first parameter
# row (``uniserve_worker::sampling::plan``).
GROUP_FIELDS = 7


def functions() -> dict[str, Callable[..., Any]]:
    return {"text.describe": describe, "text.bind": bind}


def describe(prepared) -> dict[str, int]:
    """The vocabulary shard and hidden width a text binding computes with.

    The worker sizes the binding's logits columns from this before binding;
    pipeline stages share the last stage's descriptor in a collective.
    """
    model = prepared.module
    _, vocab, _ = bind_vocabulary(model)
    return {
        "vocab_size": int(vocab.size),
        "padded_vocab": int(vocab.padded_size),
        "vocab_start": int(vocab.local_slice.start),
        "vocab_stop": int(vocab.local_slice.stop),
        "hidden_size": int(model.backbone.hidden_size),
    }


def bind_vocabulary(model) -> tuple[Any, VocabShard, Any]:
    """Share the vocabulary descriptor of the last pipeline stage.

    The vocabulary head is resident only on the last stage; its actual
    descriptor is communicated rather than reproducing the head's padding
    and partition rule here.

    Returns:
        ``(pipeline, vocab, tensor)``: the pipeline and tensor-parallel
        communicators and the local vocabulary shard.
    """
    mesh = model.backbone.mesh
    pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
    tensor = mesh.get_group("tp" if "tp" in mesh.axes else ())
    descriptor = torch.empty(4, dtype=torch.int64, device=pipeline.device)
    if pipeline.rank == pipeline.size - 1:
        vocab = model.lm_head.vocab
        descriptor.copy_(
            torch.tensor(
                (
                    vocab.size,
                    vocab.padded_size,
                    vocab.local_slice.start,
                    vocab.local_slice.stop,
                ),
                dtype=torch.int64,
            )
        )
    pipeline.broadcast(descriptor, src=pipeline.size - 1)
    size, padded, start, stop = descriptor.cpu().tolist()
    return (
        pipeline,
        VocabShard(size, slice(start, stop), padded, tensor),
        tensor,
    )


def _offsets(tasks: Sequence[Task]) -> list[int]:
    """Each task's first row within its group's candidate rows."""
    offsets, offset = [], 0
    for task in tasks:
        offsets.append(offset)
        offset += task.rows
    return offsets


class _Cursor:
    """Read a flat host descriptor in order."""

    def __init__(self, words: Sequence[int]) -> None:
        self._words = words
        self._index = 0

    def next(self) -> int:
        value = int(self._words[self._index])
        self._index += 1
        return value

    def take(self, count: int) -> tuple[int, ...]:
        start = self._index
        self._index += count
        return tuple(int(word) for word in self._words[start : self._index])

    def done(self) -> bool:
        return self._index == len(self._words)


def _float(bits: int) -> float:
    """A float32 carried as its bit pattern in an int64 word."""
    return float(
        torch.tensor([bits & 0xFFFFFFFF], dtype=torch.int64)
        .to(torch.int32)
        .view(torch.float32)
        .item()
    )


def decode_tasks(cursor: _Cursor, count: int) -> list[Task]:
    """Decode ``count`` task records of the sampling descriptor.

    A record is ``[row, rows, slot, flags, top_k, logprob_count,
    terminal_prefix, finish_count, transition_count, suppress_count,
    bias_count, logprob_id_count, draft_count, allowed]`` followed by the
    finish, transition and suppressed ids, ``bias_count`` (token, float32
    bits) pairs, the logprob token ids, the draft tokens and, when
    ``allowed`` is set, one ``[length or -1, ids...]`` list per candidate
    row (``-1`` for an unrestricted row).
    """
    tasks = []
    for _ in range(count):
        row, rows, slot, flags, top_k = cursor.take(5)
        logprob_count, terminal = cursor.take(2)
        finish, transition, suppress, bias, ids, drafts, allowed = cursor.take(
            7
        )
        finish_ids = cursor.take(finish)
        transition_ids = cursor.take(transition)
        suppressed = cursor.take(suppress)
        logit_bias = tuple(
            (token, _float(bits))
            for token, bits in zip(
                *[iter(cursor.take(2 * bias))] * 2, strict=True
            )
        )
        logprob_ids = cursor.take(ids)
        draft_ids = cursor.take(drafts)
        restrictions: tuple[tuple[int, ...] | None, ...] = (None,) * rows
        if allowed:
            values = []
            for _ in range(rows):
                length = cursor.next()
                values.append(None if length < 0 else cursor.take(length))
            restrictions = tuple(values)
        tasks.append(
            Task(
                row=row,
                rows=rows,
                slot=slot,
                force_finish=bool(flags & FORCE_FINISH),
                return_transition=bool(flags & RETURN_TRANSITION),
                tagged=bool(flags & GATED),
                penalties=bool(flags & PENALTIES),
                top_k=top_k,
                top_p=bool(flags & TOP_P),
                min_p=bool(flags & MIN_P),
                typical_p=bool(flags & TYPICAL_P),
                finish_ids=finish_ids,
                transition_ids=transition_ids,
                suppress=suppressed,
                logit_bias=logit_bias,
                logprobs=LogprobRequest(logprob_count, logprob_ids)
                if flags & LOGPROBS
                else None,
                drafts=draft_ids,
                terminal_prefix=terminal or None,
                allowed=restrictions,
            )
        )
    return tasks


def bind(
    prepared,
    stream: int,
    buffers: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> dict[str, Callable[..., Any]]:
    """Bind the text steps of a prepared causal-LM binding.

    Args:
        prepared: The ``component.prepare`` handle, already bound
            (``component.bind``): its module is the ``CausalLM`` and its
            execution context holds the operators and the K/V cache.
        stream: The CUDA stream handle every step enqueues on (0 on CPU).
        buffers: Every Rust-owned tensor, by name: the input columns
            (``input_ids``, ``positions``, ``request_pool_indices``,
            ``decode_force_finish``, ``row_columns``, ``block_tables``,
            ``start_pages``, ``cache_lengths``, ``query_lengths``,
            ``causal_values``, ``query_offsets``, ``prefix_offsets``,
            ``write_indices``, ``logit_rows``, ``draws``, ``params``,
            ``tasks``, ``commit``; a multimodal binding also
            ``input_embeddings`` and ``embedding_mask``), the outputs
            (``logits``, optional
            ``hidden``, ``completion``, ``continuation``, ``tagged``,
            ``points``, ``transition``, ``logprobs``, ``prompt_tail``) and the
            per-slot state (``slot.unit_tables``, ``slot.start_pages``,
            ``slot.table_shapes``, ``slot.verified_lengths``,
            ``slot.future_input_tokens``, ``slot.predicates``,
            ``slot.logical_lengths``, ``slot.sampling_positions``,
            ``slot.penalty_counts``, ``slot.prompt_logits``).
        layout: ``{"page_tokens": Shape, "windowed": Shape}`` per numerical
            block table.

    Returns:
        ``{"step", "capture", "release", "commit", "close"}``.
    """
    context = prepared.context
    if context is None:
        raise ValueError("text steps require a bound component")
    model = prepared.module
    b = {str(name): torch.from_dlpack(value) for name, value in buffers.items()}
    for name in (
        "decode_force_finish",
        "continuation",
        "transition",
        "slot.predicates",
    ):
        b[name] = b[name].view(torch.bool)
    page_tokens = tuple(int(value) for value in layout["page_tokens"])
    windowed = tuple(bool(value) for value in layout["windowed"])
    device = b["input_ids"].device
    computation = (
        torch.cuda.ExternalStream(int(stream), device=device)
        if device.type == "cuda"
        else None
    )
    pipeline, vocab, tensor = bind_vocabulary(model)
    last_stage = pipeline.rank == pipeline.size - 1
    width = vocab.local_slice.stop - vocab.local_slice.start
    capacity = int(b["logits"].shape[0])
    if int(b["logits"].shape[1]) != width:
        raise ValueError("logits columns disagree with the vocabulary shard")

    @contextmanager
    def scope() -> Iterator[None]:
        """Enqueue on the binding's stream outside an execution context."""
        with torch.inference_mode():
            if computation is None:
                yield
            else:
                with torch.cuda.stream(computation):
                    yield

    def views(
        rows,
        tokens,
        queries,
        prefixes,
        starts,
        causal,
        writes,
        widths,
        axes=1,
        embedded=False,
    ):
        """Attention and token views of the fixed columns for one call.

        ``causal`` holds each row's causality; when the rows differ, the
        worker staged them in ``causal_values`` for readers that consume the
        device column. ``axes`` is the rotary position axes the worker
        staged (1, or 3 for prompts placing vision blocks); ``embedded``
        replaces the masked tokens' embeddings with ``input_embeddings``.
        """
        query = SequenceLengths(
            values=b["query_lengths"][:rows],
            offsets=b["query_offsets"][: rows + 1],
            host=tuple(int(value) for value in queries),
        )
        prefix = SequenceLengths(
            values=b["cache_lengths"][:rows],
            offsets=b["prefix_offsets"][: rows + 1],
            host=tuple(int(value) for value in prefixes),
        )
        mode = tuple(bool(value) for value in causal)
        flags = b["causal_values"][:rows] if len(set(mode)) > 1 else None
        entries: dict[int, AttentionInput] = {}
        for table, table_width in enumerate(widths):
            first = None
            host_first = None
            if windowed[table]:
                first = b["start_pages"][table, :rows]
                host_first = tuple(
                    int(value)
                    for value in starts[table * rows : (table + 1) * rows]
                )
            blocks = BlockTable(
                b["block_tables"][table, :rows, : int(table_width)],
                page_tokens[table],
                first,
                host_first,
            )
            if writes:
                entries[table] = PagedInput(
                    query,
                    prefix,
                    blocks,
                    b["write_indices"][table, :tokens],
                    mode,
                    flags,
                )
            else:
                entries[table] = SegmentedInput(
                    query,
                    prefix,
                    blocks,
                    None,
                    query.values[:, None].expand(-1, max(query.host or (1,))),
                    True,
                )
        embeddings = None
        if embedded:
            embeddings = EmbeddingReplacement(
                b["input_embeddings"][:tokens],
                b["embedding_mask"][:tokens].view(torch.bool),
            )
        return TextInput(
            b["input_ids"][:tokens],
            b["positions"][0, :tokens]
            if int(axes) == 1
            else b["positions"][: int(axes), :tokens],
            AttentionBatch(entries, query),
            embeddings,
        )

    def project(hidden, indices, count, out=None):
        """Project rows on the last stage and share them with every stage."""
        if last_stage:
            values = model.compute_logits(hidden, token_indices=indices).values
            if out is not None:
                values = out.copy_(values)
        else:
            values = (
                out
                if out is not None
                else hidden.new_empty((count, width), dtype=b["logits"].dtype)
            )
        pipeline.broadcast(values, src=pipeline.size - 1)
        return values

    def decode_body(inputs, rows):
        """Final-position logits of every row and greedy decisions.

        This is the captured decode step: it writes ``logits[:rows]`` and the
        greedy selection columns for every bucket row.
        """
        hidden = model(inputs)
        indices = (
            b["query_offsets"][1 : rows + 1].to(torch.int64) - 1
        ).clamp_min(0)
        logits = project(hidden, indices, rows, out=b["logits"][:rows])
        maxima, tokens = greedy(logits, vocab)
        valid = torch.isfinite(maxima)
        active = b["slot.predicates"].index_select(
            0, b["request_pool_indices"][:rows]
        )
        finish = b["decode_force_finish"][:rows] & valid & active
        continuation = valid & active & ~finish
        completion = b["completion"]
        completion[0, :rows].copy_(valid)
        completion[1, :rows].copy_(active)
        completion[2, :rows].copy_(tokens)
        completion[3, :rows].zero_()
        b["points"][:rows].fill_(1)
        b["continuation"][:rows].copy_(continuation)
        b["tagged"][:rows].copy_(
            tokens | torch.where(continuation, TOKEN_CONTINUATION_BIT, 0)
        )
        b["transition"][:rows].zero_()

    def gathered(values):
        """Global ``[rows, vocab]`` logits of local candidate rows."""
        return Logits(values, vocab).gather()

    def broadcast(value):
        return None if tensor.size == 1 else tensor.broadcast(value, src=0)

    def write_selection(first, selection: Selection):
        count = int(selection.tokens.numel())
        end = first + count
        completion = b["completion"]
        completion[0, first:end].copy_(selection.valid)
        completion[1, first:end].copy_(selection.active)
        completion[2, first:end].copy_(selection.tokens)
        completion[3, first:end].copy_(selection.accepted)
        b["continuation"][first:end].copy_(selection.continuation)
        b["tagged"][first:end].copy_(selection.tagged)
        b["points"][first:end].copy_(selection.points)
        b["transition"][first:end].copy_(selection.transition)

    def active(tasks, first):
        """Each task's predicate: its slot's continuation flag when gated."""
        count = len(tasks)
        columns = b["tasks"][:, first : first + count]
        flags = b["slot.predicates"].index_select(0, columns[0])
        return torch.where(columns[1] != 0, flags, True)

    def penalty_rows(tasks):
        """Per candidate row, its token counts (``None`` without penalties).

        Draft prefixes accumulate into copies; the committed counts never
        change before commit.
        """
        rows: list[torch.Tensor | None] = []
        counts = b["slot.penalty_counts"]
        for task in tasks:
            if not task.penalties:
                rows.extend([None] * task.rows)
                continue
            base = counts[task.slot]
            rows.append(base)
            for index in range(1, task.rows):
                row = base.clone()
                drafts = torch.tensor(
                    task.drafts[:index], dtype=torch.long, device=device
                )
                row.index_put_(
                    (drafts,),
                    torch.ones_like(drafts, dtype=row.dtype),
                    accumulate=True,
                )
                rows.append(row)
        return rows

    def sample(candidates, descriptor):
        """Run every task group of the step's sampling descriptor."""
        cursor = _Cursor(descriptor)
        task_count, group_count = cursor.take(2)
        groups = [cursor.take(GROUP_FIELDS) for _ in range(group_count)]
        tasks = decode_tasks(cursor, task_count)
        if not cursor.done():
            raise ValueError("sampling descriptor has trailing words")
        logits = gathered(candidates)
        for path, top_k, first, count, words, size, parameters in groups:
            members = tasks[first : first + count]
            rows = sum(task.rows for task in members)
            start = members[0].row
            if all(
                task.row == start + offset
                for task, offset in zip(members, _offsets(members), strict=True)
            ):
                group_logits = logits[start : start + rows]
            else:
                # Tasks of one path may own rows anywhere in the invocation.
                selected = torch.tensor(
                    [
                        task.row + index
                        for task in members
                        for index in range(task.rows)
                    ],
                    dtype=torch.long,
                    device=logits.device,
                )
                group_logits = logits.index_select(0, selected)
            draws = b["draws"][parameters : parameters + rows]
            params = b["params"][parameters : parameters + rows]
            gates = active(members, first)
            if path in (GREEDY, GREEDY_SUPPRESSED):
                selection = select_greedy(
                    members,
                    group_logits,
                    gates,
                    suppress=path == GREEDY_SUPPRESSED,
                    broadcast=broadcast,
                )
            elif path == TOP_K:
                selection = select_top_k(
                    members,
                    group_logits,
                    draws,
                    params,
                    gates,
                    top_k,
                    broadcast=broadcast,
                )
            elif path == CATEGORICAL:
                selection = select_categorical(
                    members,
                    group_logits,
                    draws,
                    params,
                    gates,
                    penalty_rows(members),
                    broadcast=broadcast,
                )
            else:
                raise ValueError(f"unknown sampling path {path}")
            write_selection(first, selection)
            requests = [task.logprobs for task in members if task.logprobs]
            if requests:
                packed = selection.logprobs
                if (
                    packed is None
                    or int(packed.numel()) != size
                    or size != packed_words(requests)
                ):
                    raise ValueError("logprob column disagrees with its layout")
                b["logprobs"][words : words + size].copy_(packed)

    def finish_greedy(descriptor, rows):
        """Apply terminal policy to a decode graph's greedy selections."""
        cursor = _Cursor(descriptor)
        task_count, group_count = cursor.take(2)
        for _ in range(group_count):
            cursor.take(GROUP_FIELDS)
        tasks = decode_tasks(cursor, task_count)
        if len(tasks) != rows:
            raise ValueError("greedy decode tasks do not cover its rows")
        if not any(task.finish_ids for task in tasks):
            return
        completion = b["completion"]
        tokens = completion[2, :rows]
        finish, continuation, tagged, _ = decisions(
            tasks,
            tokens,
            completion[0, :rows] != 0,
            completion[1, :rows] != 0,
        )
        b["continuation"][:rows].copy_(continuation)
        b["tagged"][:rows].copy_(tagged)

    def score_prompts(candidates, descriptor):
        """Score prompt chunks into logprob words and retained tail rows.

        Each record is ``[row, count, token, continued, slot, words, tail,
        top, requested, ids...]``: the chunk's candidate rows and input
        tokens, what predicts its first token (nothing for 0, the slot's
        retained logits for 1, candidate row ``continued - 2`` otherwise:
        the vision block before it), the logprob words it writes and the
        tail row its last logits go to. A record of no rows only retains
        candidate row ``row`` (a prompt ending with a vision block).
        """
        cursor = _Cursor(descriptor)
        while not cursor.done():
            row, count, token, continued, slot, words, tail = cursor.take(7)
            top = cursor.next()
            ids = cursor.take(cursor.next())
            if not count:
                b["prompt_tail"][tail].copy_(
                    gathered(candidates[row : row + 1])[0]
                )
                continue
            logits = gathered(candidates[row : row + count])
            if continued == 1:
                previous = b["slot.prompt_logits"][slot]
            elif continued > 1:
                previous = gathered(candidates[continued - 2 : continued - 1])[
                    0
                ]
            else:
                previous = None
            retained, packed = prompt_logprobs(
                b["input_ids"][token : token + count],
                logits,
                previous,
                LogprobRequest(top, ids),
            )
            b["prompt_tail"][tail].copy_(retained)
            if packed is not None:
                b["logprobs"][words : words + int(packed.numel())].copy_(packed)

    def gather(indexed, rows, live, tokens, widths):
        table = int(max(widths))
        common = {
            "request_unit_tables": b["slot.unit_tables"],
            "request_start_pages": b["slot.start_pages"],
            "table_shapes": b["slot.table_shapes"],
            "block_tables": b["block_tables"],
            "start_pages": b["start_pages"],
            "cache_lengths": b["cache_lengths"],
            "query_lengths": b["query_lengths"],
            "query_offsets": b["query_offsets"],
            "prefix_offsets": b["prefix_offsets"],
            "write_indices": b["write_indices"],
        }
        if indexed:
            gather_request_decode_inputs(
                request_pool_indices=b["request_pool_indices"],
                request_cache_lengths=b["slot.verified_lengths"],
                request_tokens=b["slot.future_input_tokens"][:, 0],
                request_positions=b["slot.logical_lengths"],
                input_ids=b["input_ids"],
                positions=b["positions"][:1],
                rows=live,
                columns=table,
                **common,
            )
        else:
            gather_request_rows(
                columns=b["row_columns"],
                rows=rows,
                tokens=tokens,
                width=table,
                **common,
            )

    def step(
        graph,
        decode: int,
        rows: int,
        live: int,
        tokens: int,
        indexed: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        causal: int,
        writes: int,
        logit_count: int,
        all_last: int,
        cache_only: int,
        sampling: Sequence[int],
        greedy_finish: int,
        prompt: Sequence[int],
        axes: int,
        embedded: int,
        expert_capacity: int,
    ) -> None:
        """Evaluate one invocation; see the module documentation.

        ``rows`` and ``tokens`` describe the invocation including graph or
        decode-bucket padding, ``live`` its rows that produce results.
        ``logit_count`` rows are projected for a non-decode invocation, at
        the ``logit_rows`` token positions or, with ``all_last``, at every
        row's last query token; a decode invocation's candidates are its
        live rows' final-position logits. ``axes`` is the rotary position
        axes the worker staged (3 for prompts placing vision blocks) and
        ``embedded`` replaces the masked tokens' embeddings with the staged
        vision features; such invocations run eagerly. ``expert_capacity``
        is the agreed per-rank token capacity of the expert step an eager
        forward runs as (zero for a model without expert-parallel layers);
        a replayed graph was captured at that capacity.
        """
        rows, live, tokens = int(rows), int(live), int(tokens)
        with torch.inference_mode(), context.activate():
            gather(indexed, rows, live, tokens, widths)
            inputs = views(
                rows,
                tokens,
                queries,
                prefixes,
                starts,
                causal,
                writes,
                widths,
                axes,
                bool(embedded),
            )
            context.bind_attention(inputs.attention, replay=graph is not None)
            hidden = None
            if graph is not None:
                graph.replay()
                if not decode and not cache_only:
                    hidden = b["hidden"][:tokens]
            else:
                with context.expert_step(int(expert_capacity)):
                    if decode:
                        decode_body(inputs, rows)
                    elif cache_only:
                        model.fill_cache(inputs)
                    else:
                        hidden = model(inputs)

            if decode:
                candidates = b["logits"][:live]
            elif cache_only or not logit_count:
                return
            else:
                count = int(logit_count)
                if all_last:
                    indices = (
                        b["query_offsets"][1 : live + 1].to(torch.int64) - 1
                    )
                else:
                    indices = b["logit_rows"][:count]
                out = b["logits"][:count] if count <= capacity else None
                candidates = project(hidden, indices, count, out=out)

            if prompt:
                score_prompts(candidates, prompt)
            if greedy_finish:
                finish_greedy(sampling, live)
            elif sampling:
                sample(candidates, sampling)

    def capture(
        decode: int,
        rows: int,
        tokens: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        causal: int,
        writes: int,
        cache_only: int,
        pool,
        expert_capacity: int,
    ) -> graphs.Graph:
        """Capture the forward of the shapes the last eager step prepared.

        Inputs are the fixed columns, so the graph reads whatever the next
        step gathers into them. A decode graph writes the logits and greedy
        selection columns; a prefill graph writes ``hidden`` or, for a
        cache-only family, only the K/V cache. With ``expert_capacity``, the
        graph is the expert step of that capacity, including the joins of
        expert layers the forward does not reach.
        """
        if computation is None:
            raise ValueError("graph capture requires a CUDA binding")
        rows, tokens = int(rows), int(tokens)
        with torch.inference_mode():
            inputs = views(
                rows, tokens, queries, prefixes, starts, causal, writes, widths
            )
            context.bind_attention(inputs.attention, replay=False)

        def body():
            with context.activate(), context.expert_step(int(expert_capacity)):
                if decode:
                    decode_body(inputs, rows)
                elif cache_only:
                    model.fill_cache(inputs)
                else:
                    b["hidden"][:tokens].copy_(model(inputs))

        segments = None
        if context.weights is not None:
            segments = context.weights.capture
        return graphs.capture(
            body, stream=computation, pool=pool, segments=segments
        )

    def release(graph) -> None:
        graph.release()

    def commit(count: int) -> None:
        """Install ``count`` staged commit rows into the slot state."""
        with scope():
            commit_state(
                b["commit"],
                int(count),
                tokens=b["completion"][2],
                valid=b["completion"][0],
                active=b["completion"][1],
                continuation=b["continuation"].view(torch.uint8),
                points=b["points"],
                future_input_tokens=b["slot.future_input_tokens"],
                predicates=b["slot.predicates"].view(torch.uint8),
                logical_lengths=b["slot.logical_lengths"],
                sampling_positions=b["slot.sampling_positions"],
                verified_lengths=b["slot.verified_lengths"],
                penalty_counts=b["slot.penalty_counts"],
            )

    def close(aborted: bool) -> None:
        """Drop the bound views; an aborted close keeps them until exit."""
        if not aborted:
            b.clear()

    return binding(
        {
            "step": step,
            "capture": capture,
            "release": release,
            "commit": commit,
            "close": close,
        }
    )
