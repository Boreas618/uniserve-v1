"""The forward of an autoregressive language model: :class:`CausalLMRunner`.

One runner serves one causal language model on one stream. Request state
lives in tensors the caller owns and passes once, to the constructor: the
request slots' block tables and their continuation state (next input token,
logical position, verified cache length, continuation flag, sampling
position, penalty counts, retained prompt logits). Slot 0 is padding
everywhere. The tensors of one forward are parameters of its method.

- :meth:`CausalLMRunner.decode` evaluates one token per row. It gathers
  each row's token, position and attention metadata from the slot state on
  the device, runs the model, projects the final-position logits and writes
  the greedy selection with its finish decision. Every value that changes
  between forwards is the contents of a tensor, so the method is captured
  into one CUDA graph per row bucket and launched without Python.
- :meth:`CausalLMRunner.commit` installs a forward's accepted selections
  into the slot state; it is captured per row bucket the same way.
- :meth:`CausalLMRunner.prefill` evaluates rows of any length (prompt
  chunks, speculative verification) from host tokens: it gathers their
  attention metadata, runs the model and, unless the forward only writes
  the prefix cache, keeps the hidden states and projects each row's last
  position. It is captured per row and token bucket.
- :meth:`CausalLMRunner.sample` selects tokens from the logits a forward
  projected, for rows that do not use the greedy selection of ``decode``.
  :meth:`CausalLMRunner.verify` projects every position of a verification
  forward and selects with speculative acceptance, and
  :meth:`CausalLMRunner.prompt_logprobs` scores prompt tokens. These run
  eagerly.

Attention operators that plan a launch on the host read the host mirrors of
the lengths; :meth:`CausalLMRunner.plan_decode` and
:meth:`CausalLMRunner.plan_prefill` build their plans before a captured
forward is launched. Operators driven by device metadata need no call
between forwards.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from uniserve_models.bindings import capturing
from uniserve_models.model import (
    EmbeddingReplacement,
    Logits,
    TextInput,
    VocabShard,
)
from uniserve_models.nn.attention import (
    AttentionBatch,
    BlockTable,
    PagedInput,
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
    select_categorical,
    select_greedy,
    select_top_k,
)
from uniserve_models.sampling.state import commit as commit_state

__all__ = ["CausalLMRunner"]

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


def _vocabulary(model) -> tuple[Any, VocabShard, Any]:
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


class CausalLMRunner:
    """The forwards of one causal language model on one stream.

    Args:
        prepared: The prepared component: its module is the ``CausalLM``
            and its execution context holds the operators and the prefix
            cache.
        stream: The CUDA stream handle every forward enqueues on (0 on
            CPU).
        rows: The most rows one forward evaluates.
        tokens: The most tokens one forward evaluates.
        width: The most pages one row selects from a block table.
        page_tokens: Tokens per page of each numerical block table.
        windowed: Whether each numerical block table retires pages behind
            an attention window.
        unit_tables: ``[tables, S, units]`` int32, each slot's installed
            units from its installed start page on.
        start_pages: ``[groups, S]`` int32, each slot's installed start
            page.
        table_shapes: ``[tables, 3]`` int32 ``(group, page tokens, window
            or -1)``.
        verified_lengths: ``[S]`` int32, each slot's verified cache length.
        future_input_tokens: ``[S, 1]`` int64, each slot's next input token.
        logical_lengths: ``[S]`` int64, the logical position of that token.
        predicates: ``[S]`` uint8, whether the slot's last selection
            continues.
        sampling_positions: ``[S]`` int64, the semantic draw position.
        penalty_counts: ``[S, vocab]`` int32 committed token counts.
        prompt_logits: ``[S, vocab]`` last logits of a scored prompt chunk.

    Construction shares the vocabulary descriptor over the pipeline group,
    so the ranks of a group construct their runners in the same order.
    """

    def __init__(
        self,
        prepared,
        stream: int,
        rows: int,
        tokens: int,
        width: int,
        page_tokens: Sequence[int],
        windowed: Sequence[int],
        unit_tables: torch.Tensor,
        start_pages: torch.Tensor,
        table_shapes: torch.Tensor,
        verified_lengths: torch.Tensor,
        future_input_tokens: torch.Tensor,
        logical_lengths: torch.Tensor,
        predicates: torch.Tensor,
        sampling_positions: torch.Tensor,
        penalty_counts: torch.Tensor,
        prompt_logits: torch.Tensor,
    ) -> None:
        context = prepared.context
        if context is None:
            raise ValueError("a runner requires a bound component")
        self._context = context
        self._model = prepared.module
        self._page_tokens = tuple(int(value) for value in page_tokens)
        self._windowed = tuple(bool(value) for value in windowed)

        self._unit_tables = unit_tables
        self._slot_start_pages = start_pages
        self._table_shapes = table_shapes
        self._verified_lengths = verified_lengths
        self._future_input_tokens = future_input_tokens
        self._logical_lengths = logical_lengths
        self._predicates = predicates.view(torch.bool)
        self._sampling_positions = sampling_positions
        self._penalty_counts = penalty_counts
        self._prompt_logits = prompt_logits

        device = unit_tables.device
        self._device = device
        self._stream = (
            torch.cuda.ExternalStream(int(stream), device=device)
            if device.type == "cuda"
            else None
        )
        self._pipeline, self._vocab, self._tensor = _vocabulary(self._model)
        self._last_stage = self._pipeline.rank == self._pipeline.size - 1

        # Attention metadata of one forward, written by the gathers and read
        # by the attention operators. Captured graphs address these
        # tensors, so they live as long as the runner. Rows no gather wrote
        # are inert: they select no page and write no cache entry.
        rows, tokens, tables = int(rows), int(tokens), len(self._page_tokens)
        with self._scope():

            def zeros(*shape: int) -> torch.Tensor:
                return torch.zeros(shape, dtype=torch.int32, device=device)

            self._block_tables = zeros(tables, rows, int(width))
            self._start_pages = zeros(tables, rows)
            self._cache_lengths = zeros(rows)
            self._query_lengths = zeros(rows)
            self._query_offsets = zeros(rows + 1)
            self._prefix_offsets = zeros(rows + 1)
            self._write_indices = torch.full(
                (tables, tokens), -1, dtype=torch.int64, device=device
            )

            # The tokens and positions a decode gathers from the slot state.
            self._decode_ids = torch.ones(
                rows, dtype=torch.int64, device=device
            )
            self._decode_positions = torch.zeros(
                (1, rows), dtype=torch.int64, device=device
            )

    # Facts.

    def vocab(self) -> list[int]:
        """``[size, padded size, start, stop]`` of the vocabulary.

        ``size`` counts real tokens, ``padded size`` the vocabulary
        partitioned across tensor-parallel ranks, and ``[start, stop)`` is
        this rank's shard of it: the columns of its logits.
        """
        vocab = self._vocab
        return [
            int(vocab.size),
            int(vocab.padded_size),
            int(vocab.local_slice.start),
            int(vocab.local_slice.stop),
        ]

    def hidden_size(self) -> int:
        """The width of the model's hidden states."""
        return int(self._model.backbone.hidden_size)

    def plans_attention(self) -> bool:
        """Whether an attention operator plans each launch on the host.

        A captured forward of such a model is launched after
        :meth:`plan_decode` built the plans its graph reads.
        """
        return self._context.plans_attention()

    # Attention metadata.

    @contextmanager
    def _scope(self) -> Iterator[None]:
        """Enqueue on the runner's stream outside an execution context."""
        with torch.inference_mode():
            if self._stream is None:
                yield
            else:
                with torch.cuda.stream(self._stream):
                    yield

    def _attention(
        self,
        rows: int,
        tokens: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        causal: Sequence[int],
        causal_values: torch.Tensor | None,
    ) -> AttentionBatch:
        """The attention inputs of ``rows`` rows over the gathered metadata.

        ``queries``, ``prefixes`` and ``starts`` (``[table][row]``, read for
        windowed tables) are the host mirrors of the gathered lengths and
        first selected pages. ``causal`` holds each row's causality; when
        the rows differ, the caller staged them in ``causal_values`` for
        operators that consume the device column (``None`` for a forward
        whose rows are all causal).
        """
        query = SequenceLengths(
            values=self._query_lengths[:rows],
            offsets=self._query_offsets[: rows + 1],
            host=tuple(int(value) for value in queries),
        )
        prefix = SequenceLengths(
            values=self._cache_lengths[:rows],
            offsets=self._prefix_offsets[: rows + 1],
            host=tuple(int(value) for value in prefixes),
        )
        mode = tuple(bool(value) for value in causal)
        flags = (
            None
            if causal_values is None or len(set(mode)) == 1
            else causal_values[:rows]
        )
        entries = {}
        for table, table_width in enumerate(widths):
            first = None
            host_first = None
            if self._windowed[table]:
                first = self._start_pages[table, :rows]
                host_first = tuple(
                    int(value)
                    for value in starts[table * rows : (table + 1) * rows]
                )
            blocks = BlockTable(
                self._block_tables[table, :rows, : int(table_width)],
                self._page_tokens[table],
                first,
                host_first,
            )
            entries[table] = PagedInput(
                query,
                prefix,
                blocks,
                self._write_indices[table, :tokens],
                mode,
                flags,
            )
        return AttentionBatch(entries, query)

    def _decode_attention(
        self,
        rows: int,
        widths: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        slots: torch.Tensor,
    ) -> AttentionBatch:
        """Gather ``rows`` decode rows from the slot state."""
        gather_request_decode_inputs(
            request_pool_indices=slots[:rows],
            request_unit_tables=self._unit_tables,
            request_start_pages=self._slot_start_pages,
            table_shapes=self._table_shapes,
            request_cache_lengths=self._verified_lengths,
            request_tokens=self._future_input_tokens[:, 0],
            request_positions=self._logical_lengths,
            input_ids=self._decode_ids[:rows],
            positions=self._decode_positions[:, :rows],
            block_tables=self._block_tables[:, :rows],
            start_pages=self._start_pages[:, :rows],
            cache_lengths=self._cache_lengths[:rows],
            query_lengths=self._query_lengths[:rows],
            query_offsets=self._query_offsets[: rows + 1],
            prefix_offsets=self._prefix_offsets[: rows + 1],
            write_indices=self._write_indices[:, :rows],
            columns=int(max(widths)),
        )
        return self._attention(
            rows, rows, widths, (1,) * rows, prefixes, starts, (1,) * rows, None
        )

    def plan_decode(
        self,
        rows: int,
        widths: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        slots: torch.Tensor,
    ) -> None:
        """Plan the attention launches of the next captured decode.

        Operators that plan on the host (:meth:`plans_attention`) read the
        gathered metadata and its host mirrors here; the graph then reads
        the plans. The arguments are those of :meth:`decode`.
        """
        with torch.inference_mode(), self._context.activate():
            attention = self._decode_attention(
                int(rows), widths, prefixes, starts, slots
            )
            self._context.bind_attention(attention)

    # Forwards.

    def _project(self, hidden, indices, count, out):
        """Project rows on the last stage and share them with every stage."""
        if self._last_stage:
            values = self._model.compute_logits(
                hidden, token_indices=indices
            ).values
            if out is not None:
                values = out.copy_(values)
        else:
            values = (
                out
                if out is not None
                else hidden.new_empty(
                    (
                        count,
                        self._vocab.local_slice.stop
                        - self._vocab.local_slice.start,
                    ),
                    dtype=self._prompt_logits.dtype,
                )
            )
        self._pipeline.broadcast(values, src=self._pipeline.size - 1)
        return values

    def decode(
        self,
        rows: int,
        widths: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        expert_capacity: int,
        slots: torch.Tensor,
        force_finish: torch.Tensor,
        finish_ids: torch.Tensor,
        logits: torch.Tensor,
        completion: torch.Tensor,
        continuation: torch.Tensor,
        tagged: torch.Tensor,
        points: torch.Tensor,
        transition: torch.Tensor,
    ) -> None:
        """Evaluate one token of each of ``rows`` request slots.

        Each row's token, position and cache are those of its slot; a row
        of slot 0 is padding. The forward writes every row's final-position
        logits and its greedy selection: the token, whether the selection
        is valid, whether the slot continues, and whether it finishes,
        which it does when forced or when the token is one of the row's
        finish ids.

        Args:
            rows: Rows evaluated, padding included.
            widths: Pages each numerical block table holds per row.
            prefixes: Each row's cache length, mirrored on the host.
            starts: First selected pages ``[table][row]``, mirrored on the
                host; read for windowed tables.
            expert_capacity: The agreed per-rank token capacity of the
                expert step this forward runs as; zero for a model without
                expert-parallel layers.
            slots: ``[R]`` int64 request slots.
            force_finish: ``[R]`` uint8, rows that finish whatever they
                select.
            finish_ids: ``[R, F]`` int64 finish tokens, padded with -1.
            logits: ``[R, local vocab]`` output, the final-position logits.
            completion: ``[4, R]`` int64 output: valid, active, token,
                accepted drafts (zero).
            continuation: ``[R]`` uint8 output, the row continues.
            tagged: ``[R]`` int64 output, the token with
                ``TOKEN_CONTINUATION_BIT`` where the row continues.
            points: ``[R]`` int64 output, emitted tokens (one).
            transition: ``[R]`` uint8 output, zero: a row with a transition
                output is selected by :meth:`sample`.

        The host mirrors serve an eager forward and the capture; a launched
        graph reads the device columns.
        """
        rows = int(rows)
        with torch.inference_mode(), self._context.activate():
            attention = self._decode_attention(
                rows, widths, prefixes, starts, slots
            )
            if not capturing(self._device):
                self._context.bind_attention(attention)
            inputs = TextInput(
                self._decode_ids[:rows],
                self._decode_positions[0, :rows],
                attention,
                None,
            )

            with self._context.expert_step(int(expert_capacity)):
                hidden = self._model(inputs)
                indices = (
                    self._query_offsets[1 : rows + 1].to(torch.int64) - 1
                ).clamp_min(0)
                values = self._project(hidden, indices, rows, logits[:rows])
                maxima, tokens = greedy(values, self._vocab)

                valid = torch.isfinite(maxima)
                active = self._predicates.index_select(0, slots[:rows])
                eligible = valid & active
                listed = tokens.unsqueeze(1).eq(finish_ids[:rows]).any(dim=1)
                finish = (
                    force_finish[:rows].view(torch.bool) | listed
                ) & eligible
                continues = eligible & ~finish

                completion[0, :rows].copy_(valid)
                completion[1, :rows].copy_(active)
                completion[2, :rows].copy_(tokens)
                completion[3, :rows].zero_()
                points[:rows].fill_(1)
                continuation[:rows].view(torch.bool).copy_(continues)
                tagged[:rows].copy_(
                    tokens | torch.where(continues, TOKEN_CONTINUATION_BIT, 0)
                )
                transition[:rows].zero_()

    def _prefill_attention(
        self,
        rows: int,
        tokens: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        causal: Sequence[int],
        row_columns: torch.Tensor,
        causal_values: torch.Tensor,
    ) -> AttentionBatch:
        """Gather the attention metadata of ``rows`` rows of host tokens."""
        gather_request_rows(
            columns=row_columns,
            rows=rows,
            tokens=tokens,
            width=int(max(widths)),
            request_unit_tables=self._unit_tables,
            request_start_pages=self._slot_start_pages,
            table_shapes=self._table_shapes,
            block_tables=self._block_tables,
            start_pages=self._start_pages,
            cache_lengths=self._cache_lengths,
            query_lengths=self._query_lengths,
            query_offsets=self._query_offsets,
            prefix_offsets=self._prefix_offsets,
            write_indices=self._write_indices,
        )
        return self._attention(
            rows,
            tokens,
            widths,
            queries,
            prefixes,
            starts,
            causal,
            causal_values,
        )

    def plan_prefill(
        self,
        rows: int,
        tokens: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        causal: Sequence[int],
        row_columns: torch.Tensor,
        causal_values: torch.Tensor,
    ) -> None:
        """Plan the attention launches of the next captured prefill.

        The counterpart of :meth:`plan_decode`; the arguments are those of
        :meth:`prefill`.
        """
        with torch.inference_mode(), self._context.activate():
            attention = self._prefill_attention(
                int(rows),
                int(tokens),
                widths,
                queries,
                prefixes,
                starts,
                causal,
                row_columns,
                causal_values,
            )
            self._context.bind_attention(attention)

    def prefill(
        self,
        rows: int,
        tokens: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        causal: Sequence[int],
        axes: int,
        outputs: int,
        expert_capacity: int,
        row_columns: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        causal_values: torch.Tensor,
        input_embeddings: torch.Tensor | None,
        embedding_mask: torch.Tensor | None,
        hidden: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """Evaluate ``rows`` rows of host tokens over their slots' caches.

        The caller packed the rows' columns in ``row_columns`` (see
        :func:`~uniserve_models.nn.attention.gather.gather_request_rows`)
        and their tokens and positions in ``input_ids`` and ``positions``;
        a padding row has no token or holds the padding tokens, and writes
        no cache entry. The forward writes the rows' keys and values to the
        prefix cache. With ``outputs`` it also writes every token's hidden
        state and each row's last-position logits; :meth:`sample` selects
        from those logits, and :meth:`verify` and :meth:`prompt_logprobs`
        project other positions from the hidden states.

        Args:
            rows: Rows evaluated, padding included.
            tokens: Tokens evaluated, padding included.
            widths: Pages each numerical block table holds per row.
            queries: Each row's token count, mirrored on the host.
            prefixes: Each row's cache length, mirrored on the host.
            starts: First selected pages ``[table][row]``, mirrored on the
                host; read for windowed tables.
            causal: Each row's causality; a vision block attends to all of
                itself.
            axes: Rotary position axes the rows carry: 1, or 3 for prompts
                placing vision blocks.
            outputs: Whether the forward keeps hidden states and logits.
            expert_capacity: The agreed per-rank token capacity of the
                expert step this forward runs as; zero for a model without
                expert-parallel layers.
            row_columns: The packed int64 row columns.
            input_ids: ``[T]`` int64 tokens.
            positions: ``[3, T]`` int64 positions, axis-major.
            causal_values: ``[R]`` int32 causality of rows that differ.
            input_embeddings: ``[T, hidden]`` vision features replacing the
                embeddings of the tokens ``embedding_mask`` (``[T]`` uint8)
                marks, or ``None`` for a prompt without them.
            hidden: ``[T, hidden]`` output, every token's hidden state.
            logits: ``[R, local vocab]`` output, each row's last-position
                logits.

        The host mirrors serve an eager forward and the capture; a launched
        graph reads the device columns.
        """
        rows, tokens = int(rows), int(tokens)
        with torch.inference_mode(), self._context.activate():
            attention = self._prefill_attention(
                rows,
                tokens,
                widths,
                queries,
                prefixes,
                starts,
                causal,
                row_columns,
                causal_values,
            )
            if not capturing(self._device):
                self._context.bind_attention(attention)

            embeddings = None
            if input_embeddings is not None and embedding_mask is not None:
                embeddings = EmbeddingReplacement(
                    input_embeddings[:tokens],
                    embedding_mask[:tokens].view(torch.bool),
                )
            inputs = TextInput(
                input_ids[:tokens],
                positions[0, :tokens]
                if int(axes) == 1
                else positions[: int(axes), :tokens],
                attention,
                embeddings,
            )

            with self._context.expert_step(int(expert_capacity)):
                if not outputs:
                    self._model.fill_cache(inputs)
                    return
                states = hidden[:tokens].copy_(self._model(inputs))
                # A row without tokens repeats the position before it.
                indices = (
                    self._query_offsets[1 : rows + 1].to(torch.int64) - 1
                ).clamp_min(0)
                self._project(states, indices, rows, logits[:rows])

    # Selection.

    def _gathered(self, values: torch.Tensor) -> torch.Tensor:
        """Global ``[rows, vocab]`` logits of local candidate rows."""
        return Logits(values, self._vocab).gather()

    def _broadcast(self, value: torch.Tensor):
        tensor = self._tensor
        return None if tensor.size == 1 else tensor.broadcast(value, src=0)

    def _penalty_rows(self, tasks: Sequence[Task]):
        """Per candidate row, its token counts (``None`` without penalties).

        Draft prefixes accumulate into copies; the committed counts never
        change before commit.
        """
        rows: list[torch.Tensor | None] = []
        counts = self._penalty_counts
        for task in tasks:
            if not task.penalties:
                rows.extend([None] * task.rows)
                continue
            base = counts[task.slot]
            rows.append(base)
            for index in range(1, task.rows):
                row = base.clone()
                drafts = torch.tensor(
                    task.drafts[:index], dtype=torch.long, device=self._device
                )
                row.index_put_(
                    (drafts,),
                    torch.ones_like(drafts, dtype=row.dtype),
                    accumulate=True,
                )
                rows.append(row)
        return rows

    def sample(
        self,
        rows: int,
        words: Sequence[int],
        logits: torch.Tensor,
        draws: torch.Tensor,
        params: torch.Tensor,
        tasks: torch.Tensor,
        completion: torch.Tensor,
        continuation: torch.Tensor,
        tagged: torch.Tensor,
        points: torch.Tensor,
        transition: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> None:
        """Select the tokens of the tasks sampling ``logits[:rows]``.

        Args:
            rows: Candidate rows a forward wrote to ``logits``: the rows of
                a decode, or of a prefill, whose logits are their last
                positions.
            words: The sampling descriptor: ``[task count, group count]``,
                ``GROUP_FIELDS`` words per group, then one record per task
                (:func:`decode_tasks`).
            logits: ``[R, local vocab]`` candidate logits.
            draws: ``[T]`` float32, one draw per candidate row.
            params: ``[T, 8]`` float32 parameters per candidate row.
            tasks: ``[2, R]`` int64, each task's slot and gate (nonzero
                when the task follows its slot's continuation flag).
            completion, continuation, tagged, points, transition: The
                selection outputs of :meth:`decode`, one entry per task.
            logprobs: ``[L]`` int64 output, packed logprob words.

        FIXME: sampling parameters cross as the encoded ``words``; they
        belong in parameter tensors, with one method per request option.
        """
        with torch.inference_mode(), self._context.activate():
            self._sample(
                logits[: int(rows)],
                words,
                draws,
                params,
                tasks,
                completion,
                continuation,
                tagged,
                points,
                transition,
                logprobs,
            )

    def _candidates(
        self, count: int, hidden: torch.Tensor, logit_rows: torch.Tensor
    ) -> torch.Tensor:
        """The logits of ``count`` token positions of a forward."""
        return self._project(hidden, logit_rows[:count], count, None)

    def verify(
        self,
        count: int,
        words: Sequence[int],
        hidden: torch.Tensor,
        logit_rows: torch.Tensor,
        draws: torch.Tensor,
        params: torch.Tensor,
        tasks: torch.Tensor,
        completion: torch.Tensor,
        continuation: torch.Tensor,
        tagged: torch.Tensor,
        points: torch.Tensor,
        transition: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> None:
        """Verify the draft tokens of a verification forward.

        A verification row evaluates its request's next token followed by
        draft tokens. This projects the ``count`` token positions
        ``logit_rows`` of the forward's ``hidden`` states, one candidate row
        per position, and selects each task's tokens with speculative
        acceptance: the accepted draft prefix and the token after it. The
        remaining arguments are those of :meth:`sample`, whose tasks index
        these candidate rows.
        """
        with torch.inference_mode(), self._context.activate():
            self._sample(
                self._candidates(int(count), hidden, logit_rows),
                words,
                draws,
                params,
                tasks,
                completion,
                continuation,
                tagged,
                points,
                transition,
                logprobs,
            )

    def _sample(
        self,
        candidates,
        words,
        draws,
        params,
        tasks,
        completion,
        continuation,
        tagged,
        points,
        transition,
        logprobs,
    ) -> None:
        """Run every task group of a sampling descriptor."""
        cursor = _Cursor(words)
        task_count, group_count = cursor.take(2)
        groups = [cursor.take(GROUP_FIELDS) for _ in range(group_count)]
        records = decode_tasks(cursor, task_count)
        if not cursor.done():
            raise ValueError("sampling descriptor has trailing words")
        logits = self._gathered(candidates)
        for path, top_k, first, count, offset, size, parameters in groups:
            members = records[first : first + count]
            rows = sum(task.rows for task in members)
            start = members[0].row
            if all(
                task.row == start + offset_
                for task, offset_ in zip(
                    members, _offsets(members), strict=True
                )
            ):
                group_logits = logits[start : start + rows]
            else:
                # Tasks of one path may own rows anywhere in the forward.
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
            group_draws = draws[parameters : parameters + rows]
            group_params = params[parameters : parameters + rows]

            # Each task's predicate: its slot's continuation flag when
            # gated.
            columns = tasks[:, first : first + count]
            flags = self._predicates.index_select(0, columns[0])
            gates = torch.where(columns[1] != 0, flags, True)

            if path in (GREEDY, GREEDY_SUPPRESSED):
                selection = select_greedy(
                    members,
                    group_logits,
                    gates,
                    suppress=path == GREEDY_SUPPRESSED,
                    broadcast=self._broadcast,
                )
            elif path == TOP_K:
                selection = select_top_k(
                    members,
                    group_logits,
                    group_draws,
                    group_params,
                    gates,
                    top_k,
                    broadcast=self._broadcast,
                )
            elif path == CATEGORICAL:
                selection = select_categorical(
                    members,
                    group_logits,
                    group_draws,
                    group_params,
                    gates,
                    self._penalty_rows(members),
                    broadcast=self._broadcast,
                )
            else:
                raise ValueError(f"unknown sampling path {path}")
            self._write_selection(
                first,
                selection,
                completion,
                continuation,
                tagged,
                points,
                transition,
            )

            requests = [task.logprobs for task in members if task.logprobs]
            if requests:
                packed = selection.logprobs
                if (
                    packed is None
                    or int(packed.numel()) != size
                    or size != packed_words(requests)
                ):
                    raise ValueError("logprob column disagrees with its layout")
                logprobs[offset : offset + size].copy_(packed)

    @staticmethod
    def _write_selection(
        first: int,
        selection: Selection,
        completion,
        continuation,
        tagged,
        points,
        transition,
    ) -> None:
        end = first + int(selection.tokens.numel())
        completion[0, first:end].copy_(selection.valid)
        completion[1, first:end].copy_(selection.active)
        completion[2, first:end].copy_(selection.tokens)
        completion[3, first:end].copy_(selection.accepted)
        continuation[first:end].copy_(selection.continuation)
        tagged[first:end].copy_(selection.tagged)
        points[first:end].copy_(selection.points)
        transition[first:end].copy_(selection.transition)

    def prompt_logprobs(
        self,
        count: int,
        prompt: Sequence[int],
        hidden: torch.Tensor,
        logit_rows: torch.Tensor,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        prompt_tail: torch.Tensor,
    ) -> None:
        """Score the prompt tokens of a prefill forward.

        This projects the ``count`` token positions ``logit_rows`` of the
        forward's ``hidden`` states into candidate rows and scores each
        prompt chunk ``prompt`` describes into ``logprobs``, retaining the
        chunk's last logits in ``prompt_tail`` for the chunk after it.

        Each record of ``prompt`` is ``[row, count, token, continued, slot,
        words, tail, top, requested, ids...]``: the chunk's candidate rows
        and input tokens, what predicts its first token (nothing for 0, the
        slot's retained logits for 1, candidate row ``continued - 2``
        otherwise: the vision block before it), the logprob words it writes
        and the tail row its last logits go to. A record of no rows only
        retains candidate row ``row`` (a prompt ending with a vision block).
        """
        with torch.inference_mode(), self._context.activate():
            candidates = self._candidates(int(count), hidden, logit_rows)
            cursor = _Cursor(prompt)
            while not cursor.done():
                row, rows, token, continued, slot, words, tail = cursor.take(7)
                top = cursor.next()
                ids = cursor.take(cursor.next())
                if not rows:
                    prompt_tail[tail].copy_(
                        self._gathered(candidates[row : row + 1])[0]
                    )
                    continue
                scored = self._gathered(candidates[row : row + rows])
                if continued == 1:
                    previous = self._prompt_logits[slot]
                elif continued > 1:
                    previous = self._gathered(
                        candidates[continued - 2 : continued - 1]
                    )[0]
                else:
                    previous = None
                retained, packed = prompt_logprobs(
                    input_ids[token : token + rows],
                    scored,
                    previous,
                    LogprobRequest(top, ids),
                )
                prompt_tail[tail].copy_(retained)
                if packed is not None:
                    logprobs[words : words + int(packed.numel())].copy_(packed)

    # Slot state.

    def commit(
        self,
        rows: int,
        commit_rows: torch.Tensor,
        completion: torch.Tensor,
        continuation: torch.Tensor,
        points: torch.Tensor,
    ) -> None:
        """Install a forward's selections into the slot state.

        Args:
            rows: Commit rows applied.
            commit_rows: ``[8, R]`` int64, per row its slot, mode, task (-1
                without a sample), host cache length, host logical
                position, host sampling position, verification base visible
                length and whether penalties count the token
                (:mod:`uniserve_models.sampling.state`). The slots are
                unique. A row ``[0, CACHE, -1, 0, ...]`` is padding: it
                stores the padding slot's zero cache length again.
            completion, continuation, points: The forward's selection
                outputs, indexed by each row's task.

        The row count fixes the method's shape; every other value is the
        contents of a tensor, so the method is captured per row bucket.
        """
        with self._scope():
            commit_state(
                commit_rows,
                int(rows),
                tokens=completion[2],
                valid=completion[0],
                active=completion[1],
                continuation=continuation,
                points=points,
                future_input_tokens=self._future_input_tokens,
                predicates=self._predicates.view(torch.uint8),
                logical_lengths=self._logical_lengths,
                sampling_positions=self._sampling_positions,
                verified_lengths=self._verified_lengths,
                penalty_counts=self._penalty_counts,
            )
