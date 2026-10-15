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
- :meth:`CausalLMRunner.sample`, :meth:`CausalLMRunner.sample_top_k` and
  :meth:`CausalLMRunner.sample_greedy` select tokens from the logits a
  forward projected, for tasks that do not use the greedy selection of
  ``decode``: one call per group of tasks sharing a sampler.
  :meth:`CausalLMRunner.verify` projects every position of a verification
  forward and selects with speculative acceptance, and
  :meth:`CausalLMRunner.prompt_logprobs` scores prompt tokens. These run
  eagerly. Per-row values (draws, temperatures, cutoffs, penalty
  strengths) are tensor columns; what the caller knows on the host
  arrives as flat integer lists (:mod:`uniserve.sampling.select`).

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

from uniserve.bindings import capturing
from uniserve.models import (
    EmbeddingReplacement,
    Logits,
    TextInput,
    TextSize,
    VocabShard,
)
from uniserve.nn.attention import (
    AttentionBatch,
)
from uniserve.nn.attention.gather import (
    gather_request_decode_inputs,
)
from uniserve.runner.attention import AttentionMetadata
from uniserve.runner.context import execution_context, kernel_records
from uniserve.sampling.greedy import greedy
from uniserve.sampling.logprobs import (
    LogprobRequest,
    prompt_logprobs,
)
from uniserve.sampling.select import (
    TOKEN_CONTINUATION_BIT,
    Selection,
    select_categorical,
    select_greedy,
    select_top_k,
)
from uniserve.sampling.state import commit as commit_state

__all__ = ["CausalLMRunner"]


def _causal_lm(loaded):
    """The causal LM of a loaded model, or ``None`` on a rank without one."""
    from uniserve.loading.rank import capability
    from uniserve.models import CausalLM

    return capability(loaded.module, CausalLM)


def vocab_size(loaded) -> int:
    """Real vocabulary entries of the loaded model's causal LM.

    A caller sizes per-slot vocabulary tensors from it before a runner
    exists. Zero on a rank that holds no causal LM.
    """
    model = _causal_lm(loaded)
    return 0 if model is None else int(model.backbone.vocab_size)


def hidden_size(loaded) -> int:
    """The hidden width of the loaded model's causal LM.

    Zero on a rank that holds no causal LM.
    """
    model = _causal_lm(loaded)
    return 0 if model is None else int(model.backbone.hidden_size)


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


def _requests(
    counts: Sequence[int], ids: Sequence[int]
) -> list[LogprobRequest]:
    """What each requesting task reports.

    ``counts`` holds each task's top count and ``ids`` the row-major
    ``[tasks, width]`` matrix of its explicit token ids, padded with -1.
    """
    width = len(ids) // len(counts) if len(counts) else 0
    return [
        LogprobRequest(
            int(count),
            tuple(
                int(token)
                for token in ids[index * width : (index + 1) * width]
                if int(token) >= 0
            ),
        )
        for index, count in enumerate(counts)
    ]


class CausalLMRunner:
    """The forwards of one causal language model on one stream.

    Args:
        loaded: The loaded model.
        path: The ``CausalLM``'s module path in the model; empty for the
            model itself.
        device: The CUDA ordinal the forwards compute on, -1 on the host.
        stream: The CUDA stream handle every forward enqueues on (0 on
            CPU).
        scratch: The transient work areas of the stream, which every
            execution context serialized on it borrows.
        cache: The prefix cache view object the attention layers read.
        experts: The expert exchange routed experts communicate through,
            or ``None``.
        weights: The distributed expert weights' prefetch object, or
            ``None``.
        collectives: The ``NativeCollectives`` of the model's communicators
            on the stream, or ``None``.
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

    ``context`` is the runner's execution context, through which an idle
    rank joins expert steps.
    """

    def __init__(
        self,
        loaded,
        path: str,
        device: int,
        stream: int,
        scratch,
        cache,
        experts,
        weights,
        collectives,
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
        path = str(path)
        self._model = (
            loaded.module.get_submodule(path) if path else loaded.module
        )
        context = execution_context(
            self._model,
            int(stream),
            loaded.attention,
            scratch,
            device=int(device),
            cache=cache,
            experts=experts,
            weights=weights,
            collectives=collectives,
        )
        # The operators are prepared for the largest forward.
        context.prepare(TextSize(int(tokens), int(rows)))
        self.context = self._context = context
        self._verified_lengths = verified_lengths
        self._future_input_tokens = future_input_tokens
        self._logical_lengths = logical_lengths
        self._predicates = predicates.view(torch.bool)
        self._sampling_positions = sampling_positions
        self._penalty_counts = penalty_counts
        self._prompt_logits = prompt_logits

        # The state tensors live on the device the forwards compute on.
        placed = unit_tables.device
        self._device = placed
        self._stream = (
            torch.cuda.ExternalStream(int(stream), device=placed)
            if placed.type == "cuda"
            else None
        )
        self._pipeline, self._vocab, self._tensor = _vocabulary(self._model)
        self._last_stage = self._pipeline.rank == self._pipeline.size - 1

        # Attention metadata of one forward, written by the gathers and read
        # by the attention operators.
        rows, tokens = int(rows), int(tokens)
        with self._scope():
            self._attention_metadata = AttentionMetadata(
                page_tokens,
                windowed,
                unit_tables,
                start_pages,
                table_shapes,
                rows,
                tokens,
                int(width),
                placed,
            )

            # The tokens and positions a decode gathers from the slot state.
            self._decode_ids = torch.ones(
                rows, dtype=torch.int64, device=placed
            )
            self._decode_positions = torch.zeros(
                (1, rows), dtype=torch.int64, device=placed
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

    def kernels(self) -> list[str]:
        """The kernels the runner's operators selected, as JSON objects."""
        return kernel_records(self._context.kernels())

    def close(self) -> None:
        """Release the execution context after the stream drained."""
        self._context.close()

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
            request_unit_tables=self._attention_metadata.unit_tables,
            request_start_pages=self._attention_metadata.slot_start_pages,
            table_shapes=self._attention_metadata.table_shapes,
            request_cache_lengths=self._verified_lengths,
            request_tokens=self._future_input_tokens[:, 0],
            request_positions=self._logical_lengths,
            input_ids=self._decode_ids[:rows],
            positions=self._decode_positions[:, :rows],
            block_tables=self._attention_metadata.block_tables[:, :rows],
            start_pages=self._attention_metadata.start_pages[:, :rows],
            cache_lengths=self._attention_metadata.cache_lengths[:rows],
            query_lengths=self._attention_metadata.query_lengths[:rows],
            query_offsets=self._attention_metadata.query_offsets[: rows + 1],
            prefix_offsets=self._attention_metadata.prefix_offsets[: rows + 1],
            write_indices=self._attention_metadata.write_indices[:, :rows],
            columns=int(max(widths)),
        )
        return self._attention_metadata.batch(
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
                    self._attention_metadata.query_offsets[1 : rows + 1].to(
                        torch.int64
                    )
                    - 1
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
        self._attention_metadata.gather(row_columns, rows, tokens, widths)
        return self._attention_metadata.batch(
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
        :func:`~uniserve.nn.attention.gather.gather_request_rows`)
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
                    self._attention_metadata.query_offsets[1 : rows + 1].to(
                        torch.int64
                    )
                    - 1
                ).clamp_min(0)
                self._project(states, indices, rows, logits[:rows])

    # Selection.

    def _gathered(self, values: torch.Tensor) -> torch.Tensor:
        """Global ``[rows, vocab]`` logits of local candidate rows."""
        return Logits(values, self._vocab).gather()

    def _broadcast(self, value: torch.Tensor):
        tensor = self._tensor
        return None if tensor.size == 1 else tensor.broadcast(value, src=0)

    def _group(
        self, logits: torch.Tensor, rows: Sequence[int], counts: Sequence[int]
    ) -> torch.Tensor:
        """The global logits of a group's candidate rows, task by task.

        ``rows`` holds each task's first candidate row in ``logits`` and
        ``counts`` its candidate rows.
        """
        first, total = int(rows[0]), sum(int(count) for count in counts)
        expected, contiguous = first, True
        for row, count in zip(rows, counts, strict=True):
            contiguous &= int(row) == expected
            expected += int(count)
        if contiguous:
            return self._gathered(logits[first : first + total])
        # Tasks of one sampler may own rows anywhere in the forward.
        selected = torch.tensor(
            [
                int(row) + index
                for row, count in zip(rows, counts, strict=True)
                for index in range(int(count))
            ],
            dtype=torch.long,
            device=logits.device,
        )
        return self._gathered(logits.index_select(0, selected))

    def _active(self, tasks: torch.Tensor, first: int, count: int):
        """Each task's predicate: its slot's continuation flag when gated."""
        columns = tasks[:, first : first + count]
        flags = self._predicates.index_select(0, columns[0])
        return torch.where(columns[1] != 0, flags, True)

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

    def sample_greedy(
        self,
        first: int,
        rows: Sequence[int],
        suppressed: Sequence[int],
        force_finish: Sequence[int],
        finish_ids: Sequence[int],
        transition_ids: Sequence[int],
        logits: torch.Tensor,
        tasks: torch.Tensor,
        completion: torch.Tensor,
        continuation: torch.Tensor,
        tagged: torch.Tensor,
        points: torch.Tensor,
        transition: torch.Tensor,
    ) -> None:
        """Select the argmax token of each task of a group.

        Args:
            first: The group's first task: its tasks' selections are
                written from this entry of the outputs on.
            rows: Each task's row of ``logits``.
            suppressed: Scatter indices ``task * vocab + token`` of tokens
                masked from selection.
            force_finish: Per task, whether it finishes whatever it
                selects.
            finish_ids: Row-major ``[tasks, width]`` finish tokens, padded
                with -1.
            transition_ids: Row-major ``[tasks, width]`` tokens that select
                a transition, padded with -1; an empty row for a task
                without a transition output.
            logits: ``[R, local vocab]`` logits a forward projected.
            tasks: ``[2, R]`` int64, each task's slot and gate (nonzero
                when the task follows its slot's continuation flag).
            completion, continuation, tagged, points, transition: The
                selection outputs of :meth:`decode`, one entry per task.
        """
        with torch.inference_mode(), self._context.activate():
            selection = select_greedy(
                self._group(logits, rows, (1,) * len(rows)),
                self._active(tasks, int(first), len(rows)),
                suppressed=suppressed,
                force_finish=force_finish,
                finish_ids=finish_ids,
                transition_ids=transition_ids,
                broadcast=self._broadcast,
            )
            self._write_selection(
                int(first),
                selection,
                completion,
                continuation,
                tagged,
                points,
                transition,
            )

    def sample_top_k(
        self,
        first: int,
        rows: Sequence[int],
        top_k: int,
        parameters: int,
        min_p_rows: Sequence[int],
        logprob_tasks: Sequence[int],
        logprob_counts: Sequence[int],
        logprob_ids: Sequence[int],
        logprob_offset: int,
        force_finish: Sequence[int],
        finish_ids: Sequence[int],
        transition_ids: Sequence[int],
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
        """Sample each task of a group from its ``top_k`` largest logits.

        The group's tasks share the width ``top_k`` and take no token
        masks, penalties or biases. ``parameters`` is the group's first row
        of ``draws`` (``[T]`` float32) and ``params`` (``[T, 8]`` float32),
        one row per task. ``min_p_rows`` names the tasks with a
        minimum-probability cutoff. ``logprob_tasks`` names the tasks that
        report logprobs: each its top ``logprob_counts`` entry and its row
        of ``logprob_ids`` (row-major, padded with -1), packed into
        ``logprobs`` from ``logprob_offset`` on. The other arguments are
        those of :meth:`sample_greedy`.
        """
        first, count, parameters = int(first), len(rows), int(parameters)
        with torch.inference_mode(), self._context.activate():
            selection = select_top_k(
                self._group(logits, rows, (1,) * count),
                draws[parameters : parameters + count],
                params[parameters : parameters + count],
                self._active(tasks, first, count),
                int(top_k),
                min_p_rows=min_p_rows,
                logprob_tasks=logprob_tasks,
                logprob_requests=_requests(logprob_counts, logprob_ids),
                force_finish=force_finish,
                finish_ids=finish_ids,
                transition_ids=transition_ids,
                broadcast=self._broadcast,
            )
            self._write_selection(
                first,
                selection,
                completion,
                continuation,
                tagged,
                points,
                transition,
            )
            if selection.logprobs is not None:
                packed = selection.logprobs
                offset = int(logprob_offset)
                logprobs[offset : offset + int(packed.numel())].copy_(packed)

    def _select(
        self,
        candidates: torch.Tensor,
        first: int,
        row_counts: Sequence[int],
        drafts: Sequence[int],
        terminal_prefixes: Sequence[int],
        parameters: int,
        top_ks: Sequence[int],
        top_p_rows: Sequence[int],
        min_p_rows: Sequence[int],
        typical_p_rows: Sequence[int],
        allowed_rows: Sequence[int],
        allowed: Sequence[int],
        suppressed: Sequence[int],
        penalty_rows: Sequence[int],
        penalty_slots: Sequence[int],
        penalty_drafts: Sequence[int],
        bias: Sequence[int],
        bias_values: Sequence[float],
        logprob_tasks: Sequence[int],
        logprob_counts: Sequence[int],
        logprob_ids: Sequence[int],
        logprob_offset: int,
        force_finish: Sequence[int],
        finish_ids: Sequence[int],
        transition_ids: Sequence[int],
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
        """Shape, sample and write one group's global candidate rows."""
        first, parameters = int(first), int(parameters)
        rows = int(candidates.shape[0])

        # Each penalized row counts its slot's committed tokens and the
        # draft tokens before it; the committed counts change only at
        # commit.
        counts = None
        if len(penalty_rows):
            counts = self._penalty_counts.index_select(
                0,
                torch.tensor(
                    list(penalty_slots), dtype=torch.long, device=self._device
                ),
            )
            if len(penalty_drafts):
                tokens = torch.tensor(
                    list(penalty_drafts), dtype=torch.long, device=self._device
                )
                counts.reshape(-1).index_put_(
                    (tokens,),
                    torch.ones_like(tokens, dtype=counts.dtype),
                    accumulate=True,
                )

        selection = select_categorical(
            candidates,
            draws[parameters : parameters + rows],
            params[parameters : parameters + rows],
            self._active(tasks, first, len(row_counts)),
            row_counts=row_counts,
            drafts=drafts,
            terminal_prefixes=terminal_prefixes,
            top_ks=top_ks,
            top_p_rows=top_p_rows,
            min_p_rows=min_p_rows,
            typical_p_rows=typical_p_rows,
            allowed_rows=allowed_rows,
            allowed=allowed,
            suppressed=suppressed,
            penalty_rows=penalty_rows,
            penalty_counts=counts,
            bias=bias,
            bias_values=bias_values,
            logprob_tasks=logprob_tasks,
            logprob_requests=_requests(logprob_counts, logprob_ids),
            force_finish=force_finish,
            finish_ids=finish_ids,
            transition_ids=transition_ids,
            broadcast=self._broadcast,
        )
        self._write_selection(
            first,
            selection,
            completion,
            continuation,
            tagged,
            points,
            transition,
        )
        if selection.logprobs is not None:
            packed = selection.logprobs
            offset = int(logprob_offset)
            logprobs[offset : offset + int(packed.numel())].copy_(packed)

    def sample(
        self,
        first: int,
        rows: Sequence[int],
        parameters: int,
        top_ks: Sequence[int],
        top_p_rows: Sequence[int],
        min_p_rows: Sequence[int],
        typical_p_rows: Sequence[int],
        allowed_rows: Sequence[int],
        allowed: Sequence[int],
        suppressed: Sequence[int],
        penalty_rows: Sequence[int],
        penalty_slots: Sequence[int],
        bias: Sequence[int],
        bias_values: Sequence[float],
        logprob_tasks: Sequence[int],
        logprob_counts: Sequence[int],
        logprob_ids: Sequence[int],
        logprob_offset: int,
        force_finish: Sequence[int],
        finish_ids: Sequence[int],
        transition_ids: Sequence[int],
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
        """Sample each task of a group through the general shaping path.

        Each task's logits row is shaped in the canonical order
        (:func:`~uniserve.sampling.select.shape_logits`) and sampled
        by inverting the categorical CDF with its draw, or by argmax at
        zero temperature.

        Args:
            top_ks: Per task, its top-k width, or zero without one.
            top_p_rows: Tasks with a nucleus cutoff below one and no top-k
                width.
            min_p_rows: Tasks with a minimum-probability cutoff.
            typical_p_rows: Tasks with a typical-probability cutoff below one.
            allowed_rows: Tasks restricted to the tokens ``allowed``
                (scatter indices ``task * vocab + token``) names for them.
            suppressed: Scatter indices of tokens masked from selection.
            penalty_rows: Tasks with repetition, frequency or presence
                penalties, and ``penalty_slots`` the request slot of each,
                whose committed token counts the penalties read.
            bias: Scatter indices of the additive offsets ``bias_values``.

        The other arguments are those of :meth:`sample_top_k`.
        """
        with torch.inference_mode(), self._context.activate():
            counts = (1,) * len(rows)
            self._select(
                self._group(logits, rows, counts),
                first,
                counts,
                (),
                (0,) * len(rows),
                parameters,
                top_ks,
                top_p_rows,
                min_p_rows,
                typical_p_rows,
                allowed_rows,
                allowed,
                suppressed,
                penalty_rows,
                penalty_slots,
                (),
                bias,
                bias_values,
                logprob_tasks,
                logprob_counts,
                logprob_ids,
                logprob_offset,
                force_finish,
                finish_ids,
                transition_ids,
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

    def verify(
        self,
        count: int,
        first: int,
        rows: Sequence[int],
        row_counts: Sequence[int],
        drafts: Sequence[int],
        terminal_prefixes: Sequence[int],
        parameters: int,
        top_ks: Sequence[int],
        top_p_rows: Sequence[int],
        min_p_rows: Sequence[int],
        typical_p_rows: Sequence[int],
        allowed_rows: Sequence[int],
        allowed: Sequence[int],
        suppressed: Sequence[int],
        penalty_rows: Sequence[int],
        penalty_slots: Sequence[int],
        penalty_drafts: Sequence[int],
        bias: Sequence[int],
        bias_values: Sequence[float],
        logprob_tasks: Sequence[int],
        logprob_counts: Sequence[int],
        logprob_ids: Sequence[int],
        logprob_offset: int,
        force_finish: Sequence[int],
        finish_ids: Sequence[int],
        transition_ids: Sequence[int],
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
        acceptance: the accepted draft prefix and the token after it.

        Args:
            rows: Each task's first candidate row.
            row_counts: Each task's candidate rows: its drafts plus one.
            drafts: The tasks' draft tokens, concatenated.
            terminal_prefixes: Per task, the one-based length of its draft
                prefix ending at the first finish token, or zero.
            penalty_drafts: Scatter indices ``row * vocab + token`` into
                the penalized rows' token counts, one per draft token
                before each row.

        The shaping lists index candidate rows where :meth:`sample`'s index
        tasks; the other arguments are those of :meth:`sample`.
        """
        with torch.inference_mode(), self._context.activate():
            candidates = self._project(
                hidden, logit_rows[: int(count)], int(count), None
            )
            self._select(
                self._group(candidates, rows, row_counts),
                first,
                row_counts,
                drafts,
                terminal_prefixes,
                parameters,
                top_ks,
                top_p_rows,
                min_p_rows,
                typical_p_rows,
                allowed_rows,
                allowed,
                suppressed,
                penalty_rows,
                penalty_slots,
                penalty_drafts,
                bias,
                bias_values,
                logprob_tasks,
                logprob_counts,
                logprob_ids,
                logprob_offset,
                force_finish,
                finish_ids,
                transition_ids,
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

    def prompt_logprobs(
        self,
        count: int,
        rows: Sequence[int],
        row_counts: Sequence[int],
        tokens: Sequence[int],
        continued: Sequence[int],
        slots: Sequence[int],
        offsets: Sequence[int],
        tails: Sequence[int],
        tops: Sequence[int],
        ids: Sequence[int],
        hidden: torch.Tensor,
        logit_rows: torch.Tensor,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        prompt_tail: torch.Tensor,
    ) -> None:
        """Score the prompt chunks of a prefill forward.

        This projects the ``count`` token positions ``logit_rows`` of the
        forward's ``hidden`` states into candidate rows and scores each
        chunk into ``logprobs``, retaining the chunk's last logits in
        ``prompt_tail`` for the chunk after it. The lists hold one entry
        per chunk.

        Args:
            rows: The chunk's first candidate row.
            row_counts: Its candidate rows; a chunk of none only retains
                candidate row ``rows`` (a prompt ending with a vision
                block).
            tokens: Its first token in ``input_ids``.
            continued: What predicts its first token: nothing for 0, the
                slot's retained logits for 1, candidate row
                ``continued - 2`` otherwise (the vision block before it).
            slots: Its request slot.
            offsets: Where its logprob words start in ``logprobs``.
            tails: The row of ``prompt_tail`` its last logits go to.
            tops: Its top count.
            ids: Row-major ``[chunks, width]`` explicit token ids it
                reports, padded with -1.
        """
        with torch.inference_mode(), self._context.activate():
            candidates = self._project(
                hidden, logit_rows[: int(count)], int(count), None
            )
            requests = _requests(tops, ids)
            for index, request in enumerate(requests):
                row, scored = int(rows[index]), int(row_counts[index])
                tail = int(tails[index])
                if not scored:
                    prompt_tail[tail].copy_(
                        self._gathered(candidates[row : row + 1])[0]
                    )
                    continue
                chunk = self._gathered(candidates[row : row + scored])
                previous = None
                before = int(continued[index])
                if before == 1:
                    previous = self._prompt_logits[int(slots[index])]
                elif before > 1:
                    previous = self._gathered(
                        candidates[before - 2 : before - 1]
                    )[0]
                token = int(tokens[index])
                retained, packed = prompt_logprobs(
                    input_ids[token : token + scored], chunk, previous, request
                )
                prompt_tail[tail].copy_(retained)
                if packed is not None:
                    offset = int(offsets[index])
                    logprobs[offset : offset + int(packed.numel())].copy_(
                        packed
                    )

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
                (:mod:`uniserve.sampling.state`). The slots are
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
