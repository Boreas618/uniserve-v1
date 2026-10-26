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
  each row's selection with its finish decision: the greedy selection, or
  each row sampled from its top-k candidates under its own parameters and
  draw. Every value that changes between forwards is the contents of a
  tensor, so the method is captured into two CUDA graphs per row bucket,
  one per selection, and launched without Python.
- :meth:`CausalLMRunner.commit` installs a forward's accepted selections
  into the slot state; it is captured per row bucket the same way.
- :meth:`CausalLMRunner.prefill` evaluates rows of any length (prompt
  chunks, speculative verification) from host tokens: it gathers their
  attention metadata, runs the model and, unless the forward only writes
  the prefix cache, keeps the hidden states and projects each row's last
  position. It is captured per row and token bucket.
- :meth:`CausalLMRunner.select` selects the tokens of a forward's tasks
  whose selection ``decode`` does not make, each with its own sampler, from
  the logits the forward projected, and :meth:`CausalLMRunner.score` reports
  the logprobs of the tasks that request them. Every input is a tensor
  column (draws, parameters, token sets as bitmasks, the slot-indexed logit
  biases; :mod:`uniserve.sampling.select`), so both are captured per task
  bucket and launched without Python.
  :meth:`CausalLMRunner.score_prompt` scores the prompt tokens of a prefill
  forward, a block of rows per call, from hidden states it projects; it is
  captured per top width and launched once per block.
  :meth:`CausalLMRunner.verify` projects every position of a verification
  forward and selects with speculative acceptance; it runs eagerly.

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

from uniserve import adapters
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
from uniserve.prefix_cache import slots as request_slots
from uniserve.runner.attention import AttentionMetadata
from uniserve.runner.context import execution_context, kernel_records
from uniserve.sampling.greedy import greedy
from uniserve.sampling.logprobs import scatter_logprobs
from uniserve.sampling.select import (
    CATEGORICAL,
    TOKEN_CONTINUATION_BIT,
    Controls,
    Selection,
    select,
)
from uniserve.sampling.state import commit as commit_state
from uniserve.sampling.top_k import sample_top_k

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
        collectives: The process group of each of the model's
            multi-member communicators on the stream, by communicator name,
            or ``None``.
        rows: The most rows one forward evaluates.
        tokens: The most tokens one forward evaluates.
        width: The most pages one row selects from a block table.
        block_size: Tokens per page of each numerical block table.
        windowed: Whether each numerical block table retires pages behind
            an attention window.
        unit_tables: ``[tables, S, units]`` int32, each slot's installed
            units from its installed start page on.
        start_pages: ``[groups, S]`` int32, each slot's installed start
            page.
        table_shapes: ``[tables, 3]`` int32 ``(group, block size, window
            or -1)``.
        verified_lengths: ``[S]`` int32, each slot's verified cache length.
        future_input_tokens: ``[S, 1]`` int64, each slot's next input token.
        logical_lengths: ``[S]`` int64, the logical position of that token.
        predicates: ``[S]`` uint8, whether the slot's last selection
            continues.
        sampling_positions: ``[S]`` int64, the semantic draw position.
        penalty_counts: ``[S, vocab]`` int32 committed token counts.
        prompt_logits: ``[S, vocab]`` last logits of a scored prompt chunk.
        logit_bias: ``[S, vocab]`` float32 logit biases of each slot's
            request, which the caller installs.

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
        block_size: Sequence[int],
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
        logit_bias: torch.Tensor,
        adapters: torch.Tensor,
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
        self._logit_bias = logit_bias
        self._adapters = adapters

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
                block_size,
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
            # Each token's slot and the adapter it computes with
            # (``uniserve.adapters``).
            self._token_slots = torch.zeros(
                tokens, dtype=torch.int64, device=placed
            )
            self._token_adapters = torch.zeros(
                tokens, dtype=torch.int64, device=placed
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
            slot_mapping=self._attention_metadata.slot_mapping[:, :rows],
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
        candidates: int,
        slots: torch.Tensor,
        force_finish: torch.Tensor,
        finish_ids: torch.Tensor,
        logits: torch.Tensor,
        draws: torch.Tensor,
        params: torch.Tensor,
        completion: torch.Tensor,
        continuation: torch.Tensor,
        tagged: torch.Tensor,
        points: torch.Tensor,
        transition: torch.Tensor,
    ) -> None:
        """Evaluate one token of each of ``rows`` request slots.

        Each row's token, position and cache are those of its slot; a row
        of slot 0 is padding. The forward writes every row's final-position
        logits and its selection: the token, whether the selection is
        valid, whether the slot continues, and whether it finishes, which
        it does when forced or when the token is one of the row's finish
        ids. Without ``candidates`` the selection is greedy; with them each
        row samples from its top-k candidates under its parameter row and
        draw, its penalties reading its slot's committed token counts
        (:func:`~uniserve.sampling.top_k.sample_top_k`).

        Args:
            rows: Rows evaluated, padding included.
            widths: Pages each numerical block table holds per row.
            prefixes: Each row's cache length, mirrored on the host.
            starts: First selected pages ``[table][row]``, mirrored on the
                host; read for windowed tables.
            expert_capacity: The agreed per-rank token capacity of the
                expert step this forward runs as; zero for a model without
                expert-parallel layers.
            candidates: Candidates each row samples among, or zero for the
                greedy selection.
            slots: ``[R]`` int64 request slots.
            force_finish: ``[R]`` uint8, rows that finish whatever they
                select.
            finish_ids: ``[R, F]`` int64 finish tokens, padded with -1.
            logits: ``[R, local vocab]`` output, the final-position logits.
            draws: ``[T]`` float32, each row's uniform draw (read with
                ``candidates``).
            params: ``[T, PARAMETER_COLUMNS]`` float32, each row's sampling
                parameters (read with ``candidates``).
            completion: ``[4, R]`` int64 output: valid, active, token,
                accepted drafts (zero).
            continuation: ``[R]`` uint8 output, the row continues.
            tagged: ``[R]`` int64 output, the token with
                ``TOKEN_CONTINUATION_BIT`` where the row continues.
            points: ``[R]`` int64 output, emitted tokens (one).
            transition: ``[R]`` uint8 output, zero: a row with a transition
                output is selected by :meth:`select`.

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
            token_adapters = adapters.gather_rows(
                self._adapters, slots[:rows], self._token_adapters[:rows]
            )

            with (
                self._context.expert_step(int(expert_capacity)),
                adapters.selected(token_adapters),
                request_slots.selected(slots[:rows]),
            ):
                hidden = self._model(inputs)
                indices = (
                    self._attention_metadata.query_offsets[1 : rows + 1].to(
                        torch.int64
                    )
                    - 1
                ).clamp_min(0)
                values = self._project(hidden, indices, rows, logits[:rows])
                if int(candidates):
                    tokens, valid = sample_top_k(
                        self._gathered(values),
                        draws[:rows],
                        params[:rows],
                        int(candidates),
                        self._penalty_counts,
                        slots[:rows],
                    )
                    # Every tensor-parallel rank installs the source rank's
                    # selection.
                    self._broadcast(tokens)
                else:
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
        causal_values: torch.Tensor | None,
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
        causal_values: torch.Tensor | None,
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
        causal_values: torch.Tensor | None,
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
        state and each row's last-position logits; :meth:`select` selects
        from those logits, and :meth:`verify` and :meth:`score_prompt`
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
            causal_values: ``[R]`` int32 causality of each row, which the
                forward reads on the device when its rows differ or its
                graph serves rows of either causality; ``None`` applies
                ``causal``.
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
            metadata = self._attention_metadata
            token_adapters = adapters.gather_tokens(
                self._adapters,
                row_columns,
                metadata.block_tables.shape[1],
                len(metadata.block_size),
                self._token_slots[:tokens],
                self._token_adapters[:tokens],
            )

            # The first section of the row columns is each row's slot.
            with (
                self._context.expert_step(int(expert_capacity)),
                adapters.selected(token_adapters),
                request_slots.selected(row_columns[:rows]),
            ):
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

    def _controls(
        self,
        tasks: int,
        width: int,
        rows: torch.Tensor,
        paths: torch.Tensor,
        draws: torch.Tensor,
        params: torch.Tensor,
        columns: torch.Tensor,
        masks: torch.Tensor,
        mask_rows: torch.Tensor,
        biased: torch.Tensor,
        force: torch.Tensor,
        finish: torch.Tensor,
        transitions: torch.Tensor,
        drafts: torch.Tensor | None,
        terminal: torch.Tensor | None,
    ) -> Controls:
        """The selection controls of ``tasks`` tasks of ``width`` rows.

        ``columns`` holds each task's slot and gate (nonzero when the task
        follows its slot's continuation flag); the other tensors are the
        selection columns :meth:`select` documents, read from their first
        entries.
        """
        count = tasks * width
        slots = columns[0, :tasks]
        predicates = self._predicates.index_select(0, slots)
        active = torch.where(columns[1, :tasks] != 0, predicates, True)
        if drafts is None or terminal is None:
            drafts = rows.new_empty((tasks, 0))
            terminal = torch.zeros(tasks, dtype=torch.int64, device=rows.device)
        else:
            drafts = drafts[: tasks * (width - 1)].view(tasks, width - 1)
            terminal = terminal[:tasks]
        return Controls(
            rows=rows[:count].view(tasks, width),
            paths=paths[:tasks],
            params=params[:count],
            draws=draws[:count],
            slots=slots,
            active=active,
            masks=masks,
            mask_rows=mask_rows[:count],
            biased=biased[:tasks].view(torch.bool),
            force=force[:tasks].view(torch.bool),
            finish=finish[:tasks],
            transitions=transitions[:tasks],
            drafts=drafts,
            terminal=terminal,
        )

    @staticmethod
    def _write_selection(
        selection: Selection,
        completion,
        continuation,
        tagged,
        points,
        transition,
        scored,
    ) -> None:
        end = int(selection.tokens.numel())
        completion[0, :end].copy_(selection.valid)
        completion[1, :end].copy_(selection.active)
        completion[2, :end].copy_(selection.tokens)
        completion[3, :end].copy_(selection.accepted)
        continuation[:end].copy_(selection.continuation)
        tagged[:end].copy_(selection.tagged)
        points[:end].copy_(selection.points)
        transition[:end].copy_(selection.transition)
        scored[:end].copy_(selection.scored)

    def select(
        self,
        tasks: int,
        candidates: int,
        top_k: int,
        categorical: int,
        sorts: int,
        logits: torch.Tensor,
        rows: torch.Tensor,
        paths: torch.Tensor,
        draws: torch.Tensor,
        params: torch.Tensor,
        columns: torch.Tensor,
        masks: torch.Tensor,
        mask_rows: torch.Tensor,
        biased: torch.Tensor,
        force: torch.Tensor,
        finish: torch.Tensor,
        transitions: torch.Tensor,
        completion: torch.Tensor,
        continuation: torch.Tensor,
        tagged: torch.Tensor,
        points: torch.Tensor,
        transition: torch.Tensor,
        scored: torch.Tensor,
    ) -> None:
        """Select a token for each of ``tasks`` tasks of a forward.

        Each task samples from one row of the logits the forward projected,
        with its own sampler (:func:`~uniserve.sampling.select.select`).
        Every value that changes between forwards is the contents of a
        tensor, so the method is captured per task bucket and set of
        samplers. Padding tasks select over row 0 in slot 0 and their
        results are not read.

        Args:
            tasks: Tasks selected, padding included.
            candidates: The top-k sampler's candidate width.
            top_k, categorical: Some task takes the top-k sampler or the
                general one; ``sorts`` when the general one sorts a whole
                vocabulary row for some task.
            logits: ``[R, local vocab]`` logits the forward projected.
            rows: ``[T]`` int64, each task's row of ``logits``.
            paths: ``[T]`` int64, each task's sampler
                (:mod:`uniserve.sampling.select`).
            draws: ``[T]`` float32 uniform draws.
            params: ``[T, PARAMETER_COLUMNS]`` float32 parameters.
            columns: ``[2, R]`` int64, each task's slot and gate.
            masks: ``[M, words]`` int32 token sets masked rows may select.
            mask_rows: ``[T]`` int64, each task's set of ``masks``, or -1.
            biased: ``[R]`` uint8, tasks that add their slot's logit biases.
            force: ``[R]`` uint8, tasks that finish whatever they select.
            finish, transitions: ``[R, words]`` int32 token sets.
            completion: ``[4, R]`` int64 output: valid, active, token,
                accepted drafts.
            continuation: ``[R]`` uint8 output, the task continues.
            tagged: ``[R]`` int64 output, the token with
                ``TOKEN_CONTINUATION_BIT`` where the task continues.
            points: ``[R]`` int64 output, emitted tokens.
            transition: ``[R]`` uint8 output, the task transitions.
            scored: ``[R]`` int64 output, the row of ``logits`` reported
                logprobs score.
        """
        tasks = int(tasks)
        with torch.inference_mode(), self._context.activate():
            controls = self._controls(
                tasks,
                1,
                rows,
                paths,
                draws,
                params,
                columns,
                masks,
                mask_rows,
                biased,
                force,
                finish,
                transitions,
                None,
                None,
            )
            selection = select(
                self._gathered(logits),
                controls,
                candidates=int(candidates),
                top_k=bool(top_k),
                categorical=bool(categorical),
                sorts=bool(sorts),
                counts=self._penalty_counts,
                bias=self._logit_bias,
                broadcast=self._broadcast,
            )
            self._write_selection(
                selection,
                completion,
                continuation,
                tagged,
                points,
                transition.view(torch.bool),
                scored,
            )

    @staticmethod
    def _task_layout(counts: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The compact layout of reporting tasks: one block from word zero.

        ``counts`` holds a row's top count per task, then -1 for each
        padding row (:func:`~uniserve.sampling.logprobs.compact_offsets`).
        """
        kept = counts.clamp_min(0)
        index = torch.arange(int(counts.numel()), device=counts.device)
        return (
            torch.zeros_like(counts),
            index,
            (counts >= 0).sum().expand_as(counts),
            torch.cumsum(kept, 0) - kept,
            kept.sum().expand_as(counts),
            counts,
        )

    def score(
        self,
        rows: int,
        top: int,
        dump: int,
        logits: torch.Tensor,
        tasks: torch.Tensor,
        counts: torch.Tensor,
        scored: torch.Tensor,
        completion: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> None:
        """Report the logprobs of the tasks a :meth:`select` selected.

        Each reporting task's selected token and its top tokens are scored
        under the model's distribution (the unshaped logits) of the row of
        ``logits`` the task's selection read (``scored``), and packed
        compactly into the first words of ``logprobs``, each task with its
        own count of top tokens
        (:func:`~uniserve.sampling.logprobs.scatter_logprobs`). The method
        is captured per bucket of reporting tasks and top width.

        Args:
            rows: Reporting tasks, padding included.
            top: Top tokens scored per task, at least each task's count.
            dump: The word of ``logprobs`` padding scores into.
            logits: ``[R, local vocab]`` logits the forward projected.
            tasks: ``[R]`` int64, the task each row scores.
            counts: ``[R]`` int64, each row's top count, -1 for padding.
            scored: ``[R]`` int64, each task's scored row of ``logits``.
            completion: ``[4, R]`` int64, the selection's outputs.
            logprobs: ``[L]`` int64 output.
        """
        rows = int(rows)
        with torch.inference_mode(), self._context.activate():
            chosen = tasks[:rows]
            scatter_logprobs(
                self._gathered(logits).float(),
                scored.index_select(0, chosen),
                completion[2].index_select(0, chosen),
                self._task_layout(counts[:rows]),
                int(top),
                logprobs,
                int(dump),
            )

    def verify(
        self,
        count: int,
        tasks: int,
        width: int,
        candidates: int,
        sorts: int,
        top: int,
        reported: int,
        dump: int,
        hidden: torch.Tensor,
        logit_rows: torch.Tensor,
        rows: torch.Tensor,
        draws: torch.Tensor,
        params: torch.Tensor,
        columns: torch.Tensor,
        masks: torch.Tensor,
        mask_rows: torch.Tensor,
        biased: torch.Tensor,
        force: torch.Tensor,
        finish: torch.Tensor,
        transitions: torch.Tensor,
        drafts: torch.Tensor,
        terminal: torch.Tensor,
        logprob_tasks: torch.Tensor,
        logprob_counts: torch.Tensor,
        completion: torch.Tensor,
        continuation: torch.Tensor,
        tagged: torch.Tensor,
        points: torch.Tensor,
        transition: torch.Tensor,
        scored: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> None:
        """Verify the draft tokens of a verification forward.

        A verification row evaluates its request's next token followed by
        draft tokens. This projects the ``count`` token positions
        ``logit_rows`` of the forward's ``hidden`` states, one candidate row
        per position, and selects each of ``tasks`` tasks' tokens with the
        general sampler and speculative acceptance: the accepted draft
        prefix and the token after it. Every task has ``width`` candidate
        rows (its drafts plus one, padded by repeating its first row).

        Args:
            rows: ``[T * W]`` int64, each candidate row's projected row.
            drafts: ``[T * (W - 1)]`` int64 draft tokens, -1 past a task's
                own.
            terminal: ``[T]`` int64, the one-based length of a task's draft
                prefix ending at its first finish token, or zero.
            top, reported, dump: Top tokens and tasks reporting logprobs,
                which ``logprob_tasks`` and ``logprob_counts`` name as
                :meth:`score` does; their words are packed into
                ``logprobs``.

        The other arguments are those of :meth:`select`, with draws,
        parameters and mask rows per candidate row.
        """
        tasks, width = int(tasks), int(width)
        with torch.inference_mode(), self._context.activate():
            projected = self._gathered(
                self._project(
                    hidden, logit_rows[: int(count)], int(count), None
                )
            )
            controls = self._controls(
                tasks,
                width,
                rows,
                torch.full_like(rows, CATEGORICAL),
                draws,
                params,
                columns,
                masks,
                mask_rows,
                biased,
                force,
                finish,
                transitions,
                drafts,
                terminal,
            )
            selection = select(
                projected,
                controls,
                candidates=int(candidates),
                top_k=False,
                categorical=True,
                sorts=bool(sorts),
                counts=self._penalty_counts,
                bias=self._logit_bias,
                broadcast=self._broadcast,
            )
            self._write_selection(
                selection,
                completion,
                continuation,
                tagged,
                points,
                transition.view(torch.bool),
                scored,
            )
            reported = int(reported)
            if reported:
                chosen = logprob_tasks[:reported]
                scatter_logprobs(
                    projected.float(),
                    selection.scored.index_select(0, chosen),
                    selection.tokens.index_select(0, chosen),
                    self._task_layout(logprob_counts[:reported]),
                    int(top),
                    logprobs,
                    int(dump),
                )

    def score_prompt(
        self,
        rows: int,
        top: int,
        dump: int,
        hidden: torch.Tensor,
        prompt_rows: torch.Tensor,
        block: torch.Tensor,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        prompt_tail: torch.Tensor,
    ) -> None:
        """Score ``rows`` prompt rows of a prefill forward from ``block`` on.

        A prompt row is one position of a scored prompt chunk: it holds the
        logits that predict one prompt token, or the chunk's last logits,
        which the chunk after it continues from. Its logits are the
        projection of its position of ``hidden``, or the retained logits of
        its request slot (the last logits of the chunk before). Each row
        scores its target token of ``input_ids`` and its chunk's count of top
        tokens into the chunk's compact words of ``logprobs``
        (:func:`~uniserve.sampling.logprobs.scatter_logprobs`) and copies
        its logits to its row of ``prompt_tail``. A row with no target
        scores into the ``dump`` word, and a row that is no chunk's last
        copies into the last row of ``prompt_tail``, which no one reads, so
        every row runs the same operations. The method is captured per top
        width and block size, and a forward launches it once per block.

        Args:
            rows: Rows of one call.
            top: Top tokens scored per row, at least every chunk's count.
            dump: The word of ``logprobs`` the rows without a target score
                into.
            hidden: ``[T, hidden]`` hidden states of the forward.
            prompt_rows: ``[8, Q]`` int64; per row its position in
                ``hidden``, its retained slot (or -1 for its projection), its
                target token's index in ``input_ids``, its chunk's first
                word, its index in the chunk, the chunk's rows and their top
                count (or -1 for a row without a target), and its row of
                ``prompt_tail``.
            block: ``[1]`` int32, the call's first row.
            input_ids: ``[T]`` int64 tokens of the forward.
            logprobs: ``[L]`` int64 output words.
            prompt_tail: ``[R + 1, vocab]`` logits retained per chunk.
        """
        rows = int(rows)
        with torch.inference_mode(), self._context.activate():
            index = block.to(torch.long) + torch.arange(
                rows, dtype=torch.long, device=prompt_rows.device
            )
            positions, slots, targets, bases, offsets, chunk, counts, tails = (
                prompt_rows.index_select(1, index)
            )
            logits = self._gathered(
                self._project(hidden, positions, rows, None)
            )
            logits = torch.where(
                (slots >= 0).unsqueeze(1),
                self._prompt_logits.index_select(0, slots.clamp_min(0)),
                logits,
            )
            prompt_tail.index_copy_(0, tails, logits.to(prompt_tail.dtype))
            # A chunk's rows report one count, so a row's top tokens follow
            # those of the rows before it at that count.
            scatter_logprobs(
                logits.float(),
                torch.arange(rows, device=logits.device),
                input_ids.index_select(0, targets),
                (
                    bases,
                    offsets,
                    chunk,
                    offsets * counts,
                    chunk * counts,
                    counts,
                ),
                int(top),
                logprobs,
                int(dump),
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
