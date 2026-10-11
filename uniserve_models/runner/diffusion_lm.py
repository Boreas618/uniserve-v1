"""The forwards of a diffusion language model: :class:`DiffusionLMRunner`.

A ``DiffusionLM`` evaluates blocks of token ids over a request's cached
prefix. One runner serves one such model on one stream, in two kinds of
forward:

- A readout reads, at answer slots of one or more blocks, the natural
  log-probability of candidate tokens under the log-softmax over the full
  vocabulary of the soft-capped logits. :meth:`DiffusionLMRunner.attend`
  evaluates every token up to the final layer's attention into the caller's
  ``attend_state`` rows; :meth:`DiffusionLMRunner.readout` completes the
  final layer, the output norm, the head and the log-softmax for slot rows
  alone and selects the candidates. A forward with many slots calls
  ``readout`` once per slot chunk over one ``attend``.
- :meth:`DiffusionLMRunner.denoise` denoises each row's block by one step of
  block diffusion: rows at step zero start their block, the head projects
  the blocks in chunks, the sampler of ``uniserve_models.diffusion.canvas``
  steps them, and the stepped state is committed to the rows' request slots
  in the resident banks. Each row's int64 ``[1 + length]`` result holds its
  outcome (``STEP_CONTINUED``, ``STEP_STOPPED`` or ``STEP_SKIPPED``) and its
  argmax tokens, padded after the first end-of-sequence token.

Request state lives in tensors the caller owns and passes once, to the
constructor: the banks with one row per request slot (slot zero is the
padding sentinel). The tensors of one forward are parameters of its method.
Every value that changes between forwards is the contents of a tensor, the
sampler derives its random bits on the device from the seed, block and step
columns, and no method keeps state between calls. The caller captures a
method into one CUDA graph per shape by calling it between
``graphs.begin_capture`` and ``graphs.end_capture`` after one eager call of
the same shape, and launches the graph without Python.

Attention operators that plan a launch on the host read the host mirrors of
the lengths; :meth:`DiffusionLMRunner.plan_attention` builds their plans
before a captured ``attend`` or ``denoise`` is launched. Operators driven by
device metadata need no call between forwards.

Importing this module does not initialize CUDA.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch import nn

from uniserve_models.bindings import capturing
from uniserve_models.diffusion import canvas as sampler
from uniserve_models.model import DiffusionLM, DiffusionLMInput
from uniserve_models.model.logits import Logits
from uniserve_models.nn.attention import (
    AttentionBatch,
    BlockTable,
    SegmentedInput,
    SequenceLengths,
)
from uniserve_models.nn.moe import FusedMoE

__all__ = [
    "STEP_CONTINUED",
    "STEP_SKIPPED",
    "STEP_STOPPED",
    "DiffusionLMRunner",
    "attend_widths",
    "compute_dtype",
    "embedding_dtype",
    "exchanges_readout",
    "generates",
    "hidden_size",
    "sequence_length",
    "stage",
    "vocab_size",
    "workspace_bytes",
]

# The first value of a denoising step's result row: the step left its block
# running, stopped it, or was skipped because an earlier step stopped it.
STEP_CONTINUED, STEP_STOPPED, STEP_SKIPPED = 0, 1, 2


def _model(module) -> DiffusionLM:
    if not isinstance(module, DiffusionLM):
        raise ValueError(
            f"a diffusion LM runner serves a DiffusionLM, not {type(module)}"
        )
    return module


def _table(embedding: nn.Module) -> torch.Tensor:
    """The token embedding table of ``embedding``."""
    # ``nn.Module`` types a parameter read as ``Tensor | Module``.
    return cast(torch.Tensor, embedding.weight)


def _tail_exchanges(
    model: DiffusionLM, *, exchange: bool, attention_ranks: int
) -> bool:
    """Whether a model's ``readout`` exchanges tokens across ranks.

    ``readout`` completes the final layer, the output norm and the head for
    slot rows. An expert-parallel ``FusedMoE`` in the final layer exchanges
    tokens with its expert group, whose ranks read different calls and must
    join each exchange once per expert step at the step's agreed capacity; a
    captured ``readout`` would carry its capture capacity, and one launch
    per slot chunk would add exchanges the other ranks never join.
    Tensor-parallel collectives are joined alike by ranks that read the same
    slots, so they keep ``readout`` local.
    """
    if not exchange:
        return False
    final = tuple(model.backbone.layers.values())[-1]
    return any(
        isinstance(module, FusedMoE)
        and (attention_ranks > 0 or module.expert_group.size > 1)
        for module in final.modules()
    )


def _placed(loaded, path: str) -> DiffusionLM:
    """The diffusion LM at ``path`` of a loaded model."""
    path = str(path)
    root = loaded.module
    return _model(root.get_submodule(path) if path else root)


# Facts of the diffusion LM at ``path`` of a loaded model, which a caller
# sizes its tensors from before a runner exists. Each is the model's own
# read-only method of the same name; the functions exist because a rank
# reaches the library by import path with values it can hold, and a module
# is not one (a callable object crosses as a function).


def sequence_length(loaded, path: str) -> int:
    """Tokens of one generated block."""
    return _placed(loaded, path).sequence_length()


def vocab_size(loaded, path: str) -> int:
    """Real vocabulary entries, the same on every pipeline stage."""
    return _placed(loaded, path).vocab_size()


def hidden_size(loaded, path: str) -> int:
    """The width of the hidden states and token embeddings."""
    return _placed(loaded, path).hidden_size()


def compute_dtype(loaded, path: str) -> str:
    """The name of the element type the layers compute in."""
    return _placed(loaded, path).compute_dtype()


def embedding_dtype(loaded, path: str) -> str:
    """The name of the token embedding's element type."""
    return _placed(loaded, path).embedding_dtype()


def attend_widths(loaded, path: str) -> list[int]:
    """Per-token widths of the state ``attend`` writes on the last stage."""
    return _placed(loaded, path).attend_widths()


def stage(loaded, path: str) -> list[int]:
    """``[rank, size]`` of this rank in the model's pipeline."""
    return _placed(loaded, path).stage()


def generates(loaded, path: str) -> bool:
    """Whether this rank can denoise blocks."""
    return _placed(loaded, path).generates()


def exchanges_readout(
    loaded, path: str, exchange: bool, attention_ranks: int
) -> bool:
    """Whether ``readout`` of the model exchanges tokens across ranks.

    ``exchange`` says routed experts communicate through an expert exchange
    and ``attention_ranks`` counts the attention ranks of a disaggregated
    expert union (zero when every rank holds its experts). A ``readout``
    that exchanges runs once over every slot of a forward; one that does
    not runs once per slot chunk and is captured per slot bucket.
    """
    return _tail_exchanges(
        _placed(loaded, path),
        exchange=bool(exchange),
        attention_ranks=int(attention_ranks),
    )


def workspace_bytes(
    rows: int,
    sequence_length: int,
    vocab_size: int,
    hidden_size: int,
    dtype: str,
    cuda: bool,
) -> int:
    """Bytes of the sampler workspace a generating runner allocates.

    The workspace of ``rows`` blocks stepped at a time
    (``CanvasWorkspace.nbytes``); ``dtype`` names the embedding dtype. The
    caller plans device memory before the runner exists.
    """
    return sampler.CanvasWorkspace.nbytes(
        int(rows),
        int(sequence_length),
        int(vocab_size),
        int(hidden_size),
        dtype=getattr(torch, str(dtype)),
        device_type="cuda" if cuda else "cpu",
    )


def _rows(state: sampler.CanvasState, start: int, stop: int):
    """The sampler state of rows ``start:stop``, as contiguous views."""
    length = state.canvas.shape[1]
    return sampler.CanvasState(
        seed=state.seed[start:stop],
        block=state.block[start:stop],
        step=state.step[start:stop],
        canvas=state.canvas[start:stop],
        history=state.history[start:stop],
        self_conditioning=state.self_conditioning[
            start * length : stop * length
        ],
    )


def _chunk(
    workspace: sampler.CanvasWorkspace, positions: int
) -> sampler.CanvasWorkspace:
    """The leading ``positions`` rows of a sampler workspace, for one chunk.

    The scratch keeps the size the CUDA product expects for the chunk's own
    shape, which keys its reproducible algorithm table, rather than the
    size of the largest chunk the workspace holds.
    """
    hidden = workspace.product.shape[1]
    scratch = 0
    if workspace.product.device.type == "cuda":
        from uniserve_kernels.diffusion import canvas as kernels

        scratch = int(kernels.product_scratch_bytes(positions, hidden))
    return sampler.CanvasWorkspace(
        workspace.weights[:positions],
        workspace.normalizer[:positions],
        workspace.product[:positions],
        workspace.scratch[:scratch],
    )


class DiffusionLMRunner:
    """The forwards of one diffusion language model on one stream.

    Args:
        prepared: The prepared component: its module is the ``DiffusionLM``
            and its execution context holds the operators and the prefix
            cache on the runner's stream.
        page_tokens: Tokens per page of each numerical block table.
        windowed: Whether each numerical block table retires pages behind
            an attention window.
        chunk_rows: Blocks the head projects and the sampler steps at a
            time; zero on a rank that serves readouts only, which passes no
            sampling and no banks.
        max_steps, entropy_bound, t_min, t_max, confidence, stability: The
            block-diffusion sampling the deployment serves
            (``CanvasSampling``).
        bank_tokens: ``[S, C]`` int64, each slot's block.
        bank_history: ``[S, stability, C]`` int64, the argmax blocks the
            stopping rule compares.
        bank_self_conditioning: ``[S, C, H]``, the embedding of each slot's
            latest distribution.
        bank_live: ``[S]`` uint8, whether each slot's block continues.

    A generating runner allocates its sampler workspace here.
    """

    def __init__(
        self,
        prepared,
        page_tokens: Sequence[int],
        windowed: Sequence[int],
        chunk_rows: int,
        max_steps: int = 0,
        entropy_bound: float = 0.0,
        t_min: float = 0.0,
        t_max: float = 0.0,
        confidence: float = 0.0,
        stability: int = 0,
        bank_tokens: torch.Tensor | None = None,
        bank_history: torch.Tensor | None = None,
        bank_self_conditioning: torch.Tensor | None = None,
        bank_live: torch.Tensor | None = None,
    ) -> None:
        context = prepared.context
        if context is None:
            raise ValueError("a runner requires a bound component")
        self._context = context
        self._model = _model(prepared.module)
        self._page_tokens = tuple(int(value) for value in page_tokens)
        self._windowed = tuple(bool(value) for value in windowed)

        backbone = self._model.backbone
        self._pipeline = backbone._pipeline
        self._last = self._pipeline.rank == self._pipeline.size - 1
        self._device = next(backbone.layers.parameters()).device
        self._length = self._model.canvas.length

        # With an exchanging final layer, a readout's expert step spans
        # ``attend`` and ``readout``: ``attend`` leaves the final layer's
        # experts to ``readout``, which starts with every earlier expert
        # layer reached.
        exchange = context.experts
        self._exchanging = _tail_exchanges(
            self._model,
            exchange=exchange is not None,
            attention_ranks=0 if exchange is None else exchange.attention_ranks,
        )
        final = tuple(backbone.layers.values())[-1]
        self._attended = context.expert_layers(
            self._model
        ) - context.expert_layers(final)

        self._chunk_rows = int(chunk_rows)
        # The embedding table the sampler embeds every distribution with.
        self._table: torch.Tensor | None = None
        self._sampling: sampler.CanvasSampling | None = None
        self._workspace: sampler.CanvasWorkspace | None = None
        self._banks: dict[str, torch.Tensor] = {}
        if self._chunk_rows:
            embedding = backbone.embedding
            head = self._model.lm_head
            if (
                embedding is None
                or head is None
                or bank_tokens is None
                or bank_history is None
                or bank_self_conditioning is None
                or bank_live is None
            ):
                raise ValueError(
                    "denoising requires the token embedding table, the "
                    "head and the resident banks"
                )
            self._sampling = sampler.CanvasSampling(
                steps=int(max_steps),
                entropy_bound=float(entropy_bound),
                t_min=float(t_min),
                t_max=float(t_max),
                confidence=float(confidence),
                stability=int(stability),
                eos_ids=tuple(self._model.canvas.eos_token_ids),
                pad_id=self._model.canvas.pad_token_id,
            )
            self._banks = {
                "canvas": bank_tokens,
                "history": bank_history,
                "self_conditioning": bank_self_conditioning,
                "live": bank_live,
            }
            table = self._table = _table(embedding)
            self._vocab = int(cast(Any, head).vocab.size)
            self._hidden = int(backbone.hidden_size)
            # Only the sampler reads and writes its workspace.
            self._workspace = sampler.CanvasWorkspace.empty(
                self._chunk_rows,
                self._length,
                self._vocab,
                self._hidden,
                dtype=table.dtype,
                device=table.device,
            )

    # Facts.

    def plans_attention(self) -> bool:
        """Whether an attention operator plans each launch on the host.

        A captured ``attend`` or ``denoise`` of such a model is launched
        after :meth:`plan_attention` built the plans its graph reads. The
        answer is final once every shape has run eagerly.
        """
        return self._context.plans_attention()

    def kernels(self) -> list[dict[str, Any]]:
        """Kernel records of the sampler's self-conditioning product.

        On CUDA, one ``product`` record per chunk shape the product has run:
        its positions, vocabulary and hidden sizes and the cuBLASLt
        algorithm it uses.
        """
        records: list[dict[str, Any]] = []
        workspace, table = self._workspace, self._table
        if (
            workspace is None
            or table is None
            or workspace.product.device.type != "cuda"
        ):
            return records
        from uniserve_kernels.diffusion import canvas as kernels

        table = table[: self._vocab]
        for rows in range(1, self._chunk_rows + 1):
            chunk = _chunk(workspace, rows * self._length)
            algorithm = kernels.product_algorithm(
                chunk.weights, table, chunk.product, chunk.scratch
            )
            if algorithm is not None:
                records.append(
                    {
                        "path": "canvas_sampler.self_conditioning",
                        "op": "product",
                        "positions": rows * self._length,
                        "vocab": self._vocab,
                        "hidden": self._hidden,
                        "provider": "cublaslt",
                        **algorithm,
                    }
                )
        return records

    # Attention metadata.

    def _attention(
        self,
        rows: int,
        visible: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        query_lengths: torch.Tensor,
        query_offsets: torch.Tensor,
        prefix_lengths: torch.Tensor,
        prefix_offsets: torch.Tensor,
        block_tables: torch.Tensor,
        start_pages: torch.Tensor,
    ) -> AttentionBatch:
        """The attention inputs of ``rows`` sequences over their metadata.

        Every sequence reads its own cached prefix through the block tables
        and its whole self, without causal masks and without writing the
        cache. ``visible`` bounds every sequence's query tokens (a graph
        bucket's length, which sizes the captured launch). ``queries`` and
        ``prefixes`` mirror the device lengths; ``widths`` gives each
        table's page columns and ``starts`` the first selected page of every
        row of every table, ``[table][row]`` flattened, which windowed
        tables mirror on the host.
        """
        shared = SequenceLengths(
            query_lengths[:rows],
            query_offsets[: rows + 1],
            tuple(int(value) for value in queries),
        )
        prefix = SequenceLengths(
            prefix_lengths[:rows],
            prefix_offsets[: rows + 1],
            tuple(int(value) for value in prefixes),
        )
        # Every token sees its whole sequence.
        current = shared.values[:, None].expand(-1, visible)
        entries = {}
        for number, page_tokens in enumerate(self._page_tokens):
            windowed = self._windowed[number]
            first = None
            if windowed:
                first = tuple(
                    int(page)
                    for page in starts[number * rows : (number + 1) * rows]
                )
            blocks = BlockTable(
                block_tables[number, :rows, : int(widths[number])],
                page_tokens,
                start_pages[number, :rows] if windowed else None,
                first,
            )
            entries[number] = SegmentedInput(
                shared, prefix, blocks, None, current, True
            )
        return AttentionBatch(entries, shared)

    def plan_attention(
        self,
        rows: int,
        visible: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        query_lengths: torch.Tensor,
        query_offsets: torch.Tensor,
        prefix_lengths: torch.Tensor,
        prefix_offsets: torch.Tensor,
        block_tables: torch.Tensor,
        start_pages: torch.Tensor,
    ) -> None:
        """Plan the attention launches of the next captured forward.

        Operators that plan on the host (:meth:`plans_attention`) read the
        metadata and its host mirrors here; the graph then reads the plans.
        The arguments are those of :meth:`attend`; a ``denoise`` passes its
        sequence length as ``visible``.
        """
        with torch.inference_mode():
            self._context.bind_attention(
                self._attention(
                    int(rows),
                    int(visible),
                    widths,
                    queries,
                    prefixes,
                    starts,
                    query_lengths,
                    query_offsets,
                    prefix_lengths,
                    prefix_offsets,
                    block_tables,
                    start_pages,
                ),
                replay=True,
            )

    # Forwards.

    def attend(
        self,
        rows: int,
        tokens: int,
        visible: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        expert_capacity: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_lengths: torch.Tensor,
        query_offsets: torch.Tensor,
        prefix_lengths: torch.Tensor,
        prefix_offsets: torch.Tensor,
        block_tables: torch.Tensor,
        start_pages: torch.Tensor,
        *attend_state: torch.Tensor,
    ) -> None:
        """Evaluate the tokens of a readout up to the final layer's attention.

        Args:
            rows: Sequences evaluated, padding included.
            tokens: Their packed tokens.
            visible: The widest sequence's tokens.
            widths: Pages each numerical block table holds per row.
            queries, prefixes: Each sequence's tokens and cached prefix
                tokens, mirrored on the host.
            starts: First selected pages ``[table][row]``, mirrored on the
                host; read for windowed tables.
            expert_capacity: The agreed per-rank token capacity of the
                readout's expert step; zero for a model without
                expert-parallel layers. With an exchanging final layer the
                step continues in :meth:`readout`.
            input_ids, positions: ``[T]`` int64 packed tokens and their
                positions.
            query_lengths, prefix_lengths: ``[R]`` int32.
            query_offsets, prefix_offsets: ``[R + 1]`` int32.
            block_tables: ``[tables, R, W]`` int32 selected pages.
            start_pages: ``[tables, R]`` int32.
            *attend_state: Outputs on the final pipeline stage, one
                ``[T, width]`` tensor per ``attend_widths`` entry: the
                per-token state :meth:`readout` completes. Earlier stages
                forward their activations and pass none.

        The host mirrors serve an eager forward and the capture; a launched
        graph reads the device columns.
        """
        rows, tokens = int(rows), int(tokens)
        batch = self._attention(
            rows,
            int(visible),
            widths,
            queries,
            prefixes,
            starts,
            query_lengths,
            query_offsets,
            prefix_lengths,
            prefix_offsets,
            block_tables,
            start_pages,
        )

        def body() -> None:
            with self._context.expert_step(
                int(expert_capacity), join=not self._exchanging
            ):
                values = self._model.attend(
                    DiffusionLMInput(
                        input_ids[:tokens], positions[:tokens], batch
                    )
                )
                if self._last:
                    for value, target in zip(values, attend_state, strict=True):
                        target[: value.shape[0]].copy_(value)

        self._forward(batch, body)

    def readout(
        self,
        slots: int,
        candidates: int,
        expert_capacity: int,
        slot_rows: torch.Tensor,
        candidate_index: torch.Tensor,
        candidate_logprobs: torch.Tensor,
        *attend_state: torch.Tensor,
    ) -> None:
        """Read candidate log-probabilities at slot rows of an ``attend``.

        Args:
            slots: Slot rows completed.
            candidates: Candidates selected.
            expert_capacity: The capacity :meth:`attend` ran at; read when
                the final layer exchanges tokens, where this call completes
                the readout's expert step.
            slot_rows: ``[S]`` int64, the packed token row of each slot in
                ``attend_state``. Rows past a forward's live slots are
                padding whose results are not read.
            candidate_index: ``[K]`` int64, each candidate's flat index
                ``slot * vocab + token`` into the ``[slots, vocab]``
                log-probabilities.
            candidate_logprobs: ``[K]`` FP32 output, one value per
                candidate. Every pipeline stage receives the final stage's
                values.
            *attend_state: The state :meth:`attend` wrote, on the final
                stage.
        """
        slots, candidates = int(slots), int(candidates)
        capacity = int(expert_capacity) if self._exchanging else 0
        pipeline = self._pipeline

        def body() -> None:
            if self._last:
                rows = slot_rows[:slots]
                with self._context.expert_step(
                    capacity, reached=self._attended
                ):
                    hidden = self._model.finish(attend_state, rows)
                    logits = self._model.compute_logits(
                        hidden,
                        token_indices=torch.arange(
                            slots, dtype=torch.int64, device=rows.device
                        ),
                    )
                    # The softmax normalizes over the whole vocabulary, so
                    # tensor shards gather their columns for the slot rows
                    # alone.
                    logprobs = torch.log_softmax(
                        cast(Logits, logits).gather().float(), dim=-1
                    )
                torch.index_select(
                    logprobs.reshape(-1),
                    0,
                    candidate_index[:candidates],
                    out=candidate_logprobs[:candidates],
                )
            if pipeline.size > 1:
                # Every pipeline stage reports the same readout.
                pipeline.broadcast(
                    candidate_logprobs[:candidates], src=pipeline.size - 1
                )

        self._forward(None, body)

    def denoise(
        self,
        rows: int,
        first: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        expert_capacity: int,
        columns: torch.Tensor,
        positions: torch.Tensor,
        query_lengths: torch.Tensor,
        query_offsets: torch.Tensor,
        prefix_lengths: torch.Tensor,
        prefix_offsets: torch.Tensor,
        block_tables: torch.Tensor,
        start_pages: torch.Tensor,
        tokens: torch.Tensor,
        history: torch.Tensor,
        self_conditioning: torch.Tensor,
        results: torch.Tensor,
        continued: torch.Tensor,
    ) -> None:
        """Denoise ``rows`` gathered blocks by one step.

        Args:
            rows: Blocks stepped, padding included.
            first: Every row is at step zero, which skips the
                self-conditioning signal.
            widths, queries, prefixes, starts: As in :meth:`attend`; every
                sequence holds the model's block length.
            expert_capacity: The agreed capacity of the step's expert step;
                zero without expert-parallel layers.
            columns: ``[4, R]`` int64: per row its request slot, the bits of
                its sampling seed, its block and its step. The sampler
                derives every random draw from the last three. A row of
                slot zero is padding.
            positions: ``[R * C]`` int64 token positions.
            query_lengths, query_offsets, prefix_lengths, prefix_offsets,
            block_tables, start_pages: As in :meth:`attend`.
            tokens: ``[R, C]`` int64, each row's block gathered from the
                banks; stepped in place.
            history: ``[R, stability, C]`` int64, gathered and stepped
                alike.
            self_conditioning: ``[R, C, H]``, gathered and stepped alike.
            results: ``[R, 1 + C]`` int64 output: each row's outcome and
                argmax tokens.
            continued: ``[R]`` uint8 output: the row's block continues.

        Rows whose block continues (or starts) commit their stepped state
        to their slots of the banks; the others report ``STEP_SKIPPED``.
        """
        sampling, workspace = self._sampling, self._workspace
        table = self._table
        if sampling is None or workspace is None or table is None:
            raise ValueError("this runner keeps no generating blocks")
        rows, first = int(rows), bool(first)
        length, vocab, hidden_size = self._length, self._vocab, self._hidden
        model, banks, chunk_rows = self._model, self._banks, self._chunk_rows
        backbone = model.backbone
        batch = self._attention(
            rows,
            length,
            widths,
            queries,
            prefixes,
            starts,
            query_lengths,
            query_offsets,
            prefix_lengths,
            prefix_offsets,
            block_tables,
            start_pages,
        )

        def step() -> None:
            slots, seeds, blocks, steps = columns[:, :rows].unbind(0)
            soft = self_conditioning[:rows].view(rows * length, hidden_size)
            state = sampler.CanvasState(
                seed=seeds,
                block=blocks,
                step=steps,
                canvas=tokens[:rows],
                history=history[:rows],
                self_conditioning=soft,
            )
            # A queued step runs only while its block continues; step zero
            # starts a block and always runs. Decided on the device, so the
            # forward never waits for the host to observe the step before
            # it.
            active = (steps == 0) | (banks["live"].index_select(0, slots) != 0)
            sampler.start_canvas(state, vocab_size=vocab)
            # A block's first pass has a zero self-conditioning signal,
            # whose mixing is exactly the unweighted norm.
            hidden = model(
                DiffusionLMInput(
                    state.canvas.reshape(-1),
                    positions[: rows * length],
                    batch,
                    None if first else soft,
                )
            )
            output = results[:rows]
            for start in range(0, rows, chunk_rows):
                stop = min(rows, start + chunk_rows)
                count = stop - start
                logits = model.compute_logits(
                    hidden,
                    token_indices=torch.arange(
                        start * length,
                        stop * length,
                        dtype=torch.int64,
                        device=hidden.device,
                    ),
                )
                decision = sampler.CanvasDecision.empty(
                    count, length, device=hidden.device
                )
                sampler.denoise_canvas(
                    cast(Logits, logits).gather().view(count, length, vocab),
                    table,
                    backbone.embedding_scale,
                    _rows(state, start, stop),
                    sampling,
                    scores=sampler.CanvasScores.empty(
                        count, length, device=hidden.device
                    ),
                    decision=decision,
                    workspace=_chunk(workspace, count * length),
                )
                output[start:stop, 0] = decision.finished[:, 0]
                output[start:stop, 1:] = decision.tokens

            # The commit skips zero targets, keeping skipped request rows
            # and the shared padding row untouched across concurrent lanes.
            output[:, 0].masked_fill_(~active, STEP_SKIPPED)
            targets = torch.where(active, slots, torch.zeros_like(slots))
            sampler.commit_rows(banks["canvas"], targets, state.canvas)
            if state.history.shape[1]:
                sampler.commit_rows(banks["history"], targets, state.history)
            sampler.commit_rows(
                banks["self_conditioning"], targets, self_conditioning[:rows]
            )
            # Whether each row's block continues after this step: the
            # completion flag of its call and its slot's continuation.
            flags = continued[:rows]
            flags.copy_(output[:, 0] == STEP_CONTINUED)
            sampler.commit_rows(banks["live"], targets, flags)

        def body() -> None:
            with self._context.expert_step(int(expert_capacity)):
                step()

        self._forward(batch, body)

    def _forward(
        self, batch: AttentionBatch | None, body: Callable[[], None]
    ) -> None:
        """Run a forward's numerical work on the runner's stream.

        ``batch`` is the attention the body reads (``None`` without
        attention). An eager forward plans it here; a forward that is being
        captured reads the plans of the eager call that preceded it.
        """
        context = self._context
        with torch.inference_mode(), context.activate():
            if batch is not None and not capturing(self._device):
                context.bind_attention(batch)
            body()
