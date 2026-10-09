"""Token canvases: System One decision readouts and block-diffusion steps.

A ``TokenDenoiser`` evaluates canvases of token ids over a request's cached
prefix. The worker uses it in two forms:

- A readout reads, at answer slots of one or more canvas rows, the natural
  log-probability of each candidate token under the log-softmax over the
  full vocabulary of the soft-capped logits. The pass up to the final
  layer's attention (``readout_attend``) evaluates every canvas token into
  the bound ``attend_state`` rows; the rest of the final layer, the output
  norm, the head and the log-softmax evaluate the slot rows alone, either
  in a tail graph per slot bucket (``readout_tail`` then
  ``readout_candidates``) or eagerly over every slot (``readout_finish``).
- A canvas step denoises each row's resident canvas by one step of block
  diffusion (``step``): rows at step zero start their canvas, the head
  projects the full canvases in chunks of ``step_rows`` rows, the sampler of
  ``uniserve_models.diffusion.canvas`` steps them, and the stepped state is
  committed to the rows' request slots in the resident banks. Each row's
  int64 ``[1 + canvas]`` result holds its outcome (``STEP_CONTINUED``,
  ``STEP_STOPPED`` or ``STEP_SKIPPED``) and its argmax canvas, padded after
  its first end-of-sequence token.

The Rust worker owns every buffer: it writes the canvas tokens, positions,
attention metadata, slot and candidate indices and step columns, gathers
the resident rows of a step's slots into the step views, pads graph
buckets, reads the results back and decides which graph a call replays.
``canvas.bind`` builds the step functions over those buffers once; every
per-call argument is an integer or a short host list. The functions keep
no request identity, slot ownership or progress.

Importing this module does not initialize CUDA.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from uniserve_models.diffusion import canvas as sampler
from uniserve_models.model import CanvasInput, TokenDenoiser
from uniserve_models.model.logits import Logits
from uniserve_models.nn.attention import (
    AttentionBatch,
    BlockTable,
    SegmentedInput,
    SequenceLengths,
)
from uniserve_models.nn.moe import FusedMoE

from . import graphs

__all__ = [
    "STEP_CONTINUED",
    "STEP_SKIPPED",
    "STEP_STOPPED",
    "bind",
    "describe",
    "functions",
    "scratch_bytes",
]

# The first value of a canvas step's result row: the step left its block
# running, stopped it, or was skipped because an earlier step stopped it.
STEP_CONTINUED, STEP_STOPPED, STEP_SKIPPED = 0, 1, 2


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "canvas.describe": describe,
        "canvas.scratch_bytes": scratch_bytes,
        "canvas.bind": bind,
    }


def _denoiser(module) -> TokenDenoiser:
    if not isinstance(module, TokenDenoiser):
        raise ValueError(
            f"canvas calls require a token denoiser, not {type(module)}"
        )
    return module


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _table(embedding: nn.Module) -> torch.Tensor:
    """The token embedding table of ``embedding``."""
    # ``nn.Module`` types a parameter read as ``Tensor | Module``.
    return cast(torch.Tensor, embedding.weight)


def tail_exchanges(
    denoiser: TokenDenoiser, *, exchange: bool, attention_ranks: int
) -> bool:
    """Whether a denoiser's readout tail exchanges tokens across ranks.

    The tail completes the final layer, the output norm and the head for a
    readout's slot rows. An expert-parallel ``FusedMoE`` in the final layer
    exchanges tokens with its expert group, whose ranks read different calls
    and must join each exchange once per expert step at the step's agreed
    capacity; a captured tail would carry its capture capacity, and chunked
    replays would add exchanges the other ranks never join. Tensor-parallel
    collectives in the tail are joined alike by ranks that read the same
    slots, so they keep the tail local.

    Args:
        denoiser: The token denoiser.
        exchange: Routed experts communicate through an expert exchange.
        attention_ranks: Attention ranks of a disaggregated expert union;
            zero when every rank holds its experts.
    """
    if not exchange:
        return False
    final = tuple(denoiser.backbone.layers.values())[-1]
    return any(
        isinstance(module, FusedMoE)
        and (attention_ranks > 0 or module.expert_group.size > 1)
        for module in final.modules()
    )


def describe(loaded, request: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the token denoiser at ``request["path"]`` of a loaded model.

    Request: ``path``, ``exchange`` (routed experts use an expert exchange)
    and ``attention_ranks`` (attention ranks of a disaggregated union).

    Returns:
        ``canvas_length``, ``vocab_size`` (unpadded), ``hidden_size``,
        ``dtype`` (the self-conditioning embedding dtype; the compute dtype
        on a stage without the embedding),
        ``eos_token_ids``, ``pad_token_id``, ``attend_widths`` (per-token
        widths of the final stage's ``attend`` state, empty on other
        stages), ``attend_dtype``, ``generates`` (this rank steps resident
        canvases: one pipeline stage holding the unsharded embedding table
        the sampler embeds every distribution with), ``local_tail`` (the
        readout tail exchanges no tokens across ranks, so it may replay in
        chunks), ``pipeline_rank`` and ``pipeline_size``.
    """
    path = str(request["path"])
    root = loaded.module
    denoiser = _denoiser(root.get_submodule(path) if path else root)
    backbone = denoiser.backbone
    pipeline = backbone._pipeline
    last = pipeline.rank == pipeline.size - 1
    embedding = backbone.embedding
    head = denoiser.lm_head
    stream_dtype = next(backbone.layers.parameters()).dtype
    canvas = denoiser.canvas
    return {
        "canvas_length": canvas.length,
        # Every stage validates candidate ids against the same vocabulary.
        "vocab_size": head.vocab.size
        if last and head is not None
        else backbone.vocab_size,
        "hidden_size": backbone.hidden_size,
        "dtype": _dtype_name(
            stream_dtype if embedding is None else _table(embedding).dtype
        ),
        "eos_token_ids": list(canvas.eos_token_ids),
        "pad_token_id": canvas.pad_token_id,
        "attend_widths": list(backbone.attend_widths) if last else [],
        "attend_dtype": _dtype_name(stream_dtype),
        "generates": pipeline.size == 1
        and embedding is not None
        and embedding.group.size == 1,
        "local_tail": not tail_exchanges(
            denoiser,
            exchange=bool(request["exchange"]),
            attention_ranks=int(request["attention_ranks"]),
        ),
        "pipeline_rank": pipeline.rank,
        "pipeline_size": pipeline.size,
    }


def scratch_bytes(positions: int, hidden_size: int, device: int) -> int:
    """Bytes of a step chunk's self-conditioning product scratch.

    ``positions`` canvas positions of ``hidden_size`` embeddings on CUDA
    ordinal ``device`` (``-1`` for the host, which needs none). The kernels
    size the scratch and key their algorithm table by it.
    """
    if int(device) < 0:
        return 0
    from uniserve_kernels.diffusion import canvas as kernels

    return int(kernels.product_scratch_bytes(int(positions), int(hidden_size)))


@dataclass(frozen=True, slots=True)
class _Table:
    """One numerical block table: its page tokens and history window."""

    page_tokens: int
    windowed: bool


class _Attention:
    """Canvas attention views over the bound metadata columns.

    Every canvas sequence reads its own cached prefix through the block
    tables and its whole canvas, without causal masks and without writing
    the cache. The worker writes the columns; these views only borrow them.
    """

    def __init__(
        self, tensors: Mapping[str, torch.Tensor], tables: Sequence[_Table]
    ) -> None:
        self.query_lengths = tensors["attention.query_lengths"]
        self.query_offsets = tensors["attention.query_offsets"]
        self.prefix_lengths = tensors["attention.prefix_lengths"]
        self.prefix_offsets = tensors["attention.prefix_offsets"]
        self.block_tables = tensors["attention.block_tables"]
        self.start_pages = tensors["attention.start_pages"]
        self.tables = tuple(tables)

    def batch(
        self,
        rows: int,
        visible: int,
        queries: Sequence[int],
        prefixes: Sequence[int],
        widths: Sequence[int],
        starts: Sequence[int],
    ) -> AttentionBatch:
        """The attention of ``rows`` canvas sequences.

        ``visible`` bounds every sequence's query tokens (a graph bucket's
        canvas length, which sizes the captured launch). ``queries`` and
        ``prefixes`` mirror the device lengths; ``widths`` gives each
        table's page columns and ``starts`` the first selected page of every
        row of every table, ``[table][row]`` flattened, which windowed
        tables mirror on the host.
        """
        queries = tuple(int(value) for value in queries)
        prefixes = tuple(int(value) for value in prefixes)
        shared = SequenceLengths(
            self.query_lengths[:rows], self.query_offsets[: rows + 1], queries
        )
        prefix = SequenceLengths(
            self.prefix_lengths[:rows],
            self.prefix_offsets[: rows + 1],
            prefixes,
        )
        # Every canvas token sees its whole canvas.
        current = shared.values[:, None].expand(-1, visible)
        entries = {}
        for number, table in enumerate(self.tables):
            first = None
            if table.windowed:
                first = tuple(
                    int(page)
                    for page in starts[number * rows : (number + 1) * rows]
                )
            blocks = BlockTable(
                self.block_tables[number, :rows, : int(widths[number])],
                table.page_tokens,
                self.start_pages[number, :rows] if table.windowed else None,
                first,
            )
            entries[number] = SegmentedInput(
                shared, prefix, blocks, None, current, True
            )
        return AttentionBatch(entries, shared)


@dataclass(frozen=True, slots=True)
class _Tail:
    """The ``[slots, vocab]`` log-probabilities of an eager readout tail."""

    output: torch.Tensor


class _Graph:
    """A captured canvas function and the attention its replays rebind.

    ``rebind(queries, prefixes, starts)`` plans the captured attention for a
    replay's live host lengths; graphs without attention have none.
    """

    def __init__(
        self,
        graph: graphs.Graph,
        rebind: Callable[[Sequence[int], Sequence[int], Sequence], None] | None,
    ) -> None:
        self.graph = graph
        self.rebind = rebind

    @property
    def output(self) -> Any:
        return self.graph.output


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
    workspace: sampler.CanvasWorkspace, positions: int, device: int
) -> sampler.CanvasWorkspace:
    """The leading ``positions`` rows of a step workspace, for one chunk.

    The scratch keeps the size the CUDA product expects for the chunk's own
    shape, which keys its reproducible algorithm table, rather than the
    size of the largest chunk the workspace holds.
    """
    hidden = workspace.product.shape[1]
    return sampler.CanvasWorkspace(
        workspace.weights[:positions],
        workspace.normalizer[:positions],
        workspace.product[:positions],
        workspace.scratch[: scratch_bytes(positions, hidden, device)],
    )


def bind(
    prepared, buffers: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the canvas functions of one bound component over its buffers.

    ``prepared`` is the component binding (``component.bind``) whose module
    is the token denoiser; every function enqueues on its context's stream.

    ``buffers`` (worker-owned, imported once):

    - ``input_ids``, ``positions``: int64 ``[T]`` packed canvas tokens and
      their positions.
    - ``attention.query_lengths``, ``attention.prefix_lengths``: int32
      ``[R]``; ``attention.query_offsets``, ``attention.prefix_offsets``:
      int32 ``[R + 1]``; ``attention.block_tables``: int32 ``[tables, R,
      W]``; ``attention.start_pages``: int32 ``[tables, R]``.
    - Readouts: ``slot_tokens`` int64 ``[T]`` packed slot rows,
      ``tail_rows`` int64 ``[S]`` a tail chunk's slot rows,
      ``candidate_index`` int64 ``[K]`` flat indices into the tail's
      ``[slots, vocab]`` log-probabilities, ``candidate_logprobs`` FP32
      ``[K]`` the readout result; on the final stage ``attend_state.{i}``
      ``[T, attend_widths[i]]``.
    - Steps (generating ranks): ``step.columns`` int64 ``[4, Rs]`` (slot,
      seed bits, block, step per row), ``step.canvas`` int64 ``[Rs, C]``,
      ``step.history`` int64 ``[Rs, stability, C]``,
      ``step.self_conditioning`` ``[Rs, C, H]``, ``step.results`` int64
      ``[Rs, 1 + C]``, ``step.continued`` uint8 ``[Rs]`` (the block
      continues after the step), ``workspace.{weights, normalizer, product,
      scratch}`` of ``step_rows`` canvases (``CanvasWorkspace``) and the
      resident banks ``bank.{canvas, history, self_conditioning, live}``
      with one row per request slot, slot zero the padding sentinel.

    ``options``: ``tables`` (``[{page_tokens, windowed}]`` per block table),
    ``device`` (CUDA ordinal, ``-1`` for the host), ``step_rows`` and
    ``sampling`` (``{max_steps, entropy_bound, t_min, t_max,
    confidence_threshold, stability_threshold}``, absent on a rank that
    serves readouts only).

    Returns:
        ``readout_attend``, ``readout_candidates``, ``readout_finish``,
        ``readout_broadcast``, ``step`` (generating ranks), ``capture``,
        ``replay``, ``release`` and ``kernels``.
    """
    context = prepared.context
    if context is None:
        raise ValueError("canvas functions require a bound component")
    denoiser = _denoiser(prepared.module)
    backbone = denoiser.backbone
    pipeline = backbone._pipeline
    last = pipeline.rank == pipeline.size - 1
    tensors = {
        str(name): torch.from_dlpack(value) for name, value in buffers.items()
    }
    attention = _Attention(
        tensors,
        [
            _Table(int(table["page_tokens"]), bool(table["windowed"]))
            for table in options["tables"]
        ],
    )
    device = int(options["device"])
    length = denoiser.canvas.length
    state = tuple(
        tensors[f"attend_state.{index}"]
        for index in range(
            sum(name.startswith("attend_state.") for name in tensors)
        )
    )
    input_ids, positions = tensors["input_ids"], tensors["positions"]
    candidate_index = tensors["candidate_index"]
    candidate_logprobs = tensors["candidate_logprobs"]
    # With an exchanging tail, a readout's expert step spans the attend
    # pass and the tail: the pass leaves the final layer's experts to the
    # tail, which starts with every earlier expert layer reached.
    exchange = context.experts
    exchanging = tail_exchanges(
        denoiser,
        exchange=exchange is not None,
        attention_ranks=0 if exchange is None else exchange.attention_ranks,
    )
    final = tuple(backbone.layers.values())[-1]
    attended = context.expert_layers(denoiser) - context.expert_layers(final)

    def _normalized(rows: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary FP32 log-probabilities of attend-state ``rows``."""
        hidden = denoiser.finish(state, rows)
        logits = denoiser.compute_logits(
            hidden,
            token_indices=torch.arange(
                rows.shape[0], dtype=torch.int64, device=rows.device
            ),
        )
        # The softmax normalizes over the whole vocabulary, so tensor shards
        # gather their columns for the slot rows alone.
        return torch.log_softmax(cast(Logits, logits).gather().float(), dim=-1)

    def _attend(rows: int, tokens: int, batch: AttentionBatch) -> None:
        values = denoiser.attend(
            CanvasInput(input_ids[:tokens], positions[:tokens], batch)
        )
        if last:
            for value, target in zip(values, state, strict=True):
                target[: value.shape[0]].copy_(value)

    def _attend_step(rows: int, tokens: int, capacity: int, batch) -> None:
        with context.expert_step(capacity, join=not exchanging):
            _attend(rows, tokens, batch)

    def readout_attend(
        rows: int,
        tokens: int,
        visible: int,
        queries: Sequence[int],
        prefixes: Sequence[int],
        widths: Sequence[int],
        starts: Sequence[int],
        expert_capacity: int,
    ) -> None:
        """Evaluate ``tokens`` canvas tokens of ``rows`` sequences eagerly.

        Writes the final stage's per-token ``attend`` state into the leading
        rows of ``attend_state``; earlier stages forward their activations.
        ``visible`` is the widest sequence. ``expert_capacity`` is the
        agreed capacity of the readout's expert step (zero without
        expert-parallel layers); with an exchanging tail the step continues
        in ``readout_finish``.
        """
        batch = attention.batch(
            int(rows), int(visible), queries, prefixes, widths, starts
        )
        context.bind_attention(batch)
        with context.activate(), torch.inference_mode():
            _attend_step(int(rows), int(tokens), int(expert_capacity), batch)

    def readout_tail(slots: int) -> torch.Tensor:
        """Log-probabilities ``[slots, vocab]`` of ``tail_rows[:slots]``.

        The rows index ``attend_state``; rows past a call's live slots hold
        earlier values and their results are not read.
        """
        with context.activate(), torch.inference_mode():
            return _normalized(tensors["tail_rows"][: int(slots)])

    def readout_candidates(
        tail: _Graph | _Tail, start: int, count: int
    ) -> None:
        """Select candidates ``start:start + count`` from a tail's output.

        ``candidate_index`` holds each candidate's flat index into the
        tail's ``[slots, vocab]`` log-probabilities: ``(slot - first slot) *
        vocab + token`` for a replayed tail chunk, ``slot * vocab + token``
        for an eager tail.
        """
        start, count = int(start), int(count)
        with context.activate(), torch.inference_mode():
            torch.index_select(
                tail.output.reshape(-1),
                0,
                candidate_index[start : start + count],
                out=candidate_logprobs[start : start + count],
            )

    def readout_finish(slots: int, expert_capacity: int) -> _Tail:
        """Complete every slot of a readout eagerly, on the final stage.

        Completes the final layer for ``slot_tokens[:slots]`` (exchanging
        tokens with expert peers where the layer does, as the rest of the
        readout's expert step of ``expert_capacity``) and returns the
        ``[slots, vocab]`` log-probabilities for ``readout_candidates``,
        whose ``candidate_index`` then holds ``slot * vocab + token``.
        """
        if not last:
            raise ValueError("readout tails belong to the final stage")
        capacity = int(expert_capacity) if exchanging else 0
        with (
            context.activate(),
            torch.inference_mode(),
            context.expert_step(capacity, reached=attended),
        ):
            return _Tail(_normalized(tensors["slot_tokens"][: int(slots)]))

    def readout_broadcast(count: int) -> None:
        """Share the final stage's ``count`` candidate values.

        Every pipeline stage then reports the same readout.
        """
        with context.activate(), torch.inference_mode():
            pipeline.broadcast(
                candidate_logprobs[: int(count)], src=pipeline.size - 1
            )

    functions: dict[str, Callable[..., Any]] = {
        "readout_attend": readout_attend,
        "readout_candidates": readout_candidates,
        "readout_finish": readout_finish,
        "readout_broadcast": readout_broadcast,
    }

    sampling = options.get("sampling")
    step_rows = int(options.get("step_rows") or 0)
    if sampling is not None:
        constants = sampler.CanvasSampling(
            steps=int(sampling["max_steps"]),
            entropy_bound=float(sampling["entropy_bound"]),
            t_min=float(sampling["t_min"]),
            t_max=float(sampling["t_max"]),
            confidence=float(sampling["confidence_threshold"]),
            stability=int(sampling["stability_threshold"]),
            eos_ids=tuple(denoiser.canvas.eos_token_ids),
            pad_id=denoiser.canvas.pad_token_id,
        )
        if step_rows < 1:
            raise ValueError("canvas steps need a positive chunk row count")
        columns = tensors["step.columns"]
        views = {
            name: tensors[f"step.{name}"]
            for name in ("canvas", "history", "self_conditioning")
        }
        results = tensors["step.results"]
        continued = tensors["step.continued"]
        banks = {
            name: tensors[f"bank.{name}"]
            for name in ("canvas", "history", "self_conditioning", "live")
        }
        workspace = sampler.CanvasWorkspace(
            tensors["workspace.weights"],
            tensors["workspace.normalizer"],
            tensors["workspace.product"],
            tensors["workspace.scratch"],
        )
        embedding = backbone.embedding
        if embedding is None:
            raise ValueError("canvas steps require the token embedding table")
        vocab = workspace.weights.shape[1]
        hidden_size = views["self_conditioning"].shape[2]

        def _step(rows: int, first: bool, batch: AttentionBatch) -> None:
            slots, seeds, blocks, steps = columns[:, :rows].unbind(0)
            soft = views["self_conditioning"][:rows].view(
                rows * length, hidden_size
            )
            rows_state = sampler.CanvasState(
                seed=seeds,
                block=blocks,
                step=steps,
                canvas=views["canvas"][:rows],
                history=views["history"][:rows],
                self_conditioning=soft,
            )
            # A queued step runs only while its block continues; step zero
            # starts a block and always runs. Decided on the device, so the
            # call never waits for the host to observe the step before it.
            active = (steps == 0) | (banks["live"].index_select(0, slots) != 0)
            sampler.start_canvas(rows_state, vocab_size=vocab)
            # A canvas's first pass has a zero self-conditioning signal,
            # whose mixing is exactly the unweighted norm.
            hidden = denoiser(
                CanvasInput(
                    rows_state.canvas.reshape(-1),
                    positions[: rows * length],
                    batch,
                    None if first else soft,
                )
            )
            output = results[:rows]
            for start in range(0, rows, step_rows):
                stop = min(rows, start + step_rows)
                count = stop - start
                logits = denoiser.compute_logits(
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
                    _table(embedding),
                    backbone.embedding_scale,
                    _rows(rows_state, start, stop),
                    constants,
                    scores=sampler.CanvasScores.empty(
                        count, length, device=hidden.device
                    ),
                    decision=decision,
                    workspace=_chunk(workspace, count * length, device),
                )
                output[start:stop, 0] = decision.finished[:, 0]
                output[start:stop, 1:] = decision.tokens

            # The commit skips zero targets, keeping skipped request rows
            # and the shared padding row untouched across concurrent lanes.
            output[:, 0].masked_fill_(~active, STEP_SKIPPED)
            targets = torch.where(active, slots, torch.zeros_like(slots))
            sampler.commit_rows(banks["canvas"], targets, rows_state.canvas)
            if rows_state.history.shape[1]:
                sampler.commit_rows(
                    banks["history"], targets, rows_state.history
                )
            sampler.commit_rows(
                banks["self_conditioning"],
                targets,
                views["self_conditioning"][:rows],
            )
            # Whether each row's block continues after this step: the
            # completion flag of its call and its slot's continuation.
            flags = continued[:rows]
            flags.copy_(output[:, 0] == STEP_CONTINUED)
            sampler.commit_rows(banks["live"], targets, flags)

        def _expert_step(rows: int, first: bool, capacity: int, batch):
            with context.expert_step(capacity):
                _step(rows, first, batch)

        def step(
            rows: int,
            first: bool,
            queries: Sequence[int],
            prefixes: Sequence[int],
            widths: Sequence[int],
            starts: Sequence[int],
            expert_capacity: int,
        ) -> None:
            """Step ``rows`` gathered canvases once, eagerly.

            ``first`` declares every row at step zero. Writes
            ``step.results[:rows]`` and commits active rows to the banks.
            The whole step is one expert step of ``expert_capacity`` (zero
            without expert-parallel layers).
            """
            rows = int(rows)
            batch = attention.batch(
                rows, length, queries, prefixes, widths, starts
            )
            context.bind_attention(batch)
            with context.activate(), torch.inference_mode():
                _expert_step(rows, bool(first), int(expert_capacity), batch)

        functions["step"] = step

    def capture(
        function: str, arguments: Sequence, pool, expert_capacity: int
    ) -> _Graph:
        """Capture one canvas function for a fixed shape in ``pool``.

        ``function`` is ``readout_attend`` (``[rows, tokens, visible,
        queries, prefixes, widths, starts]``), ``readout_tail``
        (``[slots]``) or ``step`` (``[rows, first, queries, prefixes,
        widths, starts]``), with the bucket's padded host metadata already
        written by the worker. The function runs once eagerly first, so
        lazy initialization and algorithm selection (which may synchronize
        the device) happen outside the capture; that eager run is an expert
        step of ``expert_capacity`` like the captured one.
        """
        arguments = list(arguments)
        name = str(function)
        capacity = int(expert_capacity)
        # Distributed expert weights submit peer reads between segments.
        segments = None if context.weights is None else context.weights.capture
        if name == "readout_tail":
            slots = int(arguments[0])
            readout_tail(slots)
            with context.activate():
                graph = graphs.capture(
                    lambda: _normalized(tensors["tail_rows"][:slots]),
                    stream=context.stream,
                    pool=pool,
                    segments=segments,
                )
            return _Graph(graph, None)

        rows = int(arguments[0])
        if name == "readout_attend":
            tokens, visible = int(arguments[1]), int(arguments[2])
            queries, prefixes, widths, starts = arguments[3:7]
            readout_attend(
                rows,
                tokens,
                visible,
                queries,
                prefixes,
                widths,
                starts,
                capacity,
            )
            body = functools.partial(_attend_step, rows, tokens, capacity)
        elif name == "step" and "step" in functions:
            first, visible = bool(arguments[1]), length
            queries, prefixes, widths, starts = arguments[2:6]
            functions["step"](rows, first, *arguments[2:6], capacity)
            body = functools.partial(_expert_step, rows, first, capacity)
        else:
            raise ValueError(f"canvas function {name!r} is not capturable")

        # The captured attention reads these fixed views; a replay plans
        # them again for its own host lengths.
        widths = tuple(int(width) for width in widths)

        def views_for(queries, prefixes, starts) -> AttentionBatch:
            return attention.batch(
                rows, visible, queries, prefixes, widths, starts
            )

        batch = views_for(queries, prefixes, starts)
        context.bind_attention(batch)
        with context.activate():
            graph = graphs.capture(
                functools.partial(body, batch),
                stream=context.stream,
                pool=pool,
                segments=segments,
            )

        def rebind(queries, prefixes, starts) -> None:
            context.bind_attention(
                views_for(queries, prefixes, starts), replay=True
            )

        return _Graph(graph, rebind)

    def replay(graph: _Graph, queries, prefixes, starts) -> None:
        """Replay a captured function.

        Its attention is planned for the live host lengths first; a tail
        has none (``None``).
        """
        if graph.rebind is not None:
            graph.rebind(queries, prefixes, starts)
        with context.activate():
            graph.graph.replay()

    def release(graph: _Graph) -> None:
        graph.graph.release()
        graph.rebind = None

    def kernels() -> list[dict[str, Any]]:
        """Canvas kernel records: the readout tail and the step product.

        On CUDA, one ``product`` record per step chunk shape the
        self-conditioning product has run: its positions, vocabulary and
        hidden sizes and the cuBLASLt algorithm it uses.
        """
        records: list[dict[str, Any]] = []
        if sampling is None or device < 0:
            return records
        from uniserve_kernels.diffusion import canvas as kernels

        # Bound sampling required the embedding table above.
        assert embedding is not None
        table = _table(embedding)[:vocab]
        for rows in range(1, step_rows + 1):
            chunk = _chunk(workspace, rows * length, device)
            algorithm = kernels.product_algorithm(
                chunk.weights, table, chunk.product, chunk.scratch
            )
            if algorithm is not None:
                records.append(
                    {
                        "path": "canvas_sampler.self_conditioning",
                        "op": "product",
                        "positions": rows * length,
                        "vocab": vocab,
                        "hidden": hidden_size,
                        "provider": "cublaslt",
                        **algorithm,
                    }
                )
        return records

    functions.update(
        capture=capture, replay=replay, release=release, kernels=kernels
    )
    return functions
