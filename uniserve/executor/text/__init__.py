"""The executor of a causal language model's token batches on one rank.

:class:`TextExecutor` serves the rank's text computations: one per causal
language model component, each running on one runner per expert
microbatch. The worker reaches it through ``uniserve.executor.Executor`` at
a few points:

- :meth:`TextExecutor.manifest`: each computation, the call fields, the
  device bytes the computations allocate beside the cache pool and the
  output words of a computation's largest batch;
- :meth:`TextExecutor.bind`: the storage capacity planning granted
  (:class:`TextBinding`); the computations construct their runners;
- :meth:`TextExecutor.warmup_phases`, :meth:`TextExecutor.warmup_phase`,
  :meth:`TextExecutor.warmup_finish` and :meth:`TextExecutor.seal`: graph
  capture in phases around the worker's scratch tables
  (:mod:`.warmup`), then serving replays only;
- :meth:`TextExecutor.start` and :meth:`TextExecutor.release`: the
  requests that start in slots and the slots that retire;
- :meth:`TextExecutor.execute` and :meth:`TextExecutor.commit`: one token
  batch (:class:`~.computation.TextComputation`), then its slot state past
  the batch's commit point;
- :meth:`TextExecutor.kernels` and :meth:`TextExecutor.close`.

The modules hold the parts: the batch protocol (:mod:`.protocol`), graph
shapes and tables (:mod:`.shapes`, :mod:`.graphs`), per-slot request
constants and state (:mod:`.state`), sampling inputs and draws
(:mod:`.sampling`, :mod:`.philox`), forward rows (:mod:`.rows`), vision and
latent rows (:mod:`.context`), prompt scoring (:mod:`.prompt`), commit rows
(:mod:`.commit`), the runners' bounds, columns and staging
(:mod:`.buffers`), the runner of one binding (:mod:`.runner`), a
computation's batches (:mod:`.computation`) and warmup (:mod:`.warmup`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from uniserve.runner.causal_lm import CausalLMRunner, hidden_size, vocab_size

from .buffers import InputBuffers, PageGroup, TextBounds
from .computation import TextComputation
from .context import ImageDenoiser, ImageFacts
from .philox import DrawWindow
from .protocol import CALL_FIELD_NAMES, COMPLETION_FIELDS
from .runner import TextRunner
from .sampling import logprob_capacity
from .state import NewRequest, RequestTable, SlotState
from .warmup import Phase, finish, phases

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from uniserve.executor.experts import ExpertUnion

__all__ = ["ComputationBinding", "TextBinding", "TextExecutor"]

# The text call kinds a computation may serve, in protocol order.
TEXT_KINDS = ("prefill", "decode", "verify")


@dataclass(frozen=True, slots=True)
class ComputationBinding:
    """The storage the worker binds one text computation to.

    Attributes:
        index: The computation's index among its executor's computations.
        component: The component the computation serves.
        streams: One stream per expert microbatch, the first the device's
            primary stream; ``(None,)`` for a computation on the host.
        scratches: Each stream's scratch, which every execution context on
            it borrows.
        pools: Each stream's graph pool; ``None`` without graphs.
        relay_tokens: ``int64 [calls]`` staging of sampled tokens.
        relay_flags: ``uint8 [2, calls]`` staging of transition decisions
            (row 0) and completion flags (row 1).
        collectives: Per microbatch, the process group of each of the
            model's multi-member communicators on its stream, by
            communicator name; ``None`` for none.
    """

    index: int
    component: str
    streams: tuple[torch.cuda.Stream | None, ...]
    scratches: tuple[Any, ...]
    pools: tuple[Any, ...] | None
    relay_tokens: torch.Tensor
    relay_flags: torch.Tensor
    collectives: tuple[Mapping[str, Any] | None, ...]


@dataclass(frozen=True, slots=True)
class TextBinding:
    """What capacity planning granted the rank's text computations.

    Attributes:
        slots: The request slots.
        units: The prefix cache pool's units, sentinel included.
        unit_pool: The prefix cache's K/V pool object attention reads.
        tables: The worker's block tables by name (``unit_tables``,
            ``start_pages``, ``table_shapes``, ``verified_lengths``).
        computations: The text computations, in manifest order.
        prefetch: The distributed expert weights' prefetch object, or
            ``None``.
    """

    slots: int
    units: int
    unit_pool: Any
    tables: Mapping[str, torch.Tensor]
    computations: tuple[ComputationBinding, ...]
    prefetch: Any | None = None


@dataclass(frozen=True, slots=True)
class _Component:
    """A causal language model component placed on this rank."""

    name: str
    path: str
    kinds: tuple[str, ...]


def _components(settings: Mapping[str, Any]) -> list[_Component]:
    # Every device component declaring a causal LM call, with the text call
    # kinds it serves.
    placed = []
    for value in settings.get("components", ()):
        if value.get("host"):
            continue
        paths = {
            str(call["capability"]): str(call["path"])
            for call in value["calls"]
        }
        if "causal_lm" not in paths:
            continue
        kinds = tuple(kind for kind in TEXT_KINDS if kind in value["kinds"])
        placed.append(_Component(str(value["name"]), paths["causal_lm"], kinds))
    return placed


def _model_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, str(name).removeprefix("torch."), None)
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError(
            f"text models compute in a floating dtype, not {name!r}"
        )
    return dtype


class TextExecutor:
    """The text computations of one rank.

    Args:
        loaded: The loaded model.
        settings: The rank's deployment: ``graphs``, ``model_dtype``, the
            placed ``components``, the ``pages`` section (batch bounds,
            graph buckets, ``cache_groups``, ``image_processor``,
            ``media_axes``) and, for an expert union that exchanges
            tokens, the ``experts`` section (its ``microbatches``).
    """

    def __init__(self, loaded, settings: Mapping[str, Any]) -> None:
        self.loaded = loaded
        pages = settings["pages"]
        self.graphs = bool(settings["graphs"])
        self.dtype = _model_dtype(settings["model_dtype"])
        self.max_batch_calls = int(pages["max_batch_calls"])
        self.max_batch_tokens = int(pages["max_batch_tokens"])
        self.max_model_len = int(pages["max_model_len"])
        self.decode_sizes = [
            int(size) for size in pages["decode_graph_batch_sizes"]
        ]
        self.prefill_sizes = [
            int(size) for size in pages["prefill_graph_token_sizes"]
        ]
        self.prefill_outputs = bool(pages["prefill_outputs"])
        self.groups = [
            PageGroup.from_json(group) for group in pages["cache_groups"]
        ]
        self.image_processor = bool(pages.get("image_processor", False))
        axes = pages.get("media_axes")
        self.media_axes = None if axes is None else int(axes)
        experts = settings.get("experts")
        self.microbatches = (
            1 if experts is None else int(experts["microbatches"])
        )
        self.components = _components(settings)
        self.vocab = vocab_size(loaded)
        self.hidden = hidden_size(loaded)

        self.computations: dict[int, TextComputation] = {}
        self.state: SlotState | None = None
        self.table: RequestTable | None = None
        self.union: ExpertUnion | None = None
        self._units = 0
        self._phases: list[Phase] = []

    def _bounds(self, slots: int, units: int | None) -> TextBounds:
        return TextBounds.derive(
            max_batch_calls=self.max_batch_calls,
            max_batch_tokens=self.max_batch_tokens,
            max_model_len=self.max_model_len,
            slots=slots,
            groups=self.groups,
            units=units,
            graphs=self.graphs,
            decode_sizes=self.decode_sizes,
            prefill_token_sizes=self.prefill_sizes,
            prefill_outputs=self.prefill_outputs,
            media_axes=self.media_axes,
        )

    # The manifest.

    def manifest(self) -> dict[str, Any]:
        """The rank's text computations and what capacity planning reads.

        Returns ``computations`` (each computation's manifest entry, with the
        output words of its largest batch) and ``pages``: the call fields
        and the device bytes. ``fixed_bytes`` plus ``slot_bytes`` per
        request slot bounds what binding allocates on the device beside the
        cache pool: every runner's columns at the batch bounds (rows at
        ``max_batch_calls``, which may over-reserve with fewer slots) and
        the slot state.
        """
        bounds = self._bounds(self.max_batch_calls, None)
        dims = bounds.dimensions(
            vocab=self.vocab,
            local_vocab=self.vocab,
            hidden=self.hidden,
            embeddings=self.image_processor,
            dtype=self.dtype,
        )
        slot_bytes = SlotState.slot_bytes(self.vocab, self.dtype)
        runners = len(self.components) * self.microbatches
        # The padding slot's row and each state field's alignment slack.
        fixed = runners * dims.bytes() + slot_bytes + SlotState.ALIGNMENT_SLACK
        words = (
            self.max_batch_calls * (COMPLETION_FIELDS + 1)
            + self.microbatches * logprob_capacity(bounds.tokens)
            + bounds.tokens
        )
        return {
            "computations": [
                {
                    "component": component.name,
                    "kinds": list(component.kinds),
                    "microbatches": self.microbatches,
                    "prepares": False,
                    "commits": True,
                    "resources": ["pages"],
                    "output_words": int(words),
                }
                for component in self.components
            ],
            "pages": {
                "call_fields": list(CALL_FIELD_NAMES),
                "fixed_bytes": int(fixed),
                "slot_bytes": int(slot_bytes),
            },
        }

    # Startup.

    def bind(
        self, binding: TextBinding, union: ExpertUnion | None = None
    ) -> None:
        """Construct every computation's runners on the granted storage.

        Allocates the slot state beside the worker's verified lengths and
        each runner's columns, constructs the numerical runners (in
        computation and microbatch order on every rank: construction shares
        the vocabulary over the pipeline group) and hands an expert
        ``union`` each microbatch's context owner and graph pool.

        Raises:
            ValueError: A computation names a component the rank does not
                hold, or the runners' vocabulary differs from the model's.
        """
        if not binding.computations:
            return
        if union is not None and len(binding.computations) != 1:
            raise NotImplementedError(
                "expert parallelism serves one text component; "
                f"{len(binding.computations)} serve text"
            )
        by_name = {component.name: component for component in self.components}
        self._units = binding.units
        self.union = union
        bounds = self._bounds(binding.slots, binding.units)
        tables = binding.tables
        verified = tables["verified_lengths"]
        device = verified.device
        first = binding.computations[0].streams[0]
        with torch.cuda.stream(first):
            state = SlotState.allocate(
                binding.slots, self.vocab, self.dtype, verified
            )
        self.state = state
        self.table = RequestTable(binding.slots, self.vocab)
        window = DrawWindow(binding.slots)
        block_sizes = [table.block_size for table in bounds.tables]
        windowed = [int(table.windowed) for table in bounds.tables]

        for group, computation in enumerate(binding.computations):
            component = by_name.get(computation.component)
            if component is None:
                raise ValueError(
                    "the worker binds a text computation of "
                    f"{computation.component!r}, which the rank does not hold"
                )
            runners = []
            for microbatch, stream in enumerate(computation.streams):
                kernel = CausalLMRunner(
                    self.loaded,
                    component.path,
                    # A computation on the host runs on the current stream
                    # (handle 0) of no device (-1).
                    -1 if stream is None else int(stream.device.index),
                    0 if stream is None else stream.cuda_stream,
                    computation.scratches[microbatch],
                    binding.unit_pool,
                    None if union is None else union.exchanges[microbatch],
                    binding.prefetch,
                    computation.collectives[microbatch],
                    bounds.rows,
                    bounds.tokens,
                    bounds.width,
                    block_sizes,
                    windowed,
                    tables["unit_tables"],
                    tables["start_pages"],
                    tables["table_shapes"],
                    verified,
                    state.future_input_tokens,
                    state.logical_lengths,
                    state.predicates,
                    state.sampling_positions,
                    state.penalty_counts,
                    state.prompt_logits,
                    state.logit_bias,
                    state.adapters,
                )
                size, _, start, stop = kernel.vocab()
                if size != self.vocab:
                    raise ValueError(
                        f"the text runner's vocabulary ({size}) differs from "
                        f"the model's ({self.vocab})"
                    )
                dims = bounds.dimensions(
                    vocab=self.vocab,
                    local_vocab=stop - start,
                    hidden=kernel.hidden_size(),
                    embeddings=self.image_processor,
                    dtype=self.dtype,
                )
                with torch.cuda.stream(stream):
                    buffers = InputBuffers(dims, device)
                runners.append(
                    TextRunner(
                        owner=f"text runner of {component.name}",
                        kernel=kernel,
                        bounds=bounds,
                        buffers=buffers,
                        future_input_tokens=state.future_input_tokens,
                        logical_lengths=state.logical_lengths,
                        prompt_logits=state.prompt_logits,
                        stream=stream,
                        pool=None
                        if computation.pools is None
                        else computation.pools[microbatch],
                    )
                )
            self.computations[computation.index] = TextComputation(
                index=computation.index,
                group=group,
                label=(
                    f"{component.name}.{component.path}"
                    if component.path
                    else component.name
                ),
                runners=runners,
                bounds=bounds,
                state=state,
                table=self.table,
                window=window,
                relay_tokens=computation.relay_tokens,
                relay_flags=computation.relay_flags,
                union=union,
                prefill_outputs=self.prefill_outputs,
            )
            if union is not None:
                union.bind_joins(
                    [runner.kernel for runner in runners],
                    [runner.pool for runner in runners],
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def attach_image(
        self, facts: ImageFacts | None, denoiser: ImageDenoiser | None
    ) -> None:
        """Take the image half's facts and denoiser, which media calls ask."""
        for computation in self.computations.values():
            computation.facts = facts
            computation.denoiser = denoiser

    def warmup_phases(self) -> list[dict[str, Any]]:
        """Return the warmup phases in the order every rank runs them.

        Each names its computation, the scratch lengths of slots ``1..`` (or
        none) and each step's attention rows ``[slot, prefix, query]``.
        """
        self._phases = phases(
            [self.computations[index] for index in sorted(self.computations)],
            row_tokens=min(self.max_model_len, self.max_batch_tokens),
            units=max(self._units - 1, 0),
        )
        return [phase.describe() for phase in self._phases]

    def warmup_phase(self, index: int, *pages: torch.Tensor) -> None:
        """Run warmup phase ``index`` given each step's page columns.

        ``pages[k]`` is the host ``int64 [2 * G, rows]`` of step ``k``:
        per cache group, each row's first selected page and page count.
        """
        phase = self._phases[int(index)]
        phase.run([np.asarray(page.numpy(), dtype=np.int64) for page in pages])

    def warmup_finish(self) -> None:
        """Capture the graphs that follow every phase.

        The expert union's warmup ends after every executor's computations
        finished (``ExpertUnion.finish_warmup``), which the rank's executor
        calls.
        """
        finish(
            [self.computations[index] for index in sorted(self.computations)]
        )
        # The startup report of what each runner captured, which the graph
        # policy chose.
        for computation in self.computations.values():
            for runner in computation.runners:
                if runner.pool is not None:
                    _logger.info(
                        "%s: captured %d text graphs: %d decode and %d "
                        "prefill buckets",
                        runner.graphs.owner,
                        len(runner.graphs),
                        len(computation.bounds.decode),
                        len(computation.bounds.prefill),
                    )

    def seal(self) -> None:
        """Refuse further captures: serving replays what warmup captured."""
        for computation in self.computations.values():
            computation.seal()

    def run_eager(self, eager: bool) -> None:
        """Serve every computation eagerly (``eager``), or by the graphs
        captured at startup again.
        """  # noqa: D205
        for computation in self.computations.values():
            computation.run_eager(eager)

    def reopen_experts(self) -> None:
        """Release every computation's graphs so warmup captures them anew.

        Called when the expert union rebuilt the workspaces the graphs
        addressed; the warmup phases that follow capture again and
        :meth:`seal` closes the tables.
        """
        for computation in self.computations.values():
            computation.reopen()

    # Requests.

    def start(self, requests: Sequence[Mapping[str, Any]]) -> list[Any]:
        """Admit started requests (``NewRequest`` JSON); returns the refused.

        Installs each request's constants and resets its slot's state on
        the primary stream: a request resumed from a predecessor engine
        counts the tokens it generated there, one with logit biases installs
        them, one with an adapter selects it. No request is refused.
        """
        if self.state is None or self.table is None:
            raise RuntimeError("the text computations are not bound")
        started = [NewRequest.from_json(request) for request in requests]
        for request in started:
            self.table.start(request)
        if not started:
            return []
        stream = next(iter(self.computations.values())).stream
        with torch.cuda.stream(stream):
            self.state.reset([request.slot for request in started])
            for request in started:
                if request.generated_token_ids:
                    self.state.count_generated(
                        request.slot, request.generated_token_ids
                    )
                if request.sampling is not None and request.sampling.logit_bias:
                    self.state.install_bias(
                        request.slot, request.sampling.logit_bias
                    )
                if request.adapter:
                    self.state.select_adapter(request.slot, request.adapter)
        return []

    def release(self, slots: Sequence[int]) -> None:
        """Forget the retired slots' requests."""
        if self.table is not None:
            self.table.release([int(slot) for slot in slots])

    # Batches.

    def execute(
        self, computation: int, step: str, *tensors: torch.Tensor
    ) -> int:
        """Run a token batch of ``computation``; return record words written.

        ``tensors`` are, in order, the host call columns, forward columns,
        flat tokens and records, the page-locked output words, then the
        feature tensors the step JSON names.
        """
        calls, forward, tokens, records, words, *features = tensors
        return self.computations[int(computation)].execute(
            json.loads(str(step)),
            calls,
            forward,
            tokens,
            records,
            words,
            features,
        )

    def commit(self, computation: int) -> None:
        """Advance ``computation``'s slot state by its last executed batch."""
        self.computations[int(computation)].commit()

    def kernels(self) -> list[tuple[str, list[str]]]:
        """Each runner's kernel-table label and selected kernels."""
        return [
            entry
            for index in sorted(self.computations)
            for entry in self.computations[index].kernels()
        ]

    def close(self, aborted: bool) -> None:
        """Release every computation, newest first; the streams are drained."""
        failure: BaseException | None = None
        for index in sorted(self.computations, reverse=True):
            try:
                self.computations[index].close(bool(aborted))
            except Exception as error:
                failure = failure or error
        self.computations.clear()
        if failure is not None:
            raise failure
