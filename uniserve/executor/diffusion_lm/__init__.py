"""The token-denoising calls of a diffusion language model on one rank.

:class:`DiffusionLMExecutor` serves the rank's diffusion LM lanes: one per
diffusion LM component, each on the device's lane stream. A batch carries
readouts, which evaluate blocks of tokens over their requests' cached
prefixes and read candidate log-probabilities at answer slots, or denoising
steps, which step each request's resident block by one step of block
diffusion. The worker reaches it through ``uniserve.executor.Executor`` at
a few points:

- :meth:`DiffusionLMExecutor.manifest`: each lane and the device bytes the
  lanes allocate beside the cache pool;
- :meth:`DiffusionLMExecutor.bind`: the storage capacity planning granted
  (the text executor's ``TextBinding``); the lanes construct their runners,
  columns and banks, and :meth:`DiffusionLMExecutor.step_rows` states the
  blocks a sampler chunk steps;
- :meth:`DiffusionLMExecutor.warmup_phases`,
  :meth:`DiffusionLMExecutor.warmup_phase`,
  :meth:`DiffusionLMExecutor.warmup_finish` and
  :meth:`DiffusionLMExecutor.seal`: graph capture in one phase per lane
  around the worker's scratch tables (:mod:`.warmup`), then serving
  replays only;
- :meth:`DiffusionLMExecutor.start` and :meth:`DiffusionLMExecutor.release`:
  the requests and seeds the slots hold;
- :meth:`DiffusionLMExecutor.execute`: one batch
  (:class:`~.lane.DiffusionLMLane`); the lanes commit nothing past the
  worker's commit point;
- :meth:`DiffusionLMExecutor.reopen_experts`,
  :meth:`DiffusionLMExecutor.kernels` and :meth:`DiffusionLMExecutor.close`.

The parts:

- :mod:`.bounds`: the model's facts and every bound of the lane
  (:class:`~.bounds.Facts`, :class:`~.bounds.Bounds`,
  :class:`~.bounds.ServedSampling`);
- :mod:`.shapes`: the graph buckets, their padding and the readout slot
  chunks (:class:`~.shapes.Shapes`);
- :mod:`.calls`: the validation and packing of a batch's calls, the
  requests the lane's slots hold and the batch's completion records
  (:func:`~.calls.pack_readouts`, :func:`~.calls.pack_steps`,
  :class:`~.calls.Seeds`);
- :mod:`.buffers`: the lane's device columns and resident banks
  (:class:`~.buffers.Columns`, :class:`~.buffers.Banks`);
- :mod:`.lane`: a lane's batches (:class:`~.lane.DiffusionLMLane`);
- :mod:`.warmup`: a lane's warmup forwards and phase.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from uniserve.diffusion import canvas as sampler
from uniserve.executor.text.buffers import PageGroup
from uniserve.executor.text.protocol import CALL_FIELD_NAMES
from uniserve.runner.attention import AttentionMetadata
from uniserve.runner.diffusion_lm import DiffusionLMRunner, model_at

from .bounds import Bounds, Facts, ServedSampling
from .buffers import Banks, BankShape, Columns, Dimensions
from .calls import Seeds
from .lane import DiffusionLMLane
from .shapes import SLOT_BUCKETS
from .warmup import Phase, phases

__all__ = ["DiffusionLMExecutor"]

# The call kind a diffusion LM lane serves, by its protocol name.
KIND = "token_denoising"


@dataclass(frozen=True, slots=True)
class _Component:
    """A diffusion language model component placed on this rank."""

    name: str
    path: str


def _components(settings: Mapping[str, Any]) -> list[_Component]:
    """Every device component declaring a diffusion LM call that serves
    token-denoising calls.
    """  # noqa: D205
    placed = []
    for value in settings["components"]:
        if value["host"]:
            continue
        paths = {
            str(call["capability"]): str(call["path"])
            for call in value["calls"]
        }
        if "diffusion_lm" in paths and KIND in value["kinds"]:
            placed.append(_Component(str(value["name"]), paths["diffusion_lm"]))
    return placed


class DiffusionLMExecutor:
    """The diffusion LM lanes of one rank.

    Args:
        loaded: The loaded model.
        settings: The rank's deployment: ``device`` (``null`` on the host),
            the placed ``components`` and the ``pages`` section: the batch
            bounds, ``graph_policy``, ``cache_groups``, the served
            ``canvas_sampling`` (``null`` when the deployment generates no
            block) and the ``expert`` union's ``exchanges_tokens`` (whether
            routed experts exchange tokens rather than prefetch weights) and
            ``attention_ranks``.
    """

    def __init__(self, loaded, settings: Mapping[str, Any]) -> None:
        self.loaded = loaded
        pages = settings["pages"]
        self.cuda = settings["device"] is not None
        self.graph_policy = str(pages["graph_policy"])
        self.max_batch_calls = int(pages["max_batch_calls"])
        self.max_batch_tokens = int(pages["max_batch_tokens"])
        self.max_model_len = int(pages["max_model_len"])
        self.groups = [
            PageGroup.from_json(group) for group in pages["cache_groups"]
        ]
        sampling = pages["canvas_sampling"]
        self.sampling = (
            None if sampling is None else ServedSampling.from_json(sampling)
        )
        expert = pages["expert"]
        self.components = _components(settings)
        self.facts = {
            component.name: Facts.read(
                model_at(loaded, component.path),
                exchange=bool(expert["exchanges_tokens"]),
                attention_ranks=int(expert["attention_ranks"]),
            )
            for component in self.components
        }

        self.lanes: dict[int, DiffusionLMLane] = {}
        self.seeds: Seeds | None = None
        self._phases: list[Phase] = []

    def _bounds(self, facts: Facts, slots: int, units: int | None) -> Bounds:
        return Bounds.derive(
            max_batch_calls=self.max_batch_calls,
            max_batch_tokens=self.max_batch_tokens,
            max_model_len=self.max_model_len,
            slots=slots,
            groups=self.groups,
            units=units,
            facts=facts,
            cuda=self.cuda,
            graph_policy=self.graph_policy,
            sampling=self.sampling,
        )

    def _dimensions(self, bounds: Bounds, facts: Facts) -> Dimensions:
        denoising = bounds.denoise
        return Dimensions(
            rows=bounds.rows,
            tokens=bounds.tokens,
            tables=len(bounds.widths),
            width=max(bounds.widths, default=1),
            hidden=facts.hidden_size,
            attend_widths=facts.attend_widths,
            compute_dtype=facts.compute_dtype,
            length=facts.sequence_length,
            denoise_rows=0 if denoising is None else denoising.rows,
            depth=self._depth(),
            embedding_dtype=facts.embedding_dtype,
        )

    def _depth(self) -> int:
        # The argmax blocks a slot's stopping rule compares.
        return 0 if self.sampling is None else self.sampling.stability_threshold

    def _banks(self, facts: Facts, slots: int) -> BankShape:
        return BankShape(
            slots=slots,
            length=facts.sequence_length,
            depth=self._depth(),
            hidden=facts.hidden_size,
            dtype=facts.embedding_dtype,
        )

    # The manifest.

    def manifest(self) -> dict[str, Any]:
        """The rank's diffusion LM lanes and what capacity planning reads.

        Returns ``lanes`` (each lane's manifest entry, with the output words
        of its largest denoising batch: a readout batch of more words leases
        storage of its own) and ``pages``: the call fields and the device
        bytes. ``fixed_bytes`` plus ``slot_bytes`` per request slot bounds
        what binding allocates on the device beside the cache pool: every
        lane's columns and attention metadata at the batch bounds (rows at
        ``max_batch_calls``, before the pool bounds them, which may
        over-reserve), a generating lane's sampler workspace, its banks'
        padding row and alignment, and its banks per slot.
        """
        lanes = []
        fixed = slot_bytes = 0
        for component in self.components:
            facts = self.facts[component.name]
            bounds = self._bounds(facts, self.max_batch_calls, None)
            fixed += self._dimensions(bounds, facts).bytes()
            denoising = bounds.denoise
            if denoising is not None:
                banks = self._banks(facts, 1)
                slot_bytes += banks.slot_bytes()
                fixed += banks.fixed_bytes()
                fixed += sampler.CanvasWorkspace.nbytes(
                    denoising.chunk_rows,
                    facts.sequence_length,
                    facts.vocab_size,
                    facts.hidden_size,
                    dtype=facts.embedding_dtype,
                    device_type="cuda" if self.cuda else "cpu",
                )
            lanes.append(
                {
                    "component": component.name,
                    "kinds": [KIND],
                    "microbatches": 1,
                    "prepares": False,
                    "commits": False,
                    "resources": ["pages"],
                    "output_words": self.max_batch_calls
                    * (1 + facts.sequence_length),
                }
            )
        return {
            "lanes": lanes,
            "pages": {
                "call_fields": list(CALL_FIELD_NAMES),
                "fixed_bytes": int(fixed),
                "slot_bytes": int(slot_bytes),
            },
        }

    # Startup.

    def bind(self, binding, union=None, *, joins: bool = False) -> None:
        """Construct every lane's runner on the granted storage.

        Allocates each lane's columns, attention metadata and, on a
        generating lane, its zeroed banks on the lane stream, then
        constructs the numerical runners in lane order.

        Args:
            binding: The text executor's ``TextBinding`` of the diffusion LM
                lanes: the slots, the fitted pool's units, the K/V pool
                object, the block tables (``unit_tables``, ``start_pages``,
                ``table_shapes``), each lane's ``LaneBinding`` (one stream,
                its scratch, graph pool, relay staging and stream groups)
                and the prefetch object.
            union: The rank's expert union member, or ``None``.
            joins: The union's joins run through the first lane's runner:
                the rank has no text lane that provides them.

        Raises:
            ValueError: A lane names a component the rank does not hold or
                more than one stream, or graphs disagree with the pools.
            NotImplementedError: The union runs microbatches, or a lane
                stream is not the union's.
        """
        if union is not None and union.microbatches > 1:
            raise NotImplementedError(
                "diffusion LMs run expert steps without microbatches"
            )
        by_name = {component.name: component for component in self.components}
        self.seeds = Seeds(binding.slots)
        tables = binding.tables
        # The cache group of each numerical block table, group-major.
        table_groups = np.repeat(
            np.arange(len(self.groups), dtype=np.int64),
            [group.units_per_page for group in self.groups],
        )
        # The lanes compute where the worker's block tables live: on the
        # rank's device, or on the host.
        device = tables["unit_tables"].device
        for group, lane in enumerate(binding.lanes):
            component = by_name.get(lane.component)
            if component is None:
                raise ValueError(
                    "the worker binds a diffusion LM lane of "
                    f"{lane.component!r}, which the rank does not hold"
                )
            if len(lane.streams) != 1:
                raise ValueError("a diffusion LM lane runs on one stream")
            stream = lane.streams[0]
            if (
                union is not None
                and stream is not None
                and stream.cuda_stream != union.streams[0].cuda_stream
            ):
                raise NotImplementedError(
                    "expert parallelism runs every forward on one execution "
                    "lane"
                )
            pool = None if lane.pools is None else lane.pools[0]
            facts = self.facts[component.name]
            bounds = self._bounds(facts, binding.slots, binding.units)
            if bounds.graphs != (pool is not None):
                raise ValueError(
                    "a diffusion LM lane captures graphs exactly when its "
                    "stream has a graph pool"
                )
            dims = self._dimensions(bounds, facts)
            denoising = bounds.denoise
            with torch.cuda.stream(stream):
                columns = Columns(dims, device)
                metadata = AttentionMetadata(
                    bounds.block_size,
                    [int(value) for value in bounds.windowed],
                    tables["unit_tables"],
                    tables["start_pages"],
                    tables["table_shapes"],
                    bounds.rows,
                    bounds.tokens,
                    dims.width,
                    device,
                )
                banks = (
                    None
                    if denoising is None
                    else Banks(self._banks(facts, binding.slots), device)
                )

            # A generating runner steps blocks with the served sampling and
            # commits them to the banks.
            sampling: list[Any] = []
            if banks is not None:
                served = self.sampling
                assert served is not None
                sampling = [
                    served.max_steps,
                    served.entropy_bound,
                    served.t_min,
                    served.t_max,
                    served.confidence_threshold,
                    served.stability_threshold,
                    banks.canvas,
                    banks.history,
                    banks.self_conditioning,
                    banks.live,
                ]
            # The head projects a readout's slot bucket, whose rows may
            # exceed a forward's tokens; the execution context is prepared
            # for both.
            runner = DiffusionLMRunner(
                self.loaded,
                component.path,
                # A lane on the host computes on the current stream.
                0 if stream is None else int(stream.cuda_stream),
                lane.scratches[0],
                binding.unit_pool,
                None if union is None else union.exchanges[0],
                binding.prefetch,
                lane.collectives[0],
                max(bounds.tokens, SLOT_BUCKETS[-1]),
                bounds.rows,
                list(bounds.block_size),
                [int(value) for value in bounds.windowed],
                0 if denoising is None else denoising.chunk_rows,
                *sampling,
            )
            if union is not None and joins and group == 0:
                # Idle ranks join expert steps through the runner's expert
                # layers.
                union.bind_joins([runner], [pool])
            self.lanes[lane.index] = DiffusionLMLane(
                index=lane.index,
                group=group,
                component=component.name,
                label=(
                    f"{component.name}.{component.path}"
                    if component.path
                    else component.name
                ),
                runner=runner,
                bounds=bounds,
                facts=facts,
                sampling=None if banks is None else self.sampling,
                seeds=self.seeds,
                columns=columns,
                metadata=metadata,
                banks=banks,
                stream=stream,
                pool=pool,
                relay_flags=lane.relay_flags,
                union=union,
                table_groups=table_groups,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step_rows(self) -> int:
        """Return the blocks a sampler chunk steps at once.

        The engine warms its serving path with them; zero without a
        generating lane.
        """
        return max(
            (
                lane.bounds.denoise.chunk_rows
                for lane in self.lanes.values()
                if lane.bounds.denoise is not None
            ),
            default=0,
        )

    def warmup_phases(self) -> list[dict[str, Any]]:
        """Return the warmup phases in the order every rank runs them.

        Each names its lane, the scratch tables' lengths of slots ``1..``
        and each attending forward's rows ``[slot, prefix, query]``.
        """
        self._phases = phases(
            [self.lanes[index] for index in sorted(self.lanes)]
        )
        return [phase.describe() for phase in self._phases]

    def warmup_phase(self, index: int, *pages: torch.Tensor) -> None:
        """Run warmup phase ``index`` given each attending forward's pages.

        ``pages[k]`` is the host ``int64 [2 * G, rows]`` of forward ``k``:
        per cache group, each row's first selected page and page count.
        """
        phase = self._phases[int(index)]
        phase.run([np.asarray(page.numpy(), dtype=np.int64) for page in pages])

    def warmup_finish(self) -> None:
        """End warmup: every graph was captured in the lanes' phases.

        The expert union's warmup ends after every executor's lanes
        finished (``ExpertUnion.finish_warmup``), which the rank's executor
        calls.
        """

    def seal(self) -> None:
        """Refuse further captures: serving replays what warmup captured."""
        for lane in self.lanes.values():
            lane.seal()

    def reopen_experts(self) -> None:
        """Release every lane's graphs so warmup captures them anew.

        Called when the expert union rebuilt the workspaces the graphs
        addressed; the warmup phases that follow capture again and
        :meth:`seal` closes the tables.
        """
        for lane in self.lanes.values():
            lane.reopen()

    # Requests.

    def start(self, requests: Sequence[Mapping[str, Any]]) -> list[Any]:
        """Admit started requests (``NewRequest`` JSON); returns the refused.

        Records each request's key and sampling seed in its slot: a step
        checks its call against them and draws from the seed. A request's
        first step starts its block in the banks before any step reads
        them, so admission resets no device state. No request is refused.

        Raises:
            RuntimeError: The lanes are not bound.
            ValueError: A slot is outside the lanes, or a seed is not a
                64-bit unsigned integer.
        """
        if self.seeds is None:
            raise RuntimeError("the diffusion LM lanes are not bound")
        for value in requests:
            key = value["request_key"]
            generation = value.get("generation")
            seed = None
            if generation is not None:
                seed = generation["sampling"].get("seed")
            self.seeds.start(
                int(value["request_pool_idx"]),
                int(key["request_id"]),
                int(key["request_epoch"]),
                None if seed is None else int(seed),
            )
        return []

    def release(self, slots: Sequence[int]) -> None:
        """Forget the retired slots' requests."""
        if self.seeds is not None:
            self.seeds.release([int(slot) for slot in slots])

    # Batches.

    def execute(self, lane: int, step: str, *tensors: torch.Tensor) -> int:
        """Run a token-denoising batch on lane ``lane``; return the record
        words written.

        ``tensors`` are, in order, the host call columns, forward columns,
        flat tokens and records, and the page-locked output words.

        Raises:
            ValueError: The batch carries feature tensors.
        """  # noqa: D205
        calls, forward, tokens, records, words, *features = tensors
        if features:
            raise ValueError("token-denoising calls carry no features")
        return self.lanes[int(lane)].execute(
            json.loads(str(step)), calls, forward, tokens, records, words
        )

    def kernels(self) -> list[tuple[str, list[str]]]:
        """Each lane's kernel-table label and selected kernels."""
        return [
            entry
            for index in sorted(self.lanes)
            for entry in self.lanes[index].kernels()
        ]

    def close(self, aborted: bool) -> None:
        """Release every lane, newest first; the worker drained the streams."""
        failure: BaseException | None = None
        for index in sorted(self.lanes, reverse=True):
            try:
                self.lanes[index].close(bool(aborted))
            except Exception as error:
                failure = failure or error
        self.lanes.clear()
        if failure is not None:
            raise failure
