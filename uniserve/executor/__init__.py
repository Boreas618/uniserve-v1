"""The numerical half of a worker rank: :class:`Executor`.

The worker (Rust) owns the control plane and the data plane: requests,
batches, the lane thread and its streams, product storage and transfers,
the prefix cache's pages, the latent ledger, the graph pools' device memory
and the host groups. The executor owns every numerical decision and
computation of the rank: the components' runners, their buffers, layouts
and buckets, graph capture and replay, sampling, slot state, expert steps
and the media work of host components.

The executor serves the rank's lanes. A lane is one computation it serves
under the lane's index: a component's call kinds on one of the worker's
streams (or the host). Each lane's manifest entry states whether its
batches are prepared (the executor first states what each call reads and
writes), whether they commit (the executor advances its slot state past a
batch's commit point) and which worker resources it uses (``pages`` of the
prefix cache, the ``latents`` ledger, host-task ``reports``).

The worker calls the executor at a few points only, each from a job of the
lane thread (or at startup), with JSON for structured values and tensors
for storage:

- :meth:`Executor.manifest`: every lane and what capacity planning reads;
- :meth:`Executor.expert_storage_bytes` and :meth:`Executor.connect_experts`:
  this rank's member of an expert union, collectively, before planning;
- :meth:`Executor.bind`: the storage capacity planning granted; the
  executor constructs its runners;
- :meth:`Executor.warmup_phases`, :meth:`Executor.warmup_phase`,
  :meth:`Executor.warmup_finish` and :meth:`Executor.seal`: warmup in
  phases around the worker's scratch tables, then serving;
- :meth:`Executor.start` and :meth:`Executor.release`: the requests a batch
  starts, which the executor may refuse, and the slots the worker retires;
- :meth:`Executor.prepare`, :meth:`Executor.execute` and
  :meth:`Executor.commit`: one batch of a lane;
- :meth:`Executor.join_expert_step`, :meth:`Executor.park_experts`,
  :meth:`Executor.promote_experts` and :meth:`Executor.rebuild_experts`: the
  expert union outside the rank's own steps and across a standby's
  promotion;
- :meth:`Executor.kernels` and :meth:`Executor.close`.

Within a call the executor never calls back into the worker, except that a
distributed expert layer's weight prefetch submits the worker's copy plan
of the layer, eagerly or as a captured step's break work between graph
segments.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from uniserve.executor.experts import ExpertUnion

__all__ = ["Executor"]


@dataclass
class _Binding:
    """The storage the worker granted, as ``bind`` receives it.

    Attributes:
        slots: Request slots.
        max_calls: The most calls of a batch.
        streams: Each lane stream by index (``None`` on the host).
        scratches: Each stream's scratch.
        host_scratch: The scratch the lanes on the host share; ``None``
            when every lane computes on a device.
        pools: Each stream's graph pool; ``None`` without graphs.
        peer_pools: Under graphs on a rank whose modules also run on other
            devices (a generation tower), each stream's ``(ordinal, pool)``
            on those devices, by stream index.
        lanes: Each lane's ``component``, ``kinds``, ``streams`` (indices)
            and ``resources``, in lane order.
        experts: The stream indices of the expert microbatches.
        unit_pool: The prefix cache's K/V pool object, with pages.
        units: The prefix cache pool's units, sentinel included.
        tables: The worker's block tables by name, with pages.
        staging: Each pages lane's relay staging (tokens, flags), by lane.
        banks: A latent ledger's two banks.
        reports: The host-task report pipe's descriptor.
        collectives: Each stream group: ``component``, ``copy`` and the
            communicator ``names`` it covers, with its process group.
        prefetch: The distributed expert weights' prefetch object.
    """

    slots: int
    max_calls: int
    streams: list[torch.cuda.Stream | None]
    scratches: list[Any]
    pools: list[Any] | None
    lanes: list[dict[str, Any]]
    experts: list[int]
    peer_pools: dict[int, list[tuple[int, Any]]] = field(default_factory=dict)
    host_scratch: Any | None = None
    unit_pool: Any | None = None
    units: int = 0
    tables: dict[str, torch.Tensor] = field(default_factory=dict)
    staging: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    banks: tuple[torch.Tensor, torch.Tensor] | None = None
    reports: int | None = None
    collectives: list[tuple[dict[str, Any], Any]] = field(default_factory=list)
    prefetch: Any | None = None

    @classmethod
    def parse(
        cls, memory: Mapping[str, Any], values: Sequence[Any]
    ) -> _Binding:
        """Read ``bind``'s JSON and values, in the worker's order.

        Values: one scratch per stream; the host scratch when
        ``host_scratch``; one graph pool per stream when
        ``pools``, then, stream by stream, one per device of ``peers``; with
        ``pages`` the unit pool then each named table; each
        pages lane's relay staging (tokens, flags) in lane order; the two
        latent banks when ``latents``; one process group per stream group;
        the prefetch object when ``prefetch``.
        """
        rest = list(values)

        def take(count: int) -> list[Any]:
            taken = rest[:count]
            if len(taken) != count:
                raise ValueError("bind received fewer values than it names")
            del rest[:count]
            return taken

        streams: list[torch.cuda.Stream | None] = []
        for handle, ordinal in memory["streams"]:
            streams.append(
                None
                if ordinal is None
                else torch.cuda.ExternalStream(
                    int(handle), device=torch.device("cuda", int(ordinal))
                )
            )
        scratches = take(len(streams))
        host_scratch = take(1)[0] if memory["host_scratch"] else None
        binding = cls(
            slots=int(memory["slots"]),
            max_calls=int(memory["max_calls"]),
            streams=streams,
            scratches=scratches,
            pools=take(len(streams)) if memory["pools"] else None,
            lanes=[dict(lane) for lane in memory["lanes"]],
            experts=[int(index) for index in memory["experts"]],
            host_scratch=host_scratch,
        )
        if memory["pools"]:
            for index in range(len(streams)):
                binding.peer_pools[index] = [
                    (int(ordinal), take(1)[0]) for ordinal in memory["peers"]
                ]
        pages = memory.get("pages")
        if pages is not None:
            (binding.unit_pool,) = take(1)
            names = [str(name) for name in pages["tables"]]
            binding.tables = dict(zip(names, take(len(names)), strict=True))
            binding.units = int(pages["units"])
        for index, lane in enumerate(binding.lanes):
            if "pages" in lane["resources"]:
                tokens, flags = take(2)
                binding.staging[index] = (tokens, flags)
        if memory["latents"]:
            first, second = take(2)
            binding.banks = (first, second)
        binding.reports = memory.get("reports")
        for group in memory["collectives"]:
            (handle,) = take(1)
            binding.collectives.append((dict(group), handle))
        if memory["prefetch"]:
            (binding.prefetch,) = take(1)
        if rest:
            raise ValueError("bind received more values than it names")
        return binding

    def lane_streams(
        self, lane: Mapping[str, Any]
    ) -> tuple[torch.cuda.Stream, ...]:
        """A lane's streams, by the indices it names; a host lane names none.

        Raises:
            ValueError: The lane names a host stream.
        """
        streams = []
        for index in lane["streams"]:
            stream = self.streams[int(index)]
            if stream is None:
                raise ValueError("a device lane names a stream of the host")
            streams.append(stream)
        return tuple(streams)

    def stream_groups(self, component: str, copy: int) -> dict[str, Any] | None:
        """A component's process group per communicator name on one copy."""
        groups = {
            str(name): handle
            for group, handle in self.collectives
            if group["component"] == component and int(group["copy"]) == copy
            for name in group["names"]
        }
        return groups or None


class _VideoLanes:
    """A video model's lanes over the video executor.

    One lane per component placed on the rank, each preparing its batches:
    the denoiser's lane steps trajectories in the worker's latent ledger,
    host components' lanes report their host tasks.
    """

    def __init__(self, loaded, settings: Mapping[str, Any]) -> None:
        from uniserve.executor.video import VideoExecutor

        self.video = VideoExecutor(loaded, settings)
        self.section = dict(settings["video"])
        self.placed = list(self.video.placed.values())

    def lanes(self) -> list[dict[str, Any]]:
        entries = []
        for placed in self.placed:
            resources = []
            if "video_denoiser" in placed.paths:
                resources.append("latents")
            if placed.host:
                resources.append("reports")
            entries.append(
                {
                    "component": placed.name,
                    "kinds": sorted(placed.kinds),
                    "prepares": True,
                    "commits": False,
                    "resources": resources,
                }
            )
        return entries

    def manifest(self) -> dict[str, Any]:
        return self.video.manifest()

    def bind(self, binding: _Binding, lanes: Sequence[int]) -> None:
        """Bind the video executor: each kind's stream, scratch and pool.

        ``lanes`` are the global indices of this executor's lanes, in the
        order :meth:`lanes` stated them.
        """
        streams: dict[str, torch.cuda.Stream] = {}
        scratches: dict[str, Any] = {}
        pools: dict[str, Any] = {}
        for placed, index in zip(self.placed, lanes, strict=True):
            lane = binding.lanes[index]
            if not lane["streams"]:
                continue
            stream_index = int(lane["streams"][0])
            (stream, *_) = binding.lane_streams(lane)
            for kind in lane["kinds"]:
                streams[str(kind)] = stream
                scratches[str(kind)] = binding.scratches[stream_index]
            if binding.pools is not None:
                for capability in placed.paths:
                    pools[capability] = binding.pools[stream_index]
        self.video.bind(
            {
                "slots": binding.slots,
                "reports": binding.reports,
                "ffmpeg": self.section["ffmpeg"],
                "transient_layouts": self.section["transient_layouts"],
            },
            streams,
            pools,
            scratches,
            binding.banks,
        )

    def warmup_phases(self) -> list[dict[str, Any]]:
        # One phase that reads no prefix cache: the video executor warms
        # and captures its components in one call.
        return [{"lane": 0, "lengths": None, "steps": []}]

    def warmup_phase(self, index: int, *pages: torch.Tensor) -> None:
        del index, pages
        self.video.warmup()

    def warmup_finish(self) -> None:
        pass

    def seal(self) -> None:
        pass

    def start(self, requests: list[Mapping[str, Any]]):
        return self.video.start(requests)

    def release(self, slots: list[int]) -> None:
        self.video.release(slots)

    def prepare(self, lane: int, step: Mapping[str, Any]):
        del lane
        return self.video.prepare(step)

    def execute(self, lane: int, step: str, tensors: Sequence[torch.Tensor]):
        del lane
        return [
            int(mode) for mode in self.video.execute(json.loads(step), tensors)
        ]

    def commit(self, lane: int) -> None:
        raise RuntimeError("video lanes do not commit")

    def kernels(self):
        return self.video.kernels()

    def close(self, aborted: bool) -> None:
        self.video.close(aborted)


class _TextLanes:
    """A causal language model's lanes over the text executor."""

    def __init__(self, loaded, settings: Mapping[str, Any]) -> None:
        from uniserve.executor.text import TextExecutor

        self.text = TextExecutor(loaded, settings)
        self._manifest: dict[str, Any] | None = None

    def lanes(self) -> list[dict[str, Any]]:
        return self._stated()["lanes"]

    def pages(self) -> dict[str, Any]:
        return self._stated()["pages"]

    def _stated(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = self.text.manifest()
        return self._manifest

    def bind(self, binding: _Binding, lanes: Sequence[int], union) -> None:
        from uniserve.executor.text import TextBinding

        self.text.bind(
            TextBinding(
                slots=binding.slots,
                units=binding.units,
                unit_pool=binding.unit_pool,
                tables=binding.tables,
                lanes=_lane_bindings(binding, lanes),
                prefetch=binding.prefetch,
            ),
            union,
        )

    def warmup_phases(self) -> list[dict[str, Any]]:
        return self.text.warmup_phases()

    def warmup_phase(self, index: int, *pages: torch.Tensor) -> None:
        self.text.warmup_phase(index, *pages)

    def warmup_finish(self) -> None:
        self.text.warmup_finish()

    def seal(self) -> None:
        self.text.seal()

    def start(self, requests: list[Mapping[str, Any]]):
        return self.text.start(requests)

    def release(self, slots: list[int]) -> None:
        self.text.release(slots)

    def prepare(self, lane: int, step: Mapping[str, Any]):
        raise RuntimeError("text lanes are not prepared")

    def execute(self, lane: int, step: str, tensors: Sequence[torch.Tensor]):
        return [int(self.text.execute(lane, step, *tensors))]

    def commit(self, lane: int) -> None:
        self.text.commit(lane)

    def reopen_experts(self) -> None:
        self.text.reopen_experts()

    def kernels(self):
        return self.text.kernels()

    def close(self, aborted: bool) -> None:
        self.text.close(aborted)


class _DiffusionLMLanes:
    """A diffusion language model's lanes over the diffusion LM executor.

    Its lanes compute over the prefix cache like the text lanes, bound
    through the same lane bindings; they neither prepare nor commit.
    """

    def __init__(self, loaded, settings: Mapping[str, Any]) -> None:
        from uniserve.executor.diffusion_lm import DiffusionLMExecutor

        self.diffusion_lm = DiffusionLMExecutor(loaded, settings)
        self._manifest: dict[str, Any] | None = None

    def lanes(self) -> list[dict[str, Any]]:
        return self._stated()["lanes"]

    def pages(self) -> dict[str, Any]:
        return self._stated()["pages"]

    def _stated(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = self.diffusion_lm.manifest()
        return self._manifest

    def bind(
        self, binding: _Binding, lanes: Sequence[int], union, joins: bool
    ) -> None:
        """Bind the lanes; ``joins`` when no text lane provides the union's
        joins, which then run through the first lane's runner.
        """  # noqa: D205
        from uniserve.executor.text import TextBinding

        self.diffusion_lm.bind(
            TextBinding(
                slots=binding.slots,
                units=binding.units,
                unit_pool=binding.unit_pool,
                tables=binding.tables,
                lanes=_lane_bindings(binding, lanes),
                prefetch=binding.prefetch,
            ),
            union,
            joins=joins,
        )

    def step_rows(self) -> int:
        return self.diffusion_lm.step_rows()

    def warmup_phases(self) -> list[dict[str, Any]]:
        return self.diffusion_lm.warmup_phases()

    def warmup_phase(self, index: int, *pages: torch.Tensor) -> None:
        self.diffusion_lm.warmup_phase(index, *pages)

    def warmup_finish(self) -> None:
        self.diffusion_lm.warmup_finish()

    def seal(self) -> None:
        self.diffusion_lm.seal()

    def start(self, requests: list[Mapping[str, Any]]):
        return self.diffusion_lm.start(requests)

    def release(self, slots: list[int]) -> None:
        self.diffusion_lm.release(slots)

    def prepare(self, lane: int, step: Mapping[str, Any]):
        raise RuntimeError("diffusion LM lanes are not prepared")

    def execute(self, lane: int, step: str, tensors: Sequence[torch.Tensor]):
        return [int(self.diffusion_lm.execute(lane, step, *tensors))]

    def commit(self, lane: int) -> None:
        raise RuntimeError("diffusion LM lanes do not commit")

    def reopen_experts(self) -> None:
        self.diffusion_lm.reopen_experts()

    def kernels(self):
        return self.diffusion_lm.kernels()

    def close(self, aborted: bool) -> None:
        self.diffusion_lm.close(aborted)


class _ImageLanes:
    """An image generation model's lanes over the image executor.

    One prepared lane per component the rank holds; the denoiser's lane
    steps trajectories in the worker's latent ledger and prefills its
    guidance prefixes through the text lane of its component.
    """

    def __init__(self, loaded, settings: Mapping[str, Any]) -> None:
        from uniserve.executor.image import ImageExecutor

        self.image = ImageExecutor(loaded, settings)

    def lanes(self) -> list[dict[str, Any]]:
        return self.image.lanes()

    def manifest(self) -> dict[str, Any]:
        return self.image.manifest()

    def bind(self, binding: _Binding, lanes: Sequence[int]) -> dict[str, Any]:
        """Bind the lanes; return the report the worker reads (``image``)."""
        return self.image.bind_from(binding, lanes)

    def warmup_phases(self) -> list[dict[str, Any]]:
        return self.image.warmup_phases()

    def warmup_phase(self, index: int, *pages: torch.Tensor) -> None:
        self.image.warmup_phase(index, *pages)

    def warmup_finish(self) -> None:
        self.image.warmup_finish()

    def seal(self) -> None:
        self.image.seal()

    def start(self, requests: list[Mapping[str, Any]]):
        return self.image.start(requests)

    def release(self, slots: list[int]) -> None:
        self.image.release(slots)

    def prepare(self, lane: int, step: Mapping[str, Any]):
        return self.image.prepare(lane, step)

    def execute(self, lane: int, step: str, tensors: Sequence[torch.Tensor]):
        return [
            int(mode)
            for mode in self.image.execute(lane, json.loads(step), tensors)
        ]

    def commit(self, lane: int) -> None:
        raise RuntimeError("image lanes do not commit")

    def kernels(self):
        return self.image.kernels()

    def close(self, aborted: bool) -> None:
        self.image.close(aborted)


class Executor:
    """The executor of one worker rank.

    Args:
        loaded: The loaded model.
        config: JSON of the rank's settings: ``rank``, ``device``,
            ``graphs`` (whether CUDA graphs serve), ``attention``,
            ``model_dtype``, ``max_model_len``, the placed ``components``
            (each with its ``name``, whether it is a ``host`` component, its
            declared ``calls`` (``capability`` and ``path``), the call
            ``kinds`` it serves, its ``ranks``, whether its media units are
            ``distributed`` and its ``units_per_rank``), and one section per
            role the rank serves, each ``null`` otherwise: ``video`` for a
            video model, ``pages`` for lanes over the prefix cache (batch
            bounds, graph buckets, cache groups, block diffusion, expert
            exchange) and ``experts`` for its member of an expert union.
    """

    def __init__(self, loaded, config: str) -> None:
        settings = json.loads(str(config))
        self._loaded = loaded
        self._settings = settings
        self._graphs = bool(settings["graphs"])
        self._served: list[Any] = []
        self._video = None
        self._text = None
        self._diffusion_lm = None
        self._image = None
        if settings.get("video") is not None:
            self._video = _VideoLanes(loaded, settings)
            self._served.append(self._video)
        # A lane executor whose components serve none of its call kinds on
        # this rank states no lane and takes no part in the rank.
        if settings.get("pages") is not None and _holds(settings, "causal_lm"):
            text = _TextLanes(loaded, settings)
            if text.lanes():
                self._text = text
                self._served.append(text)
        if settings.get("pages") is not None and _holds(
            settings, "diffusion_lm"
        ):
            diffusion_lm = _DiffusionLMLanes(loaded, settings)
            if diffusion_lm.lanes():
                self._diffusion_lm = diffusion_lm
                self._served.append(diffusion_lm)
        if settings.get("image") is not None:
            self._image = _ImageLanes(loaded, settings)
            self._served.append(self._image)
        # The member of an expert union, configured before planning.
        self._union: ExpertUnion | None = None
        # Each lane in manifest order: its executor and its index there.
        self._lanes: list[tuple[Any, int]] = []
        # Each warmup phase in order: its executor and its index there.
        self._phases: list[tuple[Any, int]] = []

    # The manifest and the expert union, before planning.

    def manifest(self) -> str:
        """JSON of every lane and what capacity planning reads."""
        lanes: list[dict[str, Any]] = []
        self._lanes = []
        for served in self._served:
            for local, entry in enumerate(served.lanes()):
                lanes.append(entry)
                self._lanes.append((served, local))
        manifest: dict[str, Any] = {"lanes": lanes}
        if self._video is not None:
            manifest["video"] = self._video.manifest()
        if self._image is not None:
            manifest["image"] = self._image.manifest()
        stated = [
            served.pages()
            for served in (self._text, self._diffusion_lm)
            if served is not None
        ]
        if stated:
            # One call column layout; the device bytes beside the cache
            # pool add up over the executors.
            fields = stated[0]["call_fields"]
            if any(pages["call_fields"] != fields for pages in stated):
                raise AssertionError(
                    "the lanes over the prefix cache disagree on call fields"
                )
            manifest["pages"] = {
                "call_fields": fields,
                "fixed_bytes": sum(int(p["fixed_bytes"]) for p in stated),
                "slot_bytes": sum(int(p["slot_bytes"]) for p in stated),
            }
        return json.dumps(manifest)

    def expert_storage_bytes(self, *host: Any) -> int:
        """Configure the expert union's agreement; peer storage bytes.

        Collective over the union. ``host`` holds the worker's host
        functions over it (``rank, size, all_gather, broadcast, barrier,
        split, all_gather_fixed``). Returns the bytes of peer-mapped storage
        each microbatch's exchange needs (zero for none), which the worker
        maps before :meth:`connect_experts`.
        """
        from uniserve.executor.experts import ExpertUnion

        settings = self._settings["experts"]
        self._union = ExpertUnion.configure(
            self._loaded,
            settings,
            host,
            max_tokens=_expert_tokens(self._settings),
            graphs=self._graphs,
        )
        return self._union.storage_bytes()

    def connect_experts(
        self, memory: str, *storages: torch.Tensor | None
    ) -> None:
        """Create the expert union's exchanges, collectively.

        ``memory`` is JSON ``{"microbatches": [{stream, device, pointers}]}``
        and ``storages`` each microbatch's peer-mapped storage or ``None``.
        """
        self._require_union().connect(
            json.loads(str(memory))["microbatches"], list(storages)
        )

    # Binding, warmup and seal.

    def bind(self, memory: str, *values: Any) -> str:
        """Construct the runners on the storage the worker granted.

        ``memory`` is JSON: ``slots``, ``max_calls``, the ``streams``
        (``[handle, ordinal]`` each), whether graph ``pools`` follow and
        the ``peers`` devices they follow on, each lane's ``component``,
        ``kinds``, ``streams`` and ``resources``, the ``experts`` microbatch
        stream indices, the prefix cache's ``pages`` (``units`` and table
        names) or ``null``, whether ``latents`` banks follow, the
        ``reports`` pipe descriptor, the stream groups' ``collectives`` and
        whether a ``prefetch`` object follows. Returns JSON
        ``{"diffusion_lm_step_rows": n, "image": report}``: the rows of one
        diffusion LM sampler chunk, and an image model's report (its warmup
        shapes, captured sizes, guidance tokens, capacity and capture-order
        digests) or ``null``.
        """
        binding = _Binding.parse(json.loads(str(memory)), values)
        if len(binding.lanes) != len(self._lanes):
            raise ValueError("bind names other lanes than the manifest")
        owned: dict[int, list[int]] = {}
        for index, (served, _) in enumerate(self._lanes):
            owned.setdefault(id(served), []).append(index)
        if self._video is not None:
            self._video.bind(binding, owned[id(self._video)])
        if self._text is not None:
            self._text.bind(binding, owned[id(self._text)], self._union)
        step_rows = 0
        if self._diffusion_lm is not None:
            self._diffusion_lm.bind(
                binding,
                owned[id(self._diffusion_lm)],
                self._union,
                joins=self._text is None,
            )
            step_rows = self._diffusion_lm.step_rows()
        image = None
        if self._image is not None:
            image = self._image.bind(binding, owned[id(self._image)])
            if self._text is not None:
                # Media calls of the text lanes ask the image half for its
                # facts and denoiser; the denoiser prefills its guidance
                # prefixes through its component's text lane.
                text = self._text.text
                text.attach_image(
                    self._image.image.facts(), self._image.image.denoiser()
                )
                self._image.image.attach_text(
                    {
                        str(entry["component"]): text.lanes[local]
                        for local, entry in enumerate(self._text.lanes())
                    }
                )
        if self._union is not None and not self._lanes:
            # An expert-only rank joins through layers of its own, one per
            # microbatch stream.
            indices = binding.experts
            self._union.bind_experts(
                self._loaded,
                [binding.scratches[index] for index in indices],
                [
                    None if binding.pools is None else binding.pools[index]
                    for index in indices
                ],
            )
        return json.dumps(
            {"diffusion_lm_step_rows": int(step_rows), "image": image}
        )

    def warmup_phases(self) -> str:
        """JSON of every warmup phase in order.

        Each: the ``lane`` it runs on, the scratch tables' ``lengths`` of
        slots ``1..`` (``null`` for a phase that reads no prefix cache) and
        each step's attention rows ``[slot, prefix, query]``.
        """
        phases = []
        self._phases = []
        for served in self._served:
            first = next(
                index
                for index, (owner, _) in enumerate(self._lanes)
                if owner is served
            )
            for local, phase in enumerate(served.warmup_phases()):
                phases.append({**phase, "lane": first + int(phase["lane"])})
                self._phases.append((served, local))
        return json.dumps(phases)

    def warmup_phase(self, index: int, *pages: torch.Tensor) -> None:
        """Run warmup phase ``index`` with each step's page columns."""
        served, local = self._phases[int(index)]
        served.warmup_phase(local, *pages)

    def warmup_finish(self) -> None:
        """End warmup: each executor's last captures, then the union's.

        The union's dedicated expert ranks follow the announced steps until
        the sources announce zero, then every rank captures its joins.
        """
        for served in self._served:
            served.warmup_finish()
        if self._union is not None:
            self._union.finish_warmup()

    def seal(self) -> None:
        """Refuse captures from now on: serving replays only."""
        for served in self._served:
            served.seal()

    # Requests and batches.

    def start(self, requests: str) -> str:
        """Admit the requests a batch started (JSON ``NewRequest`` list).

        Returns JSON ``[[request_key, reason], ...]`` of those refused.
        """
        started = json.loads(str(requests))
        refused: list[tuple[Any, str]] = []
        for served in self._served:
            refused.extend(served.start(started) or ())
        return json.dumps(
            [
                [
                    {
                        "engine_id": key[0],
                        "request_id": key[1],
                        "request_epoch": key[2],
                    },
                    reason,
                ]
                for key, reason in refused
            ]
        )

    def release(self, slots: Sequence[int]) -> None:
        """Forget the retired request slots."""
        retired = [int(slot) for slot in slots]
        for served in self._served:
            served.release(retired)

    def prepare(self, lane: int, step: str) -> str:
        """JSON of what each call of lane ``lane``'s batch reads and writes."""
        served, local = self._lanes[int(lane)]
        return json.dumps(served.prepare(local, json.loads(str(step))))

    def execute(
        self, lane: int, step: str, *tensors: torch.Tensor
    ) -> list[int]:
        """Run lane ``lane``'s batch on the tensors the worker granted.

        A prepared lane returns how each call's forward ran (0 eager, 1
        replay); a lane over the prefix cache returns the record
        words it wrote.
        """
        served, local = self._lanes[int(lane)]
        return served.execute(local, str(step), tensors)

    def commit(self, lane: int) -> None:
        """Advance lane ``lane``'s slot state past its last batch."""
        served, local = self._lanes[int(lane)]
        served.commit(local)

    # The expert union outside the rank's own steps.

    def join_expert_step(self, leaving: bool) -> list[int]:
        """Join the union's next step; whether one ran and all left."""
        ran, released = self._require_union().idle(bool(leaving))
        return [int(ran), int(released)]

    def park_experts(self) -> list[int]:
        """Release the expert workspaces while a standby parks.

        Collective over the union: those that keep their addresses are
        suspended, then those that cannot are freed. Returns whether any was
        suspended and whether any was freed.
        """
        union = self._require_union()
        suspended = union.suspend()
        released = union.release()
        return [int(suspended), int(released)]

    def promote_experts(self) -> None:
        """Map the suspended expert workspaces back at their addresses."""
        self._require_union().restore()

    def rebuild_experts(self) -> None:
        """Create freed expert workspaces again and drop graphs over them.

        The warmup that follows captures the expert steps and joins anew.
        """
        self._require_union().rebuild()
        for served in self._served:
            reopen = getattr(served, "reopen_experts", None)
            if reopen is not None:
                reopen()

    # Kernels and close.

    def kernels(self) -> str:
        """JSON ``[[label, [kernel record JSON, ...]], ...]``."""
        listed = []
        for served in self._served:
            listed.extend(served.kernels())
        return json.dumps(listed)

    def close(self, aborted: bool) -> None:
        """Release every runner, then the expert union.

        The worker drained the streams. An aborted close never waits on
        peers.
        """
        failure: BaseException | None = None
        for served in reversed(self._served):
            try:
                served.close(bool(aborted))
            except Exception as error:  # noqa: BLE001 - close every part
                failure = failure or error
        if self._union is not None:
            try:
                self._union.close(bool(aborted))
            except Exception as error:  # noqa: BLE001 - close every part
                failure = failure or error
        if failure is not None:
            raise failure

    def _require_union(self):
        if self._union is None:
            raise RuntimeError("the rank is no member of an expert union")
        return self._union


def _lane_bindings(binding: _Binding, lanes: Sequence[int]) -> tuple:
    """The ``LaneBinding`` of each of an executor's lanes over the prefix
    cache, indexed locally in the order of ``lanes`` (global indices).
    """  # noqa: D205
    from uniserve.executor.text import LaneBinding

    entries = []
    for local, index in enumerate(lanes):
        lane = binding.lanes[index]
        component = str(lane["component"])
        indices = [int(stream) for stream in lane["streams"]]
        tokens, flags = binding.staging[index]
        if not indices:
            # A lane on the host runs one microbatch, without a stream or a
            # graph pool, on the rank's host scratch.
            entries.append(
                LaneBinding(
                    index=local,
                    component=component,
                    streams=(None,),
                    scratches=(binding.host_scratch,),
                    pools=None,
                    relay_tokens=tokens,
                    relay_flags=flags,
                    collectives=(binding.stream_groups(component, 0),),
                )
            )
            continue
        entries.append(
            LaneBinding(
                index=local,
                component=component,
                streams=binding.lane_streams(lane),
                scratches=tuple(binding.scratches[i] for i in indices),
                pools=None
                if binding.pools is None
                else tuple(binding.pools[i] for i in indices),
                relay_tokens=tokens,
                relay_flags=flags,
                collectives=tuple(
                    binding.stream_groups(component, copy)
                    for copy in range(len(indices))
                ),
            )
        )
    return tuple(entries)


def _holds(settings: Mapping[str, Any], capability: str) -> bool:
    """Whether a device component on this rank declares ``capability``."""
    return any(
        not component["host"]
        and any(call["capability"] == capability for call in component["calls"])
        for component in settings["components"]
    )


def _expert_tokens(settings: Mapping[str, Any]) -> int:
    """Tokens one forward of this rank sends to each expert layer at most.

    The batch token bound or, under graphs, the largest decode graph bucket
    within the batch bound, whichever is larger. A rank without lanes over
    the prefix cache (an expert-only rank) takes the sources' bound.
    """
    pages = settings.get("pages")
    if pages is None:
        return 1
    tokens = int(pages["max_batch_tokens"])
    graphs = bool(settings["graphs"]) and pages["graph_policy"] != "off"
    if not graphs:
        return tokens
    rows = int(pages["max_batch_calls"])
    return max(
        [tokens]
        + [
            int(size)
            for size in pages["decode_graph_batch_sizes"]
            if 0 < int(size) <= rows
        ]
    )
