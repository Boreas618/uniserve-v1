"""A video model's denoiser on one stream.

:class:`Denoiser` holds what a denoising rank keeps of every request: its
slot state (state tables and retained conditioning on the device, the host
sources a request is initialized from), the layouts the denoiser evaluates
and their captured steps, and the text conditioner of the pipeline stage
that holds it.

A request's samples live in the latent banks the rank's memory owns: two
banks of pages, the request slot owning a run of consecutive pages after
page 0 (:class:`~uniserve.executor.video.plan.SamplePages`). The trajectory
ledger, which bank holds a request's committed step, is the worker's: every
call names the bank it reads and the bank it writes. A solver step gathers
the read bank's pages into the sample scratch, copies the request's slot
rows into the fixed state buffers, writes the evaluation index, replays the
layout's captured step (or runs it eagerly) and scatters the scratch into
the written bank.

Layouts fix the shapes a step evaluates. Startup layouts are prepared,
warmed and captured before serving and stay resident; a request whose
layout was not prepared at startup prepares a transient layout, evaluated
eagerly, which the least recently used unused transient layout makes room
for.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from uniserve.cuda_graph import BreakableCUDAGraph, capture
from uniserve.executor.video.plan import (
    Canvas,
    ConditionShape,
    Layout,
    LayoutFields,
    MediaPlan,
    SampleLayout,
    SamplePages,
    SampleRows,
    StateEnvelope,
    VideoModel,
    VideoSize,
    rows_per_frame,
)
from uniserve.models import ConditionRole
from uniserve.runner.denoiser import DenoiserRunner
from uniserve.runner.encoder import EncoderRunner

__all__ = [
    "EAGER",
    "REPLAY",
    "Conditioning",
    "Denoiser",
    "close_all",
    "keyframe_sets",
]

# How a forward ran, as the worker's batch statistics count it (2, a
# capture, is never reported: startup captures before serving).
EAGER = 0
REPLAY = 1


def size_values(size: VideoSize) -> tuple[list[int], list[int], list[int]]:
    """A size's flat descriptors (``uniserve.runner.denoiser``).

    ``[frames, width, height, tokens]``, the conditions and the vision
    spans.
    """
    roles = {
        ConditionRole.FIRST_FRAME: 0,
        ConditionRole.LAST_FRAME: 1,
        ConditionRole.REFERENCE: 2,
    }
    conditions = []
    for shape in size.conditions:
        frames, width, height = 0, 0, 0
        if shape.pixels is not None:
            frames, canvas = shape.pixels
            width, height = canvas.width, canvas.height
        conditions.extend(
            [roles[shape.role], frames, width, height, shape.audio_samples]
        )
    spans = [value for span in size.spans for value in span]
    return (
        [size.frames, size.canvas.width, size.canvas.height, size.tokens],
        conditions,
        spans,
    )


@dataclass(frozen=True)
class Conditioning:
    """The text conditioner a denoising rank holds."""

    # The conditioner module's path in the model.
    path: str
    # Prompt capacities the conditioner serves.
    capacities: tuple[int, ...]
    # Columns and dtype of the prompt features it refines.
    width: int
    dtype: torch.dtype


@dataclass
class _Prepared:
    """One prepared layout."""

    key: Layout
    # The layout's constants, as the runner prepared them.
    handle: Any
    fields: LayoutFields
    samples: SampleLayout
    # The tensors a step takes for the layout, every state field in state
    # order: a modality's local sample in the scratch, a table's fixed state
    # buffer.
    tensors: list[torch.Tensor]
    # State tables (and the retained conditioning where the conditioner
    # retains it) copied from the slot's rows before each step:
    # ``(field, elements)``.
    copies: list[tuple[str, int]]
    startup: bool
    warmed: bool = False
    # Live requests evaluating in the layout.
    references: int = 0


class _Conditioner:
    """The text conditioner of a pipeline stage that holds it.

    One runner per prompt capacity over the same input and output.
    """

    def __init__(
        self,
        loaded,
        conditioning: Conditioning,
        stream: torch.cuda.Stream,
        attention: str,
        scratch,
        output_width: int,
        device: torch.device,
    ) -> None:
        rows = max(conditioning.capacities)
        # ``[capacity, width]`` prompt features, zero past the prompt.
        self.input = torch.zeros(
            (rows, conditioning.width), dtype=conditioning.dtype, device=device
        )
        # ``[1]`` int32 prompt length.
        self.lengths = torch.zeros(1, dtype=torch.int32, device=device)
        # ``[capacity, conditioning width]`` BF16 refined rows.
        self.output = torch.zeros(
            (rows, output_width), dtype=torch.bfloat16, device=device
        )
        # The largest capacity is prepared first, so it sizes the workspace
        # the capacities share.
        self.runners: dict[int, EncoderRunner] = {}
        for capacity in sorted(conditioning.capacities, reverse=True):
            self.runners[capacity] = EncoderRunner(
                loaded,
                conditioning.path,
                stream.cuda_stream,
                attention,
                scratch,
                [capacity, conditioning.width],
            )
        self.graphs: dict[int, BreakableCUDAGraph] = {}

    def encode(self, capacity: int) -> None:
        self.runners[capacity].encode(
            self.input[:capacity], self.lengths, self.output[:capacity]
        )


class Denoiser:
    """A video model's denoiser.

    It holds the slot state, the latent banks, the layouts and their
    captured steps.

    Args:
        loaded: The loaded model.
        component: The denoiser's component.
        model: The model's facts and sizing rules.
        plan: What the rank admits and prepares.
        envelope: The slot state every layout fits.
        pages: Each request slot's latent pages.
        slots: Request slots; slot rows ``1..=slots`` belong to requests.
        stream: The denoiser's lane stream.
        attention: The attention backend the model loaded with.
        scratch: The stream's shared work areas.
        conditioning: The text conditioner this rank holds, if any.
        condition_latents: Under CUDA graphs, ``(width, dtype)`` of the
            condition latent rows the captured condition encodings read;
            ``None`` without conditions.
        pool: The graph pool captures allocate in; ``None`` serves every
            step eagerly.
        transient: The transient layouts kept at once.
        banks: The latent banks ``[pages, page_units]`` float32, bank 0 and
            bank 1.
    """

    def __init__(
        self,
        *,
        loaded,
        component: str,
        model: VideoModel,
        plan: MediaPlan,
        envelope: StateEnvelope,
        pages: SamplePages,
        slots: int,
        stream: torch.cuda.Stream,
        attention: str,
        scratch,
        conditioning: Conditioning | None,
        condition_latents: tuple[int, torch.dtype] | None,
        pool: torch.cuda.MemPool | None,
        transient: int,
        banks: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self.component = component
        self.model = model
        self.plan = plan
        self.envelope = envelope
        self.pages = pages
        self.stream = stream
        self.pool = pool
        self.banks = banks
        self.transient_capacity = max(transient, 1)
        device = stream.device
        self.device = device
        self.runner = DenoiserRunner(
            loaded,
            component,
            stream.cuda_stream,
            attention,
            scratch,
            list(plan.maximum),
        )
        with torch.cuda.device(device):
            # FP32 scratch one request's local samples occupy during a step.
            self.samples = torch.zeros(
                envelope.samples, dtype=torch.float32, device=device
            )
            # The ``[1]`` int64 evaluation index.
            self.step = torch.zeros(1, dtype=torch.int64, device=device)
            # Fixed state buffers, one per device field.
            self.state = {
                name: torch.zeros(
                    field.elements, dtype=field.dtype, device=device
                )
                for name, field in envelope.device.items()
            }
            # Slot rows: the padding row first, then one per request slot.
            self.slot_state = {
                name: torch.zeros(
                    (slots + 1, field.elements),
                    dtype=field.dtype,
                    device=device,
                )
                for name, field in envelope.device.items()
            }
        self.sources = {
            name: torch.zeros(
                (slots + 1, field.elements), dtype=field.dtype, pin_memory=True
            )
            for name, field in envelope.host.items()
        }
        self.conditioner = (
            None
            if conditioning is None
            else _Conditioner(
                loaded,
                conditioning,
                stream,
                attention,
                scratch,
                model.text_condition_width,
                device,
            )
        )
        self.layouts: dict[Layout, _Prepared] = {}
        # Transient layouts, least recently used first.
        self.transient: list[Layout] = []
        self.graphs: dict[Layout, BreakableCUDAGraph] = {}
        self.sealed = False
        # Under CUDA graphs, the captured condition encodings and the
        # buffers they read and write: a request's condition latents are
        # copied into ``condition_rows`` back to back and its draws into
        # ``condition_noise``, the graph of its layout and conditions
        # replays into ``condition_out``, and its condition rows are copied
        # into its slot's conditioning.
        self.condition_graphs: dict[
            tuple[Layout, tuple[ConditionShape, ...]], BreakableCUDAGraph
        ] = {}
        self.condition_rows: torch.Tensor | None = None
        self.condition_noise: torch.Tensor | None = None
        self.condition_out: torch.Tensor | None = None
        if (
            pool is not None
            and condition_latents is not None
            and plan.condition_rows > 0
        ):
            width, dtype = condition_latents
            with torch.cuda.device(device):
                self.condition_rows = torch.zeros(
                    (max(plan.condition_rows, 1), width),
                    dtype=dtype,
                    device=device,
                )
                self.condition_noise = torch.zeros(
                    envelope.host["condition_noise"].elements,
                    dtype=torch.float32,
                    device=device,
                )
                self.condition_out = torch.zeros(
                    envelope.device["text_condition"].elements,
                    dtype=torch.bfloat16,
                    device=device,
                )

    # Layouts.

    def _install(self, layout: Layout, startup: bool) -> _Prepared:
        fields = self.model.fields(layout)
        samples = SampleLayout.of(fields, self.pages.page_units)
        if samples.pages > self.pages.pages:
            raise ValueError(
                "a video layout's samples exceed the request's latent pages"
            )
        copies = []
        for field in fields.state:
            if field.sample:
                continue
            held = self.envelope.device.get(field.name)
            if held is None or field.elements > held.elements:
                raise ValueError(
                    f"layout state field {field.name} exceeds the slot state"
                )
            copies.append((field.name, field.elements))
        # A step reads the slot's retained conditioning only where the
        # conditioner retains it; on a later pipeline stage the step's
        # conditioning buffer keeps the zeros its warmup wrote, and its
        # forward reads none of it.
        if self.conditioner is not None:
            rows, width = fields.text_condition
            copies.append(("text_condition", rows * width))
        tensors = []
        modality = 0
        for field in fields.state:
            if field.sample:
                offset = samples.offsets[modality]
                tensors.append(
                    self.samples[offset : offset + samples.elements[modality]]
                )
                modality += 1
            else:
                tensors.append(self.state[field.name])
        prepared = _Prepared(
            key=layout,
            handle=self.runner.prepare_layout(list(layout)),
            fields=fields,
            samples=samples,
            tensors=tensors,
            copies=copies,
            startup=startup,
        )
        self.layouts[layout] = prepared
        return prepared

    def prepare_startup(self, layout: Layout) -> _Prepared:
        """Prepare a startup layout once; startup layouts stay resident."""
        prepared = self.layouts.get(layout)
        if prepared is None:
            prepared = self._install(layout, True)
        return prepared

    def acquire(self, layout: Layout) -> None:
        """Hold layout ``layout`` for one more request.

        A layout not prepared yet is prepared as a transient layout. Unused
        transient layouts beyond the capacity give way; their constants
        outlive their last step because the stream is drained first.
        """
        prepared = self.layouts.get(layout)
        if prepared is None:
            evicted = []
            while len(self.transient) >= self.transient_capacity:
                unused = next(
                    (
                        key
                        for key in self.transient
                        if self.layouts[key].references == 0
                    ),
                    None,
                )
                if unused is None:
                    break
                self.transient.remove(unused)
                evicted.append(self.layouts.pop(unused))
            if evicted:
                self.stream.synchronize()
            del evicted
            prepared = self._install(layout, False)
            self.transient.append(layout)
        elif layout in self.transient:
            self.transient.remove(layout)
            self.transient.append(layout)
        prepared.references += 1

    def release(self, layout: Layout) -> None:
        """End a request's use of its layout."""
        prepared = self.layouts[layout]
        if prepared.references == 0:
            raise AssertionError("video layout reference underflow")
        prepared.references -= 1

    # Requests.

    def _slot_pages(self, slot: int) -> slice:
        pages = self.pages.for_slot(slot)
        return slice(pages.start, pages.stop)

    def prepare_latents(
        self, layout: Layout, slot: int, size: VideoSize, seed: int
    ) -> None:
        """Draw a request's noise and fill its slot's host sources.

        The noise is drawn from ``seed``; the sources are its layout's. Host
        work only.
        """
        prepared = self.layouts[layout]
        size_list, conditions, spans = size_values(size)
        noise = [
            self.sources[f"{name}_noise"][slot]
            for name, _ in prepared.fields.samples
        ]
        # The seed's bits; the runner reads them as unsigned.
        self.runner.draw_noise(
            seed,
            prepared.handle,
            size_list,
            conditions,
            spans,
            self.sources["condition_noise"][slot],
            *noise,
        )
        self.runner.prepare_latents(
            prepared.handle,
            size_list,
            conditions,
            spans,
            *noise,
            *(
                self.sources[f"{field.name}_source"][slot]
                for field in prepared.fields.state
            ),
        )

    def initialize(self, layout: Layout, slot: int, bank: int) -> None:
        """Copy a request's prepared host sources into its slot.

        Each modality's sample goes into ``bank`` of its pages and each table
        into its slot rows.
        """
        prepared = self.layouts[layout]
        pages = self.banks[bank][self._slot_pages(slot)].view(-1)
        with torch.cuda.stream(self.stream):
            for index, (name, _) in enumerate(prepared.fields.samples):
                offset = prepared.samples.offsets[index]
                count = prepared.samples.elements[index]
                pages[offset : offset + count].copy_(
                    self.sources[f"{name}_source"][slot][:count],
                    non_blocking=True,
                )
            for field in prepared.fields.state:
                if field.sample:
                    continue
                self.slot_state[field.name][slot][: field.elements].copy_(
                    self.sources[f"{field.name}_source"][slot][
                        : field.elements
                    ],
                    non_blocking=True,
                )

    @property
    def conditions(self) -> bool:
        """Whether this rank retains a request's conditioning.

        It does when it holds the text conditioner. A later pipeline stage
        receives the refined prompt among the preceding stage's hidden rows.
        """
        return self.conditioner is not None

    def condition(
        self, layout: Layout, slot: int, features: torch.Tensor
    ) -> int:
        """Refine a request's prompt features into its slot's conditioning.

        The rows of the layout's conditioning past the prompt become zero.
        Returns how the conditioner ran.
        """
        conditioner = self.conditioner
        if conditioner is None:
            raise AssertionError(
                f"{self.component}: a rank without the text conditioner "
                "refines no prompt"
            )
        prepared = self.layouts[layout]
        rows, width = prepared.fields.text_condition
        count = features.shape[0]
        capacity = next(
            (each for each in sorted(conditioner.runners) if each >= count),
            None,
        )
        if capacity is None:
            raise ValueError("prompt exceeds the configured text capacities")
        if (
            features.dtype != conditioner.input.dtype
            or features.shape[1:] != conditioner.input.shape[1:]
        ):
            raise ValueError(
                "prompt features disagree with the conditioner's input"
            )
        target = self.slot_state["text_condition"][slot][: rows * width].view(
            rows, width
        )
        with torch.cuda.stream(self.stream):
            conditioner.input[:count].copy_(features)
            conditioner.input[count:capacity].zero_()
            conditioner.lengths.fill_(count)
            mode = self._run_conditioner(capacity)
            target[:count].copy_(conditioner.output[:count])
            target[count:].zero_()
        return mode

    def _run_conditioner(self, capacity: int) -> int:
        conditioner = self.conditioner
        assert conditioner is not None
        graph = conditioner.graphs.get(capacity)
        if graph is not None:
            graph.replay()
            return REPLAY
        conditioner.encode(capacity)
        return EAGER

    def encode_conditions(
        self,
        layout: Layout,
        slot: int,
        size: VideoSize,
        latents: Sequence[Sequence[torch.Tensor]],
    ) -> int:
        """Write a request's encoded conditions into its slot's conditioning.

        They follow the prompt's rows. ``latents`` are the condition
        latents in request order, a condition's visual rows before its audio
        rows, each as the row tensors that together hold its rows in order.

        A condition set startup captured replays its encoding; any other is
        encoded eagerly.
        """
        prepared = self.layouts[layout]
        target = self.slot_state["text_condition"][slot]
        noise = self.sources["condition_noise"][slot]
        graph = self.condition_graphs.get((layout, size.conditions))
        with torch.cuda.stream(self.stream):
            if graph is not None:
                assert self.condition_rows is not None
                assert self.condition_noise is not None
                assert self.condition_out is not None
                flat = self.condition_rows.view(-1)
                offset = 0
                for part in (part for parts in latents for part in parts):
                    count = part.numel()
                    if offset + count > flat.numel():
                        raise ValueError(
                            "condition latents exceed the captured "
                            "encodings' rows"
                        )
                    flat[offset : offset + count].copy_(part.reshape(-1))
                    offset += count
                draws = prepared.fields.condition_noise
                self.condition_noise[:draws].copy_(
                    noise[:draws], non_blocking=True
                )
                graph.replay()
                # The conditions' rows follow the prompt's in the retained
                # conditioning; only they are copied, so the prompt's rows
                # the conditioner wrote stay.
                width = prepared.fields.text_condition[1]
                first = layout.text_rows * width
                stop = first + layout.condition_rows * width
                target[first:stop].copy_(self.condition_out[first:stop])
                return REPLAY
            size_list, conditions, spans = size_values(size)
            self.runner.encode_conditions(
                prepared.handle,
                size_list,
                conditions,
                spans,
                noise,
                target,
                [len(parts) for parts in latents],
                *(part for parts in latents for part in parts),
            )
        return EAGER

    def denoise(
        self, layout: Layout, slot: int, read: int, write: int, step: int
    ) -> int:
        """Advance a request's trajectory by one solver step.

        The step reads its samples in bank ``read`` and writes them to bank
        ``write``, at evaluation ``step``. Returns how the step ran.
        """
        prepared = self.layouts[layout]
        span = self._slot_pages(slot)
        count = prepared.samples.pages * self.pages.page_units
        with torch.cuda.stream(self.stream):
            self.samples[:count].copy_(
                self.banks[read][span].reshape(-1)[:count]
            )
            for name, elements in prepared.copies:
                self.state[name][:elements].copy_(
                    self.slot_state[name][slot][:elements]
                )
            self.step.fill_(step)
            mode = self._run_step(prepared)
            self.banks[write][span].reshape(-1)[:count].copy_(
                self.samples[:count]
            )
        return mode

    def _run_step(self, prepared: _Prepared) -> int:
        graph = self.graphs.get(prepared.key)
        if graph is not None:
            graph.replay()
            return REPLAY
        if self.pool is not None and self.sealed and prepared.startup:
            raise AssertionError(
                f"{self.component}: startup layout {list(prepared.key)} has "
                "no captured step"
            )
        self._step(prepared)
        return EAGER

    def _step(self, prepared: _Prepared) -> None:
        self.runner.denoise(
            prepared.handle,
            self.step,
            self.state["text_condition"],
            *prepared.tensors,
        )

    def final_samples(self, layout: Layout) -> list[torch.Tensor | None]:
        """The final step's local sample of each modality, in modality order.

        Each is a ``[rows, width]`` view of the scratch, ``None`` for a
        modality this rank holds no rows of.
        """
        prepared = self.layouts[layout]
        result: list[torch.Tensor | None] = []
        for index, (_, rows) in enumerate(prepared.fields.samples):
            local = rows.stop - rows.start
            if local == 0:
                result.append(None)
                continue
            offset = prepared.samples.offsets[index]
            result.append(
                self.samples[offset : offset + local * rows.width].view(
                    local, rows.width
                )
            )
        return result

    def sample_rows(self, layout: Layout) -> list[SampleRows]:
        """The rows of each modality's complete sample in a layout."""
        return [rows for _, rows in self.layouts[layout].fields.samples]

    # Startup.

    def warm(self, layout: Layout, slot: int = 1) -> None:
        """Run one placeholder step of startup layout ``layout``.

        The step reads slot ``slot``'s host sources, filled before
        (:meth:`prepare_latents`), to prepare its kernels and plans before
        any capture, then waits for the stream. The scratch's values are
        discarded.
        """
        prepared = self.layouts[layout]
        with torch.cuda.stream(self.stream):
            for index, (name, _) in enumerate(prepared.fields.samples):
                offset = prepared.samples.offsets[index]
                count = prepared.samples.elements[index]
                self.samples[offset : offset + count].copy_(
                    self.sources[f"{name}_source"][slot][:count],
                    non_blocking=True,
                )
            for field in prepared.fields.state:
                if field.sample:
                    continue
                self.state[field.name][: field.elements].copy_(
                    self.sources[f"{field.name}_source"][slot][
                        : field.elements
                    ],
                    non_blocking=True,
                )
            self.state["text_condition"].zero_()
            self.step.zero_()
            self._step(prepared)
        self.stream.synchronize()
        prepared.warmed = True

    def capture(self, layout: Layout) -> bool:
        """Capture a warmed startup layout's step; whether it captured one."""
        if self.pool is None or layout in self.graphs:
            return False
        prepared = self.layouts[layout]
        if not prepared.startup or not prepared.warmed:
            raise AssertionError("only warmed startup layouts are captured")
        self.graphs[layout] = capture(
            lambda: self._step(prepared), stream=self.stream, pool=self.pool
        )
        return True

    def warm_conditioner(self, capture_graphs: bool) -> None:
        """Warm the conditioner at every capacity, largest first.

        It runs on zero features, then, with ``capture_graphs``, each
        capacity is captured.
        """
        conditioner = self.conditioner
        if conditioner is None:
            return
        capacities = sorted(conditioner.runners, reverse=True)
        with torch.cuda.stream(self.stream):
            conditioner.input.zero_()
            for capacity in capacities:
                conditioner.lengths.fill_(capacity)
                conditioner.encode(capacity)
        self.stream.synchronize()
        if not capture_graphs or self.pool is None:
            return
        for capacity in capacities:
            conditioner.graphs[capacity] = capture(
                functools.partial(conditioner.encode, capacity),
                stream=self.stream,
                pool=self.pool,
            )

    def capture_conditions(
        self, layout: Layout, conditions: tuple[ConditionShape, ...]
    ) -> bool:
        """Capture the encoding of keyframe ``conditions``.

        It encodes into the conditioning of startup layout ``layout``, behind
        its eager encoding, which warms it. Returns whether it captured one.
        """
        if (
            self.pool is None
            or self.condition_rows is None
            or self.condition_noise is None
            or self.condition_out is None
            or (layout, conditions) in self.condition_graphs
        ):
            return False
        prepared = self.layouts[layout]
        parts = []
        row = 0
        for condition in conditions:
            if (
                condition.pixels is None
                or condition.pixels[0] != 1
                or condition.audio_samples
            ):
                raise ValueError("only keyframe conditions are captured")
            # A keyframe's latent is one latent frame of its canvas.
            rows = rows_per_frame(condition.pixels[1])
            parts.append([self.condition_rows[row : row + rows]])
            row += rows
        size = VideoSize(layout.frames, layout.canvas, 1, conditions)
        size_list, condition_values, spans = size_values(size)
        noise = self.condition_noise
        out = self.condition_out

        def encode() -> None:
            self.runner.encode_conditions(
                prepared.handle,
                size_list,
                condition_values,
                spans,
                noise,
                out,
                [1] * len(parts),
                *(part for each in parts for part in each),
            )

        # The capture encodes zero latents with zero draws.
        with torch.cuda.stream(self.stream):
            self.condition_rows.zero_()
            noise.zero_()
            encode()
        self.condition_graphs[(layout, conditions)] = capture(
            encode, stream=self.stream, pool=self.pool
        )
        return True

    def seal(self) -> None:
        """Refuse eager steps of startup layouts.

        Serving replays what startup captured.
        """
        self.sealed = True

    def kernels(self) -> list[str]:
        """The kernels the denoiser's operators selected.

        The denoiser's, then the conditioner's at each capacity.
        """
        kernels = list(self.runner.kernels())
        if self.conditioner is not None:
            for runner in self.conditioner.runners.values():
                kernels.extend(runner.kernels())
        return kernels

    def close(self, aborted: bool) -> None:
        """Release graphs and the runners; the caller drained the stream.

        An aborted close keeps everything device work may still address
        until the process exits.
        """
        if aborted:
            return
        graphs = [*self.graphs.values(), *self.condition_graphs.values()]
        runners: list[Any] = [self.runner]
        if self.conditioner is not None:
            graphs.extend(self.conditioner.graphs.values())
            runners.extend(self.conditioner.runners.values())
            self.conditioner.graphs.clear()
        self.graphs.clear()
        self.condition_graphs.clear()
        close_all(graphs, runners)


def close_all(
    graphs: Iterable[BreakableCUDAGraph], runners: Iterable[Any]
) -> None:
    """Release ``graphs``, then close ``runners``, each in order.

    Every release and close is attempted before the first failure is
    raised. The caller drained the streams the graphs and runners use.
    """
    failure: BaseException | None = None
    for graph in list(graphs):
        try:
            graph.release()
        except Exception as error:
            failure = failure or error
    for runner in list(runners):
        try:
            runner.close(False)
        except Exception as error:
            failure = failure or error
    if failure is not None:
        raise failure


def keyframe_sets(canvas: Canvas) -> list[tuple[ConditionShape, ...]]:
    """Every keyframe set a request may bring on ``canvas``."""

    def keyframe(role: ConditionRole) -> ConditionShape:
        return ConditionShape(role, (1, canvas), 0)

    return [
        (keyframe(ConditionRole.FIRST_FRAME),),
        (keyframe(ConditionRole.LAST_FRAME),),
        (
            keyframe(ConditionRole.FIRST_FRAME),
            keyframe(ConditionRole.LAST_FRAME),
        ),
    ]
