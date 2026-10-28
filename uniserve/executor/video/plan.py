"""What a video rank admits and the layouts it prepares.

A video request's numerical size is its output frame count, canvas, prompt
length, conditions (in request order) and the vision spans of its presented
prompt. The denoiser evaluates it in a *layout*: a size whose shapes,
constants and captured graphs every request it holds shares
(``VideoDenoiser.layout_size``). :class:`MediaPlan` fixes, at startup, the
frame counts, canvases and prompt capacities a rank admits and the layouts
it prepares; :meth:`MediaPlan.admit` admits a sized request in the layout of
the smallest prepared prompt capacity that holds it.

The denoiser is the authority on the sizes it serves and on the rows a
request's conditions take (its ``make_size``). A request whose size the
denoiser or the plan does not serve is refused alone, for the reason the
refusal states.

A request's samples live in the latent pages the rank's storage owns
(:class:`SamplePages`); its state tables and retained conditioning in slot
rows sized for every layout the rank prepares (:class:`StateEnvelope`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import torch

from uniserve.distributed.distribution import Segment, Span
from uniserve.distributed.logical import Logical
from uniserve.errors import SetupError
from uniserve.loading.rank import capability
from uniserve.media import image, video
from uniserve.models import (
    AudioDecoder,
    Condition,
    ConditionRole,
    VideoDecoder,
    VideoDenoiser,
    VideoEncoder,
    VideoPostprocessor,
)

__all__ = [
    "CANVAS_MULTIPLE",
    "OUTSIDE_PLAN",
    "Admission",
    "Canvas",
    "ConditionShape",
    "Layout",
    "LayoutFields",
    "MediaPlan",
    "SampleLayout",
    "SamplePages",
    "SampleRows",
    "StateEnvelope",
    "VideoModel",
    "VideoSize",
    "admitted_canvases",
    "decode_conditioning",
    "prompt_capacities",
    "rows_per_frame",
    "vision_spans",
]

# The refusal of a request whose frame count, canvas or prompt lies outside
# what the rank admits.
OUTSIDE_PLAN = (
    "media input exceeds the worker frame, canvas or conditioning capacity"
)

# The first default text capacity, in prompt tokens: a short prompt pads to
# it alone.
TEXT_CAPACITY_FIRST = 1024

# The step, in prompt tokens, of the default text capacities above the
# first. A request's layout holds at most one step of padding rows beyond
# its prompt's tiles.
TEXT_CAPACITY_STEP = 2048

# A request's samples occupy at most this many pages, which bounds the
# copies a step's gather and scatter make.
REQUEST_PAGES = 8

# Each modality's sample starts on this element boundary.
SAMPLE_ALIGNMENT = 256

# Bytes of one sample value in the latent pages: the banks are float32.
_SAMPLE_BYTES = 4

# Pixels per side of a latent patch: a canvas side is a multiple of it, and
# one latent frame holds one denoiser row per patch
# (``models::video::plan::CANVAS_MULTIPLE``).
CANVAS_MULTIPLE = 32

_ROLES = {
    "first_frame": ConditionRole.FIRST_FRAME,
    "last_frame": ConditionRole.LAST_FRAME,
    "reference": ConditionRole.REFERENCE,
}


def _aligned(elements: int) -> int:
    return -(-elements // SAMPLE_ALIGNMENT) * SAMPLE_ALIGNMENT


def _elements(shape: Sequence[int]) -> int:
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def requested_frames(seconds: float, frame_rate: int) -> int:
    """Frames of ``seconds`` of video at ``frame_rate``.

    Rounded half to even, as the reference implementations round with
    Python's ``round``.
    """
    return round(seconds * frame_rate)


class Canvas(NamedTuple):
    """A generated raster."""

    width: int
    height: int


def rows_per_frame(canvas: Canvas) -> int:
    """Denoiser rows of one latent frame of ``canvas``: one per patch."""
    return (canvas.width // CANVAS_MULTIPLE) * (
        canvas.height // CANVAS_MULTIPLE
    )


class Layout(NamedTuple):
    """A layout's extents.

    ``[frames, width, height, text_rows, condition_rows]``.
    """

    frames: int
    width: int
    height: int
    text_rows: int
    condition_rows: int

    @property
    def canvas(self) -> Canvas:
        return Canvas(self.width, self.height)


@dataclass(frozen=True)
class ConditionShape:
    """A condition as the denoiser sizes it."""

    role: ConditionRole
    # Encoded pixel frames and raster; ``None`` for audio alone.
    pixels: tuple[int, Canvas] | None
    # Model-rate audio samples; zero without a track.
    audio_samples: int

    @classmethod
    def of(cls, condition: Mapping[str, Any]) -> ConditionShape:
        """The shape of one decoded ``VideoCondition``."""
        media = condition["media"]
        pixels = None
        audio = None
        if "image" in media:
            size = media["image"]["size"]
            pixels = (1, Canvas(int(size["width"]), int(size["height"])))
        elif "video" in media:
            clip = media["video"]["clip"]
            canvas = clip["canvas"]
            pixels = (
                int(clip["vae_frames"]),
                Canvas(int(canvas["width"]), int(canvas["height"])),
            )
            audio = media["video"].get("soundtrack")
        else:
            audio = media["audio"]
        return cls(
            _ROLES[condition["role"]],
            pixels,
            0 if audio is None else int(audio["samples"]),
        )

    def condition(self) -> Condition:
        """The denoiser's condition of this shape."""
        pixels = None
        if self.pixels is not None:
            frames, canvas = self.pixels
            pixels = video.Config(
                frames, image.Config(canvas.height, canvas.width)
            )
        return Condition(self.role, pixels, self.audio_samples)


@dataclass(frozen=True)
class VideoSize:
    """A request's exact numerical size."""

    frames: int
    canvas: Canvas
    # Presented prompt tokens.
    tokens: int
    conditions: tuple[ConditionShape, ...] = ()
    # ``(start, stop)`` token ranges of the prompt's vision blocks.
    spans: tuple[tuple[int, int], ...] = ()


def vision_spans(tags: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """The vision blocks of a presented prompt.

    They are the maximal runs of tag 0 in its token tags, as ``(start,
    stop)`` token ranges.
    """
    spans = []
    start = None
    for index, tag in enumerate(tags):
        if tag == 0 and start is None:
            start = index
        elif tag != 0 and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(tags)))
    return tuple(spans)


@dataclass(frozen=True)
class StateField:
    """One state field of a layout on this rank."""

    name: str
    dtype: torch.dtype
    # Elements of the field on this rank.
    elements: int
    # A sample modality, stored in latent pages rather than slot rows.
    sample: bool

    @property
    def bytes(self) -> int:
        return self.elements * self.dtype.itemsize


class SampleRows(NamedTuple):
    """A modality's sample in one layout."""

    # Rows and width of the complete sample across all ranks.
    rows: int
    width: int
    # This rank's rows ``[start, stop)``.
    start: int
    stop: int


@dataclass(frozen=True)
class LayoutFields:
    """The request state of one layout on this rank."""

    state: tuple[StateField, ...]
    # Sample rows by modality, in modality order.
    samples: tuple[tuple[str, SampleRows], ...]
    # Elements of each modality's native draw, in modality order.
    noise: tuple[tuple[str, int], ...]
    # FP32 elements of the condition draws.
    condition_noise: int
    # ``(rows, width)`` of the retained conditioning.
    text_condition: tuple[int, int]
    # Elements of the layout's workspace buffers.
    workspace: int

    def storage(self) -> int:
        """Elements of the layout's state and workspace on this rank.

        They are what the layout takes of the storage every layout shares.
        """
        return sum(field.elements for field in self.state) + self.workspace


class VideoModel:
    """A video model's facts and sizing rules.

    Admission and capacity planning size a request before any runner of
    its layout exists, so the facts and sizing rules are read from the
    loaded modules here rather than from a runner.
    """

    def __init__(self, loaded, component: str) -> None:
        self.loaded = loaded
        self.component = str(component)
        self.denoiser: VideoDenoiser = _denoiser(loaded, self.component)
        self.decoder: VideoDecoder = _require(loaded, VideoDecoder)
        self.audio: AudioDecoder = _require(loaded, AudioDecoder)
        self.clock: VideoPostprocessor = _require(loaded, VideoPostprocessor)
        denoiser = self.denoiser
        self.modalities: tuple[str, ...] = tuple(denoiser.modalities)
        self.num_steps = int(denoiser.num_steps)
        self.tasks: tuple[str, ...] = tuple(denoiser.tasks)
        self.text_only = bool(denoiser.text_only)
        self.text_condition_width = int(denoiser.text_condition_width)
        self.sample_dtype: torch.dtype = denoiser.prediction_dtype
        self.frame_rate = int(self.clock.frame_rate)
        self.sample_rate = int(self.audio.sample_rate)
        # Canvases the denoiser offers, each named by the ``(short_edge,
        # aspect_ratio)`` a request selects it with.
        self.canvases: tuple[tuple[int, str, Canvas], ...] = tuple(
            (
                int(short_edge),
                str(aspect_ratio),
                Canvas(raster.width, raster.height),
            )
            for (short_edge, aspect_ratio), raster in denoiser.canvases.items()
        )

    def facts(self) -> dict[str, Any]:
        """What the engine is told about the denoiser.

        ``models::video::DenoiserInfo`` without its canvases.
        """
        denoiser = self.denoiser
        shifts = denoiser.schedule_shifts
        limits = denoiser.max_references
        fewest = int(denoiser.legal_frame_count(1))
        clip = int(denoiser.legal_frame_count(fewest + 1)) - fewest
        encoder = _require(self.loaded, VideoEncoder)
        whole, *_, tail = encoder.latent_slices(fewest)
        return {
            "tasks": list(self.tasks),
            # Sigma points of the fixed schedule, the clean endpoint
            # included: one more than the network evaluations.
            "schedule_points": self.num_steps + 1,
            "video_shift": float(shifts["video"]),
            "audio_shift": float(shifts["audio"]),
            "frame_rate": self.frame_rate,
            "audio_sample_rate": self.sample_rate,
            "frames": {
                "min_frames": fewest,
                "clip_frames": clip,
                "clip_latent_frames": whole.stop - whole.start,
                "tail_latent_frames": tail.stop - tail.start,
            },
            "references": {
                "images": int(limits.images),
                "videos": int(limits.videos),
                "audios": int(limits.audios),
                "total": int(limits.total),
                "image_short_edge": int(limits.image_short_edge),
            },
        }

    # Sizes and layouts.

    def size(self, size: VideoSize):
        """The denoiser's size of ``size``.

        Raises:
            ValueError: The denoiser does not serve the size.
        """
        return self.denoiser.make_size(
            size.frames,
            size.tokens,
            canvas=image.Config(size.canvas.height, size.canvas.width),
            conditions=tuple(shape.condition() for shape in size.conditions),
            vision_spans=size.spans,
        )

    def resolve(self, layout: Layout):
        """The denoiser's layout of ``layout``.

        Raises:
            ValueError: ``layout`` names no layout of the denoiser.
        """
        denoiser = self.denoiser
        resolved = denoiser.layout_size(
            denoiser.make_size(
                layout.frames,
                layout.text_rows,
                canvas=image.Config(layout.height, layout.width),
            )
        )
        if layout.condition_rows:
            resolved = denoiser.condition_layout(
                resolved, layout.condition_rows
            )
        if _describe(resolved) != layout:
            raise ValueError(f"{list(layout)} is not a layout of the denoiser")
        return resolved

    def legal_frame_count(self, frames: int) -> int:
        return int(self.denoiser.legal_frame_count(int(frames)))

    def max_condition_rows(self, frames: int, canvas: Canvas) -> int:
        """Packed rows of the largest condition set a request may bring."""
        denoiser = self.denoiser
        raster = image.Config(canvas.height, canvas.width)
        conditions = denoiser.max_conditions(int(frames), raster)
        if not conditions:
            return 0
        size = denoiser.make_size(
            int(frames), 1, canvas=raster, conditions=conditions
        )
        return int(size.condition_rows)

    def layout_size(self, size: VideoSize) -> Layout:
        """The smallest layout holding ``size``."""
        return _describe(self.denoiser.layout_size(self.size(size)))

    def condition_layout(self, layout: Layout, rows: int) -> Layout:
        """``layout`` widened so its condition region holds ``rows``."""
        return _describe(
            self.denoiser.condition_layout(self.resolve(layout), int(rows))
        )

    def latent_elements(self, modality: str, layout: Layout) -> int:
        """Elements of one modality's complete sample in ``layout``."""
        return _elements(
            self.denoiser.latent_shape(modality, self.resolve(layout))
        )

    def fields(self, layout: Layout) -> LayoutFields:
        """The request state of ``layout`` on this rank."""
        denoiser = self.denoiser
        resolved = self.resolve(layout)
        buffers = denoiser.state_buffers(resolved)
        outputs = denoiser.output_layout(resolved)
        samples = []
        for name in self.modalities:
            shape = denoiser.latent_shape(name, resolved)
            local = outputs[name].local_slice[0]
            samples.append(
                (
                    name,
                    SampleRows(
                        int(shape[0]),
                        int(shape[1]),
                        int(local.start),
                        int(local.stop),
                    ),
                )
            )
        workspace = cast(Any, denoiser).workspace_buffers(resolved)
        return LayoutFields(
            state=tuple(
                StateField(
                    name,
                    config.dtype,
                    _elements(config.shape),
                    name in self.modalities,
                )
                for name, config in buffers.items()
            ),
            samples=tuple(samples),
            noise=tuple(
                (name, _elements(denoiser.noise_shape(name, resolved)))
                for name in self.modalities
            ),
            condition_noise=int(denoiser.condition_noise_capacity(resolved)),
            text_condition=(
                int(denoiser.text_condition_rows(resolved)),
                self.text_condition_width,
            ),
            workspace=sum(
                _elements(config.shape) for config in workspace.values()
            ),
        )

    # The media timeline.

    def unit_frames(self, frames: int) -> list[int]:
        """Frames of each media unit of a video of ``frames`` frames."""
        return [
            clip.stop - clip.start for clip in self.decoder.clips(int(frames))
        ]

    def track_samples(self, frames: int) -> int:
        """Samples of the audio track generated with ``frames`` frames."""
        return int(self.audio.track_samples(int(frames), self.frame_rate))


def _require(loaded, kind: type):
    module = capability(loaded.module, kind)
    if module is None:
        raise SetupError(f"the model has no {kind.__name__}")
    return module


def _denoiser(loaded, component: str) -> VideoDenoiser:
    """The video denoiser at the root of a declared component."""
    entry = loaded.description.entries.get(component)
    if entry is None:
        raise SetupError(f"the model declares no component {component!r}")
    root = loaded.module
    module = root.get_submodule(entry.component) if entry.component else root
    if not isinstance(module, VideoDenoiser):
        raise SetupError(f"component {component!r} is not a video denoiser")
    return module


def _describe(layout) -> Layout:
    return Layout(
        int(layout.num_frames),
        int(layout.canvas.width),
        int(layout.canvas.height),
        int(layout.num_text_tokens),
        int(layout.condition_rows),
    )


def admitted_canvases(
    offered: Sequence[tuple[int, str, Canvas]],
    resolutions: Sequence[str],
    aspect_ratios: Sequence[str],
) -> list[tuple[int, str, Canvas]]:
    """The canvases a deployment prepares and admits, in its order.

    ``offered`` are the canvases the denoiser generates, each named by the
    ``(short_edge, aspect_ratio)`` a request selects it with. Every
    resolution class of ``resolutions`` (``768p``) is crossed with every
    ratio of ``aspect_ratios`` (``16:9``), resolution-major, and each pair
    admits the canvas the denoiser names by it. Without resolutions, or
    without aspect ratios, every one the denoiser offers takes their place,
    in its own order; without either, every offered canvas is admitted.

    Raises:
        ValueError: A resolution is not a class, or a name repeats.
        SetupError: The denoiser offers no canvas, or names none by a
            selected pair.
    """
    if not offered:
        raise SetupError("the video denoiser prepares no canvas")
    if not resolutions and not aspect_ratios:
        return list(offered)
    edges = []
    for name in resolutions:
        digits = name.removesuffix("p")
        if not name.endswith("p") or not digits or not digits.isdigit():
            raise ValueError(
                f"a video resolution is a class such as 768p, got {name!r}"
            )
        edges.append(int(digits))
    if not edges:
        for short_edge, _, _ in offered:
            if short_edge not in edges:
                edges.append(short_edge)
    ratios = list(aspect_ratios)
    if not ratios:
        for _, ratio, _ in offered:
            if ratio not in ratios:
                ratios.append(ratio)
    if len(set(edges)) != len(edges) or len(set(ratios)) != len(ratios):
        raise ValueError("a video resolution or aspect ratio repeats")
    admitted = []
    for edge in edges:
        for ratio in ratios:
            for served in offered:
                if served[0] == edge and served[1] == ratio:
                    admitted.append(served)
                    break
            else:
                generated = ", ".join(
                    f"{short_edge}p {name}" for short_edge, name, _ in offered
                )
                raise SetupError(
                    f"the video denoiser generates only {generated}, not "
                    f"{edge}p {ratio}"
                )
    return admitted


def prompt_capacities(listed: Sequence[int], max_text_tokens: int) -> list[int]:
    """The prompt capacities, in tokens, a deployment prepares layouts for.

    They ascend. The prompt capacity ``max_text_tokens``
    (``--max-model-len``) is always the largest. ``listed``
    (``video_config.text_capacities``) names the capacities below it; a
    listed capacity equal to it repeats the largest.
    Without a list the capacities are :data:`TEXT_CAPACITY_FIRST`, then
    steps of :data:`TEXT_CAPACITY_STEP` below the prompt capacity.

    Raises:
        ValueError: A listed capacity is zero or above the prompt capacity.
    """
    if any(tokens == 0 or tokens > max_text_tokens for tokens in listed):
        raise ValueError(
            "video text capacities must be positive and not exceed the "
            f"prompt capacity {max_text_tokens}, got {list(listed)}"
        )
    if listed:
        below = list(listed)
    else:
        below = [
            TEXT_CAPACITY_FIRST,
            *range(TEXT_CAPACITY_STEP, max_text_tokens, TEXT_CAPACITY_STEP),
        ]
        below = [tokens for tokens in below if tokens < max_text_tokens]
    return sorted({*below, max_text_tokens})


@dataclass(frozen=True)
class _CanvasExtent:
    canvas: Canvas
    # State and workspace elements of the canvas's largest layout.
    storage: int
    # Generated video elements of the canvas's longest video.
    generated: int


def preparation_order(extents: Sequence[_CanvasExtent]) -> list[Canvas]:
    """The order a deployment prepares its canvases in.

    Decreasing generated elements, in their given order on ties. Generated
    elements count the complete sample, the same on every rank, so every
    rank prepares, warms and captures its layouts in one order, the order
    in which their forwards exchange tensors with the other ranks.
    """
    ordered = sorted(extents, key=lambda extent: -extent.generated)
    return [extent.canvas for extent in ordered]


def storage_leading(extents: Sequence[_CanvasExtent]) -> int:
    """The canvas whose largest layout takes the most storage on this rank.

    Returns the canvas's index: the first such canvas on ties. The storage
    is the layout's state and workspace, and that layout sizes the storage
    every layout of the rank shares.

    Generated rows alone do not select it: 768p 21:9 generates as many rows
    as 16:9 but pads to more sparse-attention tiles. Under sequence
    parallelism a rank's share of a layout differs between ranks, so ranks
    may select different canvases.
    """
    leading = 0
    for index, extent in enumerate(extents):
        if extent.storage > extents[leading].storage:
            leading = index
    return leading


@dataclass(frozen=True)
class Admission:
    """A request admitted against a plan."""

    size: VideoSize
    # The request's condition rows.
    condition_rows: int
    # The layout the request evaluates in.
    layout: Layout
    # The prompt capacity the layout was selected at.
    text_capacity: int


@dataclass
class MediaPlan:
    """What a rank admits and the layouts it prepares, fixed at startup."""

    # Admitted frame counts, ascending.
    frame_counts: list[int]
    # Admitted canvases in the order the deployment names them, with the
    # names the denoiser gives them.
    admitted: list[tuple[int, str, Canvas]]
    # The admitted canvases in preparation order (:func:`preparation_order`).
    canvases: list[Canvas]
    # Prompt capacities of the prepared layouts, ascending.
    text_capacities: list[int]
    # The longest admitted prompt.
    max_text_tokens: int
    # Condition rows a request may occupy: the deployment's stated capacity,
    # or without one the rows of the largest condition set the denoiser
    # admits; zero for a denoiser that admits none.
    condition_rows: int
    # The layout among ``largest`` that takes the most storage on this rank
    # (:func:`storage_leading`); it sizes the storage every layout shares.
    maximum: Layout
    # Startup layouts: canvases (in preparation order) by frame counts
    # (longest first) by prompt capacities (largest first). Empty for a
    # denoiser that is not text-only: every request it admits brings
    # conditions.
    layouts: list[Layout]
    # Under CUDA graphs, the startup layouts of requests with conditions, in
    # the order of ``layouts``: each widened to the condition capacity, which
    # holds every condition set the plan admits. Empty without graphs, where
    # such a request evaluates in a layout of its own.
    conditioned: list[Layout]
    # Each canvas's longest layout at the prompt capacity, widened to the
    # condition capacity; their envelope bounds every request's state.
    largest: list[Layout]

    @classmethod
    def new(
        cls,
        model: VideoModel,
        max_seconds: float,
        min_seconds: float | None,
        max_text_tokens: int,
        requested: Sequence[int],
        stated_rows: int | None,
        admitted: list[tuple[int, str, Canvas]],
        graphs: bool,
    ) -> MediaPlan:
        """Plan a deployment's admitted sizes on the loaded denoiser.

        ``requested`` are the configured prompt capacities below the prompt
        capacity ``max_text_tokens``, which is always the largest
        (:func:`prompt_capacities`); each maps to the text rows of its
        layout, capped at the largest layout's. ``stated_rows`` is the
        deployment's condition capacity (``video_config.max_condition_rows``);
        without it the plan provisions the rows of the largest condition set
        the denoiser admits at the longest frame count, on the admitted
        canvas where it packs the most, so no request is refused for its
        conditions. A denoiser that admits no condition gets none either
        way. ``admitted`` are the canvases the deployment prepares
        (:func:`admitted_canvases`), in the order it names them. Under CUDA
        ``graphs`` requests with conditions also evaluate in startup layouts,
        widened to the condition capacity (``conditioned``); without graphs
        such a request evaluates in a layout of its own.

        Raises:
            SetupError: The denoiser admits no frame count of the duration
                capacity.
        """
        rate = model.frame_rate
        longest = model.legal_frame_count(requested_frames(max_seconds, rate))
        first = (
            1
            if min_seconds is None
            else max(requested_frames(min_seconds, rate), 1)
        )
        count = model.legal_frame_count(first)
        frame_counts = [count]
        while count < longest:
            count = model.legal_frame_count(count + 1)
            frame_counts.append(count)
        if count != longest:
            raise SetupError(
                "the video denoiser admits no frame count of the duration "
                "capacity"
            )
        admitted_rows = max(
            (
                model.max_condition_rows(longest, canvas)
                for _, _, canvas in admitted
            ),
            default=0,
        )
        condition_rows = (
            0
            if admitted_rows == 0
            else (admitted_rows if stated_rows is None else stated_rows)
        )

        def widen(layout: Layout) -> Layout:
            if condition_rows == 0:
                return layout
            return model.condition_layout(layout, condition_rows)

        largest = []
        extents = []
        for _, _, canvas in admitted:
            plain = model.layout_size(
                VideoSize(longest, canvas, max_text_tokens)
            )
            layout = widen(plain)
            extents.append(
                _CanvasExtent(
                    canvas,
                    model.fields(layout).storage(),
                    model.latent_elements("video", plain),
                )
            )
            largest.append(layout)
        ordered = preparation_order(extents)
        reference = ordered[0]
        largest_text = model.layout_size(
            VideoSize(longest, reference, max_text_tokens)
        ).text_rows
        text_capacities = sorted(
            {
                min(
                    model.layout_size(
                        VideoSize(longest, reference, tokens)
                    ).text_rows,
                    largest_text,
                )
                for tokens in prompt_capacities(requested, max_text_tokens)
            }
        )

        layouts = []
        conditioned = []
        for canvas in ordered:
            for frames in reversed(frame_counts):
                for tokens in reversed(text_capacities):
                    plain = model.layout_size(VideoSize(frames, canvas, tokens))
                    if graphs and condition_rows > 0:
                        conditioned.append(widen(plain))
                    if model.text_only:
                        layouts.append(plain)
        return cls(
            frame_counts=frame_counts,
            admitted=admitted,
            canvases=ordered,
            text_capacities=text_capacities,
            max_text_tokens=max_text_tokens,
            condition_rows=condition_rows,
            maximum=largest[storage_leading(extents)],
            layouts=layouts,
            conditioned=conditioned,
            largest=largest,
        )

    @property
    def max_frames(self) -> int:
        """The longest admitted video's frames."""
        return self.frame_counts[-1]

    def startup(self) -> list[Layout]:
        """Every layout startup prepares, each once, in a shared order.

        The order every rank shares is ``layouts``, ``conditioned``, then
        ``largest``. A layout's warm step exchanges tensors with the other
        ranks, so the order must not depend on this rank's share of a
        layout; the ``maximum``, which does, is one of ``largest``.
        """
        prepared: list[Layout] = []
        for layout in (*self.layouts, *self.conditioned, *self.largest):
            if layout not in prepared:
                prepared.append(layout)
        return prepared

    def refusal(self, size: VideoSize) -> str | None:
        """The refusal of a size outside the plan; ``None`` if it admits it.

        The plan admits its frame counts, canvases and prompt capacity.
        """
        if (
            size.frames not in self.frame_counts
            or size.canvas not in self.canvases
            or size.tokens > self.max_text_tokens
        ):
            return OUTSIDE_PLAN
        return None

    def startup_layout(
        self, frames: int, canvas: Canvas, capacity: int, conditions: bool
    ) -> Layout | None:
        """The startup layout of a size at prompt capacity ``capacity``.

        One of ``layouts`` for a size without conditions and one of
        ``conditioned`` for a size with them, in the order :meth:`new` lists
        them; ``None`` outside the plan.
        """
        if (
            canvas not in self.canvases
            or frames not in self.frame_counts
            or capacity not in self.text_capacities
        ):
            return None
        canvas_index = self.canvases.index(canvas)
        frame_index = (
            len(self.frame_counts) - 1 - self.frame_counts.index(frames)
        )
        capacities = len(self.text_capacities)
        capacity_index = capacities - 1 - self.text_capacities.index(capacity)
        layouts = self.conditioned if conditions else self.layouts
        index = (
            canvas_index * len(self.frame_counts) + frame_index
        ) * capacities + capacity_index
        return layouts[index] if index < len(layouts) else None

    def admit(self, model: VideoModel, size: VideoSize) -> Admission | str:
        """The admission of a request size, or its refusal.

        The denoiser's size rule refuses what it does not serve. The request
        evaluates in the layout of the smallest prompt capacity that holds
        it: a startup layout for a size without conditions, otherwise the
        denoiser's layout of the size at that capacity, widened to the
        condition capacity where startup prepared conditioned layouts.
        """
        refusal = self.refusal(size)
        if refusal is not None:
            return refusal
        try:
            resolved = model.size(size)
        except ValueError as error:
            return str(error)
        condition_rows = int(resolved.condition_rows)
        if condition_rows > self.condition_rows:
            return (
                f"conditions: the conditions take {condition_rows} denoiser "
                f"rows, more than the {self.condition_rows} this deployment "
                "serves; serve with a larger video_config.max_condition_rows"
            )
        conditioned = bool(size.conditions)
        for capacity in self.text_capacities:
            layout = self.startup_layout(
                size.frames, size.canvas, capacity, conditioned
            )
            if layout is None:
                layout = model.layout_size(
                    VideoSize(
                        size.frames,
                        size.canvas,
                        capacity,
                        size.conditions,
                    )
                )
            if model.denoiser.holds(model.resolve(layout), resolved):
                return Admission(size, condition_rows, layout, capacity)
        return "media input has no admitted capacity layout"


@dataclass(frozen=True)
class SamplePages:
    """The latent pages of one request slot."""

    # Elements of one page.
    page_units: int
    # Pages a request slot owns.
    pages: int

    @classmethod
    def plan(cls, model: VideoModel, largest: Sequence[Layout]) -> SamplePages:
        """Pages holding the complete samples of every ``largest`` layout.

        The global extents bound every sequence-parallel shard, even when a
        shorter prompt moves more media rows onto one rank.

        Raises:
            SetupError: The denoiser describes empty samples.
        """
        capacity = max(
            sum(
                _aligned(model.latent_elements(name, layout))
                for name in model.modalities
            )
            for layout in largest
        )
        if capacity == 0:
            raise SetupError("the video denoiser describes empty samples")
        page_units = _aligned(-(-capacity // REQUEST_PAGES))
        return cls(page_units, -(-capacity // page_units))

    def for_slot(self, slot: int) -> range:
        """Page zero is padding; every slot owns a consecutive run after it."""
        return range(1 + (slot - 1) * self.pages, 1 + slot * self.pages)


@dataclass(frozen=True)
class SampleLayout:
    """Where a layout's local samples sit in a slot's pages."""

    # Element offset of each modality's local sample, in modality order.
    offsets: tuple[int, ...]
    # Elements of each modality's local sample.
    elements: tuple[int, ...]
    # Pages the samples occupy.
    pages: int

    @classmethod
    def of(cls, fields: LayoutFields, page_units: int) -> SampleLayout:
        offsets = []
        elements = []
        cursor = 0
        for name, _ in fields.samples:
            state = next(
                (
                    each
                    for each in fields.state
                    if each.name == name and each.sample
                ),
                None,
            )
            if state is None:
                raise SetupError(f"layout state lacks sample {name}")
            offsets.append(cursor)
            elements.append(state.elements)
            cursor += _aligned(state.elements)
        return cls(tuple(offsets), tuple(elements), -(-cursor // page_units))

    def storage(self, fields: LayoutFields) -> dict[str, Any]:
        """How a request slot's pages hold this rank's share of the samples.

        The copy manifest library's description of one storage
        (``storage::manifest::StorageLayout``), the slot's pages back to
        back. Each modality's complete sample is a logical tensor,
        ``<modality>_latents`` of ``[rows, width]``; the rank holds its rows
        ``[start, stop)`` of it (``fields.samples``, the denoiser's
        distribution of the sequence) and every channel, row-major float32
        at its element offset of the pages. The storage ends with the last
        value the rank holds, so the slot's later pages hold nothing of the
        request; a modality the rank holds no row of has no field.
        ``fields`` are the layout's this sample layout was made of.
        """
        described = []
        end = 0
        for (name, rows), offset, elements in zip(
            fields.samples, self.offsets, self.elements, strict=True
        ):
            count = rows.stop - rows.start
            if count * rows.width != elements:
                raise SetupError(
                    f"the {name} sample's rows disagree with its elements"
                )
            if count == 0:
                continue
            logical = Logical(
                (rows.rows, rows.width),
                ("row", "channel"),
                (1, 0),
                (
                    (Segment(Span(rows.start, count), 0),),
                    (Segment(Span(0, rows.width), 0),),
                ),
            )
            described.append(
                {
                    "tensor": f"{name}_latents",
                    # No checkpoint source loads it: the trajectory's
                    # content keys its values.
                    "source": "",
                    "dtype": "float32",
                    "element": _SAMPLE_BYTES,
                    "offset": offset * _SAMPLE_BYTES,
                    "shape": [count, rows.width],
                    "strides": [rows.width, 1],
                    "logical": logical.describe(),
                    "encoding": {"kind": "affine"},
                }
            )
            end = max(end, (offset + elements) * _SAMPLE_BYTES)
        return {"bytes": end, "fields": described}


@dataclass(frozen=True)
class SlotField:
    """One field of a request slot: its elements and dtype."""

    name: str
    dtype: torch.dtype
    elements: int


@dataclass
class StateEnvelope:
    """The capacity of every field across the layouts a rank admits."""

    # Device fields of a request slot: every state table and the retained
    # conditioning, as flat elements.
    device: dict[str, SlotField] = field(default_factory=dict)
    # Pinned host fields of a request slot: every state field's source, each
    # modality's native draw and the condition draws.
    host: dict[str, SlotField] = field(default_factory=dict)
    # Sample elements the scratch holds: a slot's pages, which hold the
    # complete samples and so any rank's share of them.
    samples: int = 0

    @classmethod
    def of(
        cls, layouts: Sequence[LayoutFields], pages: SamplePages
    ) -> StateEnvelope:
        """The envelope of ``layouts``' fields; the text conditioning is BF16.

        Raises:
            SetupError: A field changes its dtype across layouts, or a
                layout's samples exceed a slot's pages.
        """
        envelope = cls(samples=pages.pages * pages.page_units)
        for fields in layouts:
            for state in fields.state:
                if state.sample:
                    # A rank's rows of a sample move with the rows before
                    # them in the sequence: the complete sample bounds every
                    # share.
                    rows = dict(fields.samples)[state.name]
                    elements = rows.rows * rows.width
                else:
                    envelope._grow(
                        envelope.device, state.name, state.dtype, state.elements
                    )
                    elements = state.elements
                envelope._grow(
                    envelope.host, f"{state.name}_source", state.dtype, elements
                )
            for name, elements in fields.noise:
                envelope._grow(
                    envelope.host, f"{name}_noise", torch.float32, elements
                )
            envelope._grow(
                envelope.host,
                "condition_noise",
                torch.float32,
                max(fields.condition_noise, 1),
            )
            text_rows, text_width = fields.text_condition
            envelope._grow(
                envelope.device,
                "text_condition",
                torch.bfloat16,
                text_rows * text_width,
            )
            if SampleLayout.of(fields, pages.page_units).pages > pages.pages:
                raise SetupError(
                    "a video layout's samples exceed the request's latent pages"
                )
        return envelope

    @staticmethod
    def _grow(
        fields: dict[str, SlotField],
        name: str,
        dtype: torch.dtype,
        elements: int,
    ) -> None:
        held = fields.get(name)
        if held is None:
            fields[name] = SlotField(name, dtype, elements)
        elif held.dtype != dtype:
            raise SetupError(
                f"state field {name} changes its dtype across layouts"
            )
        elif elements > held.elements:
            fields[name] = SlotField(name, dtype, elements)

    def slot_bytes(self) -> int:
        """Device bytes of one request slot's state."""
        return sum(
            each.elements * each.dtype.itemsize for each in self.device.values()
        )


def decode_conditioning(payload: bytes | str) -> dict[str, Any]:
    """A request's decoded ``models::video::Conditioning``."""
    return json.loads(payload)
