"""Plan a conditioned MiniMax-H3 request and present it to the conditioner.

A MiniMax-H3 request is sized before any pixel is decoded. The target canvas
and frame count, the size every condition is prepared at, the Qwen3-VL vision
grids, the denoiser's condition rows and the tokenized presentation all
follow from the request and a few probed facts per condition: the displayed
size of an image, the displayed aspect, frame rate and frame count of a
video, and the sample rate and sample count of a soundtrack.

The rules are those of ``uniserve-core`` (``crates/core/src/video``), the
one implementation the server plans requests with. This module calls them
through the rank runtime library (``uniserve._worker``) and returns their
results as the dataclasses below: ``plan_request`` resolves a
``RequestPlan`` and ``present`` tokenizes the presentation the conditioner
reads with the caller's tokenizer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from uniserve.media import image
from uniserve.worker import library

__all__ = [
    "NAMED_ASPECT_RATIOS",
    "AudioClip",
    "AudioFacts",
    "Condition",
    "ConditionPlan",
    "ConditionType",
    "CoverCrop",
    "FramePosition",
    "ImageFacts",
    "ImageVision",
    "KeyframeFit",
    "PlanError",
    "Presentation",
    "ReferenceBounds",
    "RequestPlan",
    "Role",
    "Target",
    "Task",
    "TextSegment",
    "Tokenizer",
    "VideoClip",
    "VideoFacts",
    "VideoVision",
    "VisionConfig",
    "VisionSegment",
    "buckets",
    "canvas",
    "frame_count",
    "named_canvas",
    "plan_request",
    "present",
    "presentation_segments",
    "reference_bounds",
]


def _canvases(values: Sequence[int]) -> tuple[image.Config, ...]:
    """Canvases from the rules' flat ``[width, height, ...]``."""
    return tuple(
        image.Config(height=int(height), width=int(width))
        for width, height in zip(values[::2], values[1::2], strict=True)
    )


def _ratios(values: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Aspect ratios from the rules' flat ``[width, height, ...]``."""
    return tuple(
        (int(width), int(height))
        for width, height in zip(values[::2], values[1::2], strict=True)
    )


# The aspect ratios t2va and ref2va accept besides ``auto``, as
# ``(width, height)`` in canonical order.
NAMED_ASPECT_RATIOS = _ratios(library()["video_aspect_ratios"]())


class Task(StrEnum):
    """A MiniMax-H3 task, by its request name."""

    T2VA = "t2va"
    FL2VA = "fl2va"
    REF2VA = "ref2va"


class ConditionType(StrEnum):
    """The media type of a condition, by its request name.

    ``VIDEO`` conditions on a soundtrack when the video has one;
    ``VIDEO_AUDIO`` requires one.
    """

    IMAGE = "image"
    VIDEO = "video"
    VIDEO_AUDIO = "video_audio"
    AUDIO = "audio"


class Role(StrEnum):
    """Whether a condition anchors a generated frame or is a reference."""

    KEYFRAME = "keyframe"
    REFERENCE = "reference"


class FramePosition(StrEnum):
    """The generated frame a keyframe anchors."""

    FIRST = "first"
    LAST = "last"


class PlanError(ValueError):
    """A request MiniMax-H3 cannot serve.

    Attributes:
        field: The request field at fault, such as ``"conditions[2]"`` or
            ``"target.aspect_ratio"``.
    """

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field


@dataclass(frozen=True, slots=True)
class ImageFacts:
    """An image as displayed, after its EXIF orientation."""

    width: int
    height: int


@dataclass(frozen=True, slots=True)
class AudioFacts:
    """A soundtrack's native sample rate and decoded sample count."""

    sample_rate: int
    samples: int


@dataclass(frozen=True, slots=True)
class VideoFacts:
    """A video's display aspect, frame rate, frame count and soundtrack.

    Attributes:
        display_width: Width of the displayed aspect. Only the ratio to
            ``display_height`` matters; a probe passes the coded size scaled
            by the sample aspect ratio and turned by the display rotation.
        display_height: Height of the displayed aspect.
        frame_rate: The container's average frame rate.
        frames: Decoded frames of the whole video.
        soundtrack: The first audio stream, when the video has one.
    """

    display_width: int
    display_height: int
    frame_rate: Fraction
    frames: int
    soundtrack: AudioFacts | None = None


@dataclass(frozen=True, slots=True)
class Condition:
    """One request condition with the facts probed from its media.

    Attributes:
        type: The condition's media type.
        role: Keyframe or reference.
        media: Facts of the media, matching ``type``.
        frame_index: ``0`` or ``-1`` for a keyframe; ``None`` otherwise.
        start_seconds: Offset into a video reference; ``None`` for zero.
    """

    type: ConditionType
    role: Role
    media: ImageFacts | VideoFacts | AudioFacts
    frame_index: int | None = None
    start_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class Target:
    """The requested output: short edge, aspect ratio and duration.

    The default short edge is the canvas rule's 768 pixels.
    """

    aspect_ratio: str = "auto"
    duration_seconds: float | None = None
    short_edge: int = 768


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """Qwen3-VL processor geometry: patching and the pixel budgets.

    Attributes:
        patch_size: Pixels per vision patch side.
        temporal_patch_size: Frames merged into one temporal patch.
        merge_size: Patches per merged vision token side.
        image_min_pixels: Lower pixel budget of an image.
        image_max_pixels: Upper pixel budget of an image.
        video_min_pixels: Lower budget of a video, frames times pixels.
        video_max_pixels: Upper budget of a video, frames times pixels.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    image_min_pixels: int
    image_max_pixels: int
    video_min_pixels: int
    video_max_pixels: int

    @classmethod
    def from_processor(cls, directory: str | Path) -> VisionConfig:
        """Read the image and video processor configs of a checkpoint.

        Args:
            directory: The checkpoint's ``processor`` directory, holding
                ``preprocessor_config.json`` and
                ``video_preprocessor_config.json``.

        Raises:
            ValueError: A config cannot be read or parsed, or the two
                configs disagree on the patch geometry.
        """
        root = Path(directory)
        values = library()["video_vision_config"](
            str(root / "preprocessor_config.json"),
            str(root / "video_preprocessor_config.json"),
        )
        return cls(*(int(value) for value in values))

    def _values(self) -> list[int]:
        """The geometry in the rules' order, which is the fields' order."""
        return [
            self.patch_size,
            self.temporal_patch_size,
            self.merge_size,
            self.image_min_pixels,
            self.image_max_pixels,
            self.video_min_pixels,
            self.video_max_pixels,
        ]


@dataclass(frozen=True, slots=True)
class CoverCrop:
    """An aspect-preserving resize followed by a centred crop.

    Attributes:
        width: Width the image is resized to before cropping.
        height: Height the image is resized to before cropping.
        left: Left edge of the crop in the resized image.
        top: Top edge of the crop in the resized image.
    """

    width: int
    height: int
    left: int
    top: int


@dataclass(frozen=True, slots=True)
class KeyframeFit:
    """How a keyframe is put on the target canvas.

    Attributes:
        position: The generated frame the keyframe anchors.
        cover_crop: ``None`` when the keyframe is stretched onto the canvas
            (the first keyframe); the crop otherwise.
    """

    position: FramePosition
    cover_crop: CoverCrop | None


@dataclass(frozen=True, slots=True)
class AudioClip:
    """The samples of a soundtrack the request conditions on.

    Attributes:
        sample_rate: The native sample rate.
        start_sample: Native samples skipped at the start.
        source_samples: Native samples kept after the start offset.
        samples: Samples after resampling to 32 kHz.
        latents: Audio latents per channel.
        rows: Denoiser rows of the clip: one per latent per stereo channel,
            packed channel-major.
    """

    sample_rate: int
    start_sample: int
    source_samples: int
    samples: int
    latents: int
    rows: int


@dataclass(frozen=True, slots=True)
class VideoClip:
    """The frames of a reference video the request conditions on.

    Attributes:
        canvas: The canvas of the video's own aspect the frames are put on.
        start_frame: Frames of the 24 fps timeline skipped at the start.
        frames: 24 fps frames kept after the start offset, at most the
            generated frame count; the conditioner samples these.
        vae_frames: The leading ``17 * n + 5`` frames the VAE encodes.
        latent_frames: Latent frames of the VAE encoding.
        soundtrack: The video's soundtrack, when it has one.
    """

    canvas: image.Config
    start_frame: int
    frames: int
    vae_frames: int
    latent_frames: int
    soundtrack: AudioClip | None


@dataclass(frozen=True, slots=True)
class ImageVision:
    """The conditioner's view of one image.

    Attributes:
        grid: Qwen ``(t, h, w)`` patch grid, ``t`` being 1.
        tokens: Vision tokens of the image.
    """

    grid: tuple[int, int, int]
    tokens: int


@dataclass(frozen=True, slots=True)
class VideoVision:
    """The conditioner's view of one reference video.

    Attributes:
        grid: Qwen ``(t, h, w)`` patch grid; ``t`` counts vision blocks.
        block_tokens: Vision tokens of each block.
        frame_indices: The 24 fps frames sampled at 2 fps.
        block_timestamps: The timestamp label of each block, in seconds.
    """

    grid: tuple[int, int, int]
    block_tokens: int
    frame_indices: tuple[int, ...]
    block_timestamps: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ConditionPlan:
    """How one condition is prepared and what it contributes.

    Attributes:
        index: Position of the condition in the request.
        type: The condition's media type.
        prepared: The keyframe fit, the size an image reference is resized
            to (LANCZOS), or the video or audio clip.
        vision: The conditioner's view, for conditions it reads.
        video_rows: Denoiser video condition rows.
        audio_rows: Denoiser audio condition rows.
    """

    index: int
    type: ConditionType
    prepared: KeyframeFit | image.Config | VideoClip | AudioClip
    vision: ImageVision | VideoVision | None
    video_rows: int
    audio_rows: int


@dataclass(frozen=True, slots=True)
class RequestPlan:
    """The resolved size of a request and of every condition.

    Attributes:
        task: The request's task.
        canvas: The generated canvas.
        duration_seconds: The requested duration: the target's, or the
            soundtrack's length after its start offset when that sets it.
        num_frames: Generated frames, of the form ``17 * n + 5``.
        latent_frames: Generated video latent frames.
        audio_latents: Generated audio latents per stereo channel.
        target_video_rows: Denoiser rows of the generated video.
        target_audio_rows: Denoiser rows of the generated stereo audio.
        condition_video_rows: Denoiser video rows of all conditions.
        condition_audio_rows: Denoiser audio rows of all conditions.
        conditions: One plan per condition, in request order.
    """

    task: Task
    canvas: image.Config
    duration_seconds: float
    num_frames: int
    latent_frames: int
    audio_latents: int
    target_video_rows: int
    target_audio_rows: int
    condition_video_rows: int
    condition_audio_rows: int
    conditions: tuple[ConditionPlan, ...]
    # The rules' presentation of this plan, a function of the rank runtime
    # library: ``(prompt, encode, token_id) -> [rejection, segments,
    # token_ids, tags]``; see ``present``.
    _present: Callable[..., Any] = field(repr=False, compare=False)


def canvas(aspect_width: float, aspect_height: float) -> image.Config:
    """Resolve a display aspect ratio into the released canvas.

    The short edge starts at 768, the area is capped at ``768 * 1344`` and
    both sides are then rounded to the nearest multiple of 32, so the final
    area may slightly exceed the cap. Only the ratio of the arguments
    matters: an aspect ratio or a keyframe's displayed dimensions.

    Raises:
        ValueError: The ratio is not positive or lies outside 1:4 to 4:1.
    """
    values = library()["video_canvas"](
        float(aspect_width), float(aspect_height)
    )
    return _canvases(values)[0]


def buckets() -> tuple[image.Config, ...]:
    """Return the FastH3 training buckets, which DMD exports generate.

    The named aspect ratios' canvases at the canvas rule's 768-pixel short
    edge, then at the 480-pixel one, each in ``NAMED_ASPECT_RATIOS`` order.
    """
    return _canvases(library()["video_buckets"]())


@dataclass(frozen=True, slots=True)
class ReferenceBounds:
    """The largest references a request generating some frames may bring.

    Attributes:
        images: The two largest rasters an image reference is encoded at,
            the encoding short edge at 4:1 and at 1:4.
        video_frames: Encoded frames of the longest reference video.
        video_canvases: Every raster a reference video is encoded at, the
            canvas rule's canvases.
        audio_samples: The most 32 kHz samples of a soundtrack or audio
            reference.
    """

    images: tuple[image.Config, image.Config]
    video_frames: int
    video_canvases: tuple[image.Config, ...]
    audio_samples: int


def reference_bounds(num_frames: int) -> ReferenceBounds:
    """Bound every reference of a request that generates ``num_frames``.

    An image's rows grow with how far its aspect lies from 1:1 at the
    encoding short edge; a video keeps at most the generated frames, of
    which the VAE encodes the leading whole windows on the canvas of its
    aspect; a track is truncated to the generated duration. A denoiser
    sizes these shapes with its own packing to state the largest condition
    set it admits.
    """
    images, frames, samples, canvases = library()["video_reference_bounds"](
        int(num_frames)
    )
    widest, tallest = _canvases(images)
    return ReferenceBounds(
        images=(widest, tallest),
        video_frames=int(frames),
        video_canvases=_canvases(canvases),
        audio_samples=int(samples),
    )


def frame_count(seconds: float, max_seconds: float | None = None) -> int:
    """Return the generated frame count of a duration.

    The duration times 24 rounds half to even and aligns up to the next
    ``17 * n + 5``: 4 seconds give 107 frames, 15 seconds 362.

    Args:
        seconds: The requested duration.
        max_seconds: The longest duration served; by default the model's
            15 seconds.

    Raises:
        ValueError: ``seconds`` lies outside ``[4, max_seconds]``.
    """
    return int(
        library()["video_frame_count"](float(seconds), _seconds(max_seconds))
    )


def named_canvas(task: Task, target: Target) -> image.Config | None:
    """Return the canvas a target names without media.

    ``None`` for ``fl2va`` with ``auto``, whose canvas follows the first
    keyframe. At the canvas rule's 768-pixel short edge a ratio resolves
    through ``canvas``; another short edge names its training bucket.

    Raises:
        PlanError: The task does not accept the aspect ratio, or the short
            edge has no canvas for it.
    """
    rejection, values = library()["video_named_canvas"](
        task.value, target.short_edge, target.aspect_ratio
    )
    _raise(rejection)
    return None if values is None else _canvases(values)[0]


def plan_request(
    task: Task,
    target: Target,
    conditions: Sequence[Condition],
    vision: VisionConfig,
    *,
    tasks: Sequence[Task] = tuple(Task),
    max_seconds: float | None = None,
    canvases: Sequence[image.Config] | None = None,
) -> RequestPlan:
    """Validate a request and resolve every size it implies.

    Args:
        task: The request's task.
        target: The requested output.
        conditions: The conditions in request order, with probed facts.
        vision: The conditioner's processor geometry.
        tasks: The tasks of the denoiser that serves the request.
        max_seconds: The longest duration served; by default the model's
            15 seconds.
        canvases: The only canvases the checkpoint serves, when restricted.

    Raises:
        PlanError: The request breaks a rule; ``field`` names the culprit.
    """
    rejection, values = library()["video_plan"](
        task.value,
        [
            target.short_edge,
            target.aspect_ratio,
            _seconds(target.duration_seconds),
        ],
        [_condition(condition) for condition in conditions],
        vision._values(),
        [
            [served.value for served in tasks],
            _seconds(max_seconds),
            None
            if canvases is None
            else [
                side for size in canvases for side in (size.width, size.height)
            ],
        ],
    )
    _raise(rejection)
    sizes, duration_seconds, planned, presentation = values
    (
        width,
        height,
        num_frames,
        latent_frames,
        audio_latents,
        target_video_rows,
        target_audio_rows,
        condition_video_rows,
        condition_audio_rows,
    ) = (int(value) for value in sizes)
    return RequestPlan(
        task=task,
        canvas=image.Config(height=height, width=width),
        duration_seconds=float(duration_seconds),
        num_frames=num_frames,
        latent_frames=latent_frames,
        audio_latents=audio_latents,
        target_video_rows=target_video_rows,
        target_audio_rows=target_audio_rows,
        condition_video_rows=condition_video_rows,
        condition_audio_rows=condition_audio_rows,
        conditions=tuple(
            _condition_plan(index, condition, condition_values)
            for index, (condition, condition_values) in enumerate(
                zip(conditions, planned, strict=True)
            )
        ),
        _present=presentation,
    )


@dataclass(frozen=True, slots=True)
class TextSegment:
    """Presentation text, tokenized on its own."""

    text: str


@dataclass(frozen=True, slots=True)
class VisionSegment:
    """A vision block: start marker, ``tokens`` pads and end marker."""

    pad: str
    tokens: int


@dataclass(frozen=True, slots=True)
class Presentation:
    """The tokenized presentation and the modality tag of every token.

    Vision tokens, their markers included, carry the video tag; all other
    tokens carry the text tag.
    """

    token_ids: tuple[int, ...]
    tags: tuple[int, ...]


class Tokenizer(Protocol):
    """The tokenizer calls a presentation needs."""

    def encode(self, text: str, add_special_tokens: bool) -> list[int]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


def presentation_segments(
    plan: RequestPlan, prompt: str
) -> list[TextSegment | VisionSegment]:
    """List the presentation of a request, before tokenization.

    ``fl2va`` labels each keyframe ``"<Picture i>: "`` before its vision
    block. ``ref2va`` labels its references in request order, numbered per
    modality: a soundtrack ``"<Audio j>: "`` (a video's first, before its
    ``"<Video k>: "`` label), an image ``"<Picture i>: "`` before its block,
    and a video one ``"<{t:.1f} seconds>"`` label per vision block. Audio
    contributes labels only and ``ref2va`` keyframes nothing. The prompt
    follows verbatim.
    """
    _, segments, _, _ = plan._present(prompt, None, None)
    return [
        TextSegment(str(segment[0]))
        if len(segment) == 1
        else VisionSegment(str(segment[0]), int(segment[1]))
        for segment in segments
    ]


def present(
    tokenizer: Tokenizer, plan: RequestPlan, prompt: str
) -> Presentation:
    """Tokenize the presentation of a request.

    Every text segment is tokenized on its own without special tokens and
    without a chat template, and the pieces are concatenated. ``tokenizer``
    is the checkpoint's tokenizer as ``transformers`` loads it, with the
    special tokens its ``tokenizer_config.json`` declares beyond
    ``tokenizer.json`` (such as the ``<d>`` dialogue marker), which a prompt
    may contain.

    Raises:
        PlanError: The prompt is blank or contains a vision placeholder,
            which would misalign the conditioner's vision inputs.
    """
    rejection, _, token_ids, tags = plan._present(
        prompt,
        lambda text: list(tokenizer.encode(text, add_special_tokens=False)),
        tokenizer.convert_tokens_to_ids,
    )
    _raise(rejection)
    return Presentation(
        token_ids=tuple(int(token) for token in token_ids),
        tags=tuple(int(tag) for tag in tags),
    )


def _seconds(value: float | None) -> float | None:
    return None if value is None else float(value)


def _raise(rejection: Sequence[Any] | None) -> None:
    """Raise the rules' ``[field, message]`` rejection, if any."""
    if rejection is not None:
        field_path, message = rejection
        raise PlanError(str(field_path), str(message))


def _condition(condition: Condition) -> list[Any]:
    """One condition as the rules take it.

    ``[type, role, frame_index, start_seconds, media_kind, media_facts]``.
    """
    media = condition.media
    if isinstance(media, ImageFacts):
        kind, facts = "image", [media.width, media.height]
    elif isinstance(media, AudioFacts):
        kind, facts = "audio", [media.sample_rate, media.samples]
    else:
        rate = Fraction(media.frame_rate)
        track = media.soundtrack
        kind, facts = (
            "video",
            [
                media.display_width,
                media.display_height,
                rate.numerator,
                rate.denominator,
                media.frames,
                *(() if track is None else (track.sample_rate, track.samples)),
            ],
        )
    return [
        condition.type.value,
        condition.role.value,
        condition.frame_index,
        _seconds(condition.start_seconds),
        kind,
        facts,
    ]


def _audio_clip(values: Sequence[int]) -> AudioClip:
    return AudioClip(*(int(value) for value in values))


def _condition_plan(
    index: int, condition: Condition, values: Sequence[Any]
) -> ConditionPlan:
    """A planned condition from the rules' values.

    The values are ``[rows, prepared, vision, timestamps]``. The rules plan
    a keyframe as its fit and a reference by its media, so the request's own
    role and media select the preparation.
    """
    rows, prepared_values, vision_values, timestamps = values
    video_rows, audio_rows = (int(value) for value in rows)
    numbers = [int(value) for value in prepared_values]
    prepared: KeyframeFit | image.Config | VideoClip | AudioClip
    if condition.role is Role.KEYFRAME:
        # The anchored frame index, then the cover crop of a cropped one.
        anchor, *crop = numbers
        prepared = KeyframeFit(
            position=FramePosition.FIRST if anchor == 0 else FramePosition.LAST,
            cover_crop=CoverCrop(*crop) if crop else None,
        )
    elif isinstance(condition.media, ImageFacts):
        width, height = numbers
        prepared = image.Config(height=height, width=width)
    elif isinstance(condition.media, VideoFacts):
        width, height, start, frames, vae_frames, latent_frames, *track = (
            numbers
        )
        prepared = VideoClip(
            canvas=image.Config(height=height, width=width),
            start_frame=start,
            frames=frames,
            vae_frames=vae_frames,
            latent_frames=latent_frames,
            soundtrack=_audio_clip(track) if track else None,
        )
    else:
        prepared = _audio_clip(numbers)

    vision: ImageVision | VideoVision | None = None
    if len(vision_values):
        t, h, w, tokens, *frame_indices = (
            int(value) for value in vision_values
        )
        if isinstance(condition.media, VideoFacts):
            vision = VideoVision(
                grid=(t, h, w),
                block_tokens=tokens,
                frame_indices=tuple(frame_indices),
                block_timestamps=tuple(float(value) for value in timestamps),
            )
        else:
            vision = ImageVision(grid=(t, h, w), tokens=tokens)
    return ConditionPlan(
        index=index,
        type=condition.type,
        prepared=prepared,
        vision=vision,
        video_rows=video_rows,
        audio_rows=audio_rows,
    )
