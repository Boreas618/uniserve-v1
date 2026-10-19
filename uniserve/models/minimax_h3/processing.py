"""The MiniMax-H3 request rules that Python computation and tools read.

The serving path plans and presents every MiniMax-H3 request in Rust
(``crates/server/src/serving/video``): the canvas and frame count, the size
every condition is prepared at, the Qwen3-VL vision grids and the tokenized
presentation. Those rules are ``uniserve-core``'s (``crates/core/src/video``),
checked against ``tests/python/fixtures/minimax_h3_plan.json``, which is
generated from the diffusers reference pipeline.

This module calls the parts of the same rules that the model package and its
tools use, through the rank runtime library (``uniserve._worker``): the
tasks, the canvas rule, the largest references a request may bring, which a
denoiser provisions its condition capacity for, the canvas a target names
among the canvases a deployment serves, and the frame count of a duration.
The canvases the model generates, keyed by the names a request selects them
with, are the model's own (``config.NAMED_CANVASES`` and
``config.DMD_CANVASES``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from uniserve.media import image
from uniserve.worker import library

from .packing import FPS, align_num_frames, video_latent_frames

__all__ = [
    "PlanError",
    "ReferenceBounds",
    "Task",
    "canvas",
    "frame_count",
    "named_canvas",
    "reference_bounds",
]


def _canvases(values: Sequence[int]) -> tuple[image.Config, ...]:
    """Canvases from the rules' flat ``[width, height, ...]``."""
    return tuple(
        image.Config(height=int(height), width=int(width))
        for width, height in zip(values[::2], values[1::2], strict=True)
    )


class Task(StrEnum):
    """A MiniMax-H3 task, by its request name.

    The members are the rules' tasks (``uniserve_core::VideoTask``) in their
    canonical order; importing the module refuses an enum that differs from
    them, so a task the rules add or rename cannot be missed here.
    """

    T2VA = "t2va"
    FL2VA = "fl2va"
    REF2VA = "ref2va"


_RULE_TASKS = tuple(str(name) for name in library()["video_tasks"]())
if tuple(task.value for task in Task) != _RULE_TASKS:
    raise ImportError(
        f"processing.Task names {[task.value for task in Task]}, but the "
        f"request rules name the tasks {list(_RULE_TASKS)}"
    )


class PlanError(ValueError):
    """A target MiniMax-H3 cannot serve.

    Attributes:
        field: The request field at fault, such as ``"target.aspect_ratio"``.
    """

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field


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


def reference_bounds(
    num_frames: int, *, image_short_edge: int
) -> ReferenceBounds:
    """Bound every reference of a request that generates ``num_frames``.

    An image's rows grow with how far its aspect lies from 1:1 at the
    encoding short edge, at most ``image_short_edge`` (the denoiser's
    ``ReferenceLimits.image_short_edge``); a video keeps at most the
    generated frames, of which the VAE encodes the leading frames forming a
    count H3 generates on the canvas of its aspect; a track is truncated to
    the generated duration. A denoiser sizes these shapes with its own
    packing to state the largest condition set it admits.
    """
    images, frames, samples, canvases = library()["video_reference_bounds"](
        int(num_frames), FPS, _frame_clips(), int(image_short_edge)
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

    The duration times the model's output frame rate (``packing.FPS``, 24)
    rounds half to even and aligns up to the next ``17 * n + 5``: 4 seconds
    give 107 frames, 15 seconds 362.

    Args:
        seconds: The requested duration.
        max_seconds: The longest duration served; by default the model's
            15 seconds.

    Raises:
        ValueError: ``seconds`` lies outside ``[4, max_seconds]``.
    """
    longest = None if max_seconds is None else float(max_seconds)
    return int(
        library()["video_frame_count"](
            float(seconds), longest, FPS, _frame_clips()
        )
    )


def _frame_clips() -> list[int]:
    """The H3 frame rule as the request rules take it.

    ``[min_frames, clip_frames, clip_latent_frames, tail_latent_frames]``,
    read from the counts the model generates (``packing.align_num_frames``)
    and the latent frames it encodes each to
    (``packing.video_latent_frames``), as the denoiser runner declares them
    in the worker's handshake.
    """
    fewest = align_num_frames(1)
    clip = align_num_frames(fewest + 1) - fewest
    per_clip = video_latent_frames(fewest + clip) - video_latent_frames(fewest)
    return [fewest, clip, per_clip, video_latent_frames(fewest) - per_clip]


def named_canvas(
    task: Task,
    *,
    short_edge: int,
    aspect_ratio: str,
    canvases: Mapping[tuple[int, str], image.Config],
) -> image.Config | None:
    """Return the canvas a target names without media.

    ``canvases`` are the canvases a deployment serves, keyed by the
    ``(short_edge, "W:H")`` a request selects them with, such as the model's
    ``config.NAMED_CANVASES``. ``auto`` is 16:9 for ``t2va`` and ``ref2va``;
    ``None`` is ``fl2va`` with ``auto``, whose canvas follows the first
    keyframe. ``fl2va`` also takes any ratio from 1:4 to 4:1, which the canvas
    rule resolves at its 768-pixel short edge.

    Raises:
        PlanError: No served canvas the ratio names at the short edge, and
            the canvas rule does not apply to it.
    """
    served = [
        [int(edge), str(ratio), raster.width, raster.height]
        for (edge, ratio), raster in canvases.items()
    ]
    rejection, values = library()["video_named_canvas"](
        task.value, int(short_edge), aspect_ratio, served
    )
    _raise(rejection)
    return None if values is None else _canvases(values)[0]


def _raise(rejection: Sequence[Any] | None) -> None:
    """Raise the rules' ``[field, message]`` rejection, if any."""
    if rejection is not None:
        field_path, message = rejection
        raise PlanError(str(field_path), str(message))
