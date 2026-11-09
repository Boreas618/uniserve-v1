"""Build video requests and inspect complete MP4s after measurement.

Every request is UniServe's model-agnostic ``/v1/videos`` body: the prompt,
the output shape the point states (``size``, ``num_frames`` and ``fps``),
the seed, and the row's ``input_references``, each naming its media by a
``file://`` URL under the point's ``video.reference_root``. The point states
the size and frame rate; a row may state its own frame count, which must
already be one the model generates. Validation holds every MP4 to its own
request's stated shape and soundtrack, and a UniServe output also to the shape
the server reported generating.
Media files are read when the request is built, before measurement, and
inspection uses each request's shape and never runs inside a load slot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from ..transport.native_video import native_request
from ..transport.video import VideoOutputError, inspect_video_bytes
from ..types import (
    VIDEOS_SYNC,
    BenchmarkPoint,
    Example,
    ReferenceMedia,
    RequestRecord,
    TaskName,
    TaskRequest,
    ValidationResult,
    VideoConfig,
    VideoShape,
    parse_size,
)
from .base import BenchmarkTask

# Media types of the reference files the request API accepts, by suffix:
# JPEG, PNG and WEBP images, MP4 and QuickTime videos, MP3 and WAV audio.
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
}

# The media type family each reference type requires.
_REFERENCE_MEDIA = {
    "image": "image/",
    "video": "video/",
    "audio": "audio/",
}

# The keys a workload row's reference may carry: its request type, its file
# under the reference root, and the request's optional anchor and offset.
_REFERENCE_KEYS = frozenset(
    {"type", "media", "frame_index", "start_time_seconds"}
)

# The point's fields that state the output shape and soundtrack every
# request is held to; the frame count may come from each row instead.
_STATED_FIELDS = ("size", "fps", "audio_sample_rate", "audio_channels")


class VideoTask(BenchmarkTask):
    """Builds video requests and validates the stated media contract."""

    name: ClassVar[TaskName] = TaskName.VIDEO
    allowed_endpoints: ClassVar[tuple[str, ...]] = (VIDEOS_SYNC, "/v1/videos")
    default_endpoint: ClassVar[str] = VIDEOS_SYNC
    default_stream: ClassVar[bool] = False

    def __init__(self, point: BenchmarkPoint) -> None:
        """Bind the point to the reference media root it reads rows from."""
        super().__init__(point)
        root = point.video.reference_root
        self.reference_root = Path(root).resolve() if root else None

    @classmethod
    def check_video(cls, video: VideoConfig, context: str) -> None:
        """Require the output shape and soundtrack a video point measures."""
        missing = [
            name for name in _STATED_FIELDS if getattr(video, name) is None
        ]
        if missing:
            raise ValueError(
                f"{context}: a video point states its output shape and "
                f"soundtrack and requires {missing}"
            )

    def build_request(self, example: Example) -> TaskRequest:
        """Build one request from the row's frame count, seed and references.

        A row without its own frame count or seed takes the point's
        ``video.num_frames`` and load seed. The row's reference media are read
        here, so a backend that uploads them sends bytes already in storage.

        Raises:
            ValueError: The point does not state its shape, neither the row
                nor the point states a frame count, a reference does not fit
                the request API, or the backend cannot be sent the request
                as the same work.
            OSError: A reference media file cannot be read.
        """
        video = self.point.video
        self.check_video(video, f"benchmark {self.point.name} video")
        frames = (
            example.num_frames
            if example.num_frames is not None
            else video.num_frames
        )
        if (
            isinstance(frames, bool)
            or not isinstance(frames, int)
            or frames < 1
        ):
            raise ValueError(
                f"row {example.id}: num_frames must be a positive integer, "
                f"stated by the row or by the point's video.num_frames"
            )
        assert video.size is not None and video.fps is not None
        width, height = parse_size(video.size)
        shape = VideoShape(width, height, frames, video.fps)
        references, media = self._references(example)

        payload: dict[str, Any] = {
            "model": self.point.model,
            "prompt": example.prompt,
            "size": shape.size,
            "num_frames": shape.frames,
            "fps": shape.fps,
            "seed": int(
                example.seed
                if example.seed is not None
                else self.point.load.seed
            ),
        }
        if references:
            payload["input_references"] = references
        request = TaskRequest(
            self.point.endpoint,
            payload,
            stream=False,
            video=video,
            video_shape=shape,
            reference_media=tuple(media),
        )
        # A request its backend cannot take as the same work fails here,
        # before measurement, rather than as a measured transport failure.
        native_request(request)
        return request

    def _references(
        self, example: Example
    ) -> tuple[list[dict[str, Any]], list[ReferenceMedia]]:
        """Return a row's request references and their media, in row order.

        Each reference names its file by an absolute ``file://`` URL. A row
        is checked only as far as the request API's reference fields and the
        backends' upload forms depend on it; the server enforces the model's
        own rules on which references it takes.

        Raises:
            ValueError: A reference has an unknown key or type, a file whose
                suffix is not its type's media, or the point has no
                reference root.
            OSError: A media file cannot be read.
        """
        rows = example.input_references or []
        if not rows:
            return [], []
        if self.reference_root is None:
            raise ValueError(
                f"row {example.id}: input_references need the point's "
                f"video.reference_root"
            )

        references: list[dict[str, Any]] = []
        media: list[ReferenceMedia] = []
        for index, row in enumerate(rows):
            context = f"row {example.id} input_references[{index}]"
            if unknown := sorted(set(row) - _REFERENCE_KEYS):
                raise ValueError(f"{context}: unknown keys {unknown}")
            kind = row.get("type")
            if kind not in _REFERENCE_MEDIA:
                raise ValueError(f"{context}: unknown type {kind!r}")
            anchor = row.get("frame_index")
            if anchor is not None and (
                isinstance(anchor, bool) or not isinstance(anchor, int)
            ):
                raise ValueError(f"{context}: frame_index must be an integer")

            path = (self.reference_root / str(row["media"])).resolve()
            mime = MEDIA_TYPES.get(path.suffix.lower())
            if mime is None or not mime.startswith(_REFERENCE_MEDIA[kind]):
                raise ValueError(f"{context}: {path.name} is not {kind} media")

            reference: dict[str, Any] = {"type": kind, "url": path.as_uri()}
            for name in ("frame_index", "start_time_seconds"):
                if row.get(name) is not None:
                    reference[name] = row[name]
            references.append(reference)
            media.append(ReferenceMedia(str(path), mime, path.read_bytes()))
        return references, media

    def inspect_output(self, record: RequestRecord) -> None:
        """Decode and classify a response outside the load window."""
        if not record.success:
            return
        try:
            record.decoded_video = inspect_video_bytes(
                record.video_body or b"", declared_mime=record.video_mime
            )
        except VideoOutputError as error:
            record.mark_failure(error.classifier, str(error))
        record.media_checks = self.validate_output([record]).checks
        if not all(record.media_checks.values()):
            failed = [
                key for key, passed in record.media_checks.items() if not passed
            ]
            if record.success:
                record.mark_failure("invalid_video_output", ", ".join(failed))

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Validate raster, frame rate, codecs, soundtrack and length.

        Each output is held to the shape its own request states
        (``RequestRecord.video_shape``) and to the point's stated soundtrack;
        a record without a stated shape fails every check. A UniServe output
        must also come with the server's report of the shape it generated,
        equal to the stated one.
        """
        video_config = self.point.video
        outputs = [record.decoded_video for record in records]
        shapes = [record.video_shape for record in records]
        present = (
            bool(outputs)
            and all(output is not None for output in outputs)
            and all(shape is not None for shape in shapes)
        )
        pairs = [
            (video, shape)
            for video, shape in zip(outputs, shapes, strict=True)
            if video is not None and shape is not None
        ]
        videos = [video for video, _ in pairs]

        # Both streams are measured against the stated media length rather
        # than each other: the video may differ from the stated frame count
        # by one frame, and the audio duration from the stated duration by
        # one frame period. A one-frame-short video with full-length audio is
        # therefore valid, while a missing or truncated stream is not.
        frame_tolerance = 1
        checks = {
            "decoded_video": present,
            "h264_video": present
            and all(video.video_codec == "h264" for video in videos),
            "stated_size": present
            and all(
                (video.width, video.height) == (shape.width, shape.height)
                for video, shape in pairs
            ),
            # The container frame rate is a rational; require exactly the
            # stated one.
            "stated_fps": present
            and all(
                video.fps_numerator == shape.fps * video.fps_denominator
                for video, shape in pairs
            ),
            "stated_frame_count": present
            and all(
                abs(video.frame_count - shape.frames) <= frame_tolerance
                for video, shape in pairs
            ),
            "stated_aac_audio": present
            and all(
                video.audio_codec == "aac"
                and video.audio_channels == video_config.audio_channels
                and video.audio_sample_rate == video_config.audio_sample_rate
                for video in videos
            ),
            "stated_audio_duration": present
            and all(
                abs(video.audio_duration_s - shape.duration_s)
                <= 1.0 / shape.fps
                for video, shape in pairs
            ),
            "nonzero_video_variance": present
            and all(
                math.isfinite(video.video_variance) and video.video_variance > 0
                for video in videos
            ),
            "nonzero_audio_rms": present
            and all(
                math.isfinite(video.audio_rms) and video.audio_rms > 0
                for video in videos
            ),
        }
        if video_config.backend == "uniserve":
            # UniServe reports the shape it generated in the job record or
            # the synchronous response headers.
            checks["served_shape"] = present and all(
                record.served_shape is not None
                and record.served_shape == record.video_shape
                for record in records
            )
        return ValidationResult(
            checks=checks,
            statistics={
                "completed_videos": len(videos),
                "media_bytes": sum(video.byte_size for video in videos),
            },
        )


__all__ = ["MEDIA_TYPES", "VideoTask"]
