"""What the video cases ask a deployment's video API, and how they hold its
answers to the request.

One helper per question, shared by every video case instead of a copy per
file: a text-to-video request and a prompt of an exact token length; the
checks of a generated MP4 against the shape its request resolved to and its
model's media format; the comparison of two generations frame by frame; the
reading of an asynchronous video job (``/v1/videos``); and the workers'
graph dispatches a request ran.
"""  # noqa: D205

from __future__ import annotations

import dataclasses
import io
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import httpx
import huggingface_hub
import numpy as np
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from uniserve_testing import admin
from uniserve_testing.assets import Checkpoint
from uniserve_testing.servers import Server

#: How long one video generation may take, in seconds (scaled by the
#: server's time scale): the longest the cases send is the base release's
#: 49 denoising steps of an eight-second video, after the requests resident
#: before it.
GENERATION_S = 3600.0

#: How often a case reads a state the server announces no event of, in
#: seconds: a video job's record and a deployment's in-flight count. A case
#: that acts at a denoising step boundary reads several times per step.
READ_S = 0.05

#: The streams of a generated MP4, as PyAV names them.
STREAMS = ("video", "audio")

#: The tolerance of two generations that compose the same BF16 computation
#: in another order (another parallel layout, or a request computed beside
#: another one): video pixels as fractions of full scale and audio samples
#: agree to within this, absolutely and relatively.
BF16_COMPOSITION = 2e-2


@dataclasses.dataclass(frozen=True)
class MediaFormat:
    """The encoding every response of a video model carries."""

    video_codec: str
    audio_codec: str
    fps: int
    audio_channels: int
    audio_sample_rate: int


#: MiniMax-H3's responses, FastH3's among them: an H.264/AAC MP4 at 24 fps
#: with stereo 32-kHz audio (``docs/minimax_h3/minimax_h3.md``).
MINIMAX_H3 = MediaFormat(
    video_codec="h264",
    audio_codec="aac",
    fps=24,
    audio_channels=2,
    audio_sample_rate=32_000,
)


def api(server: Server) -> httpx.Client:
    """A client of `server`'s client API whose answers may take a whole
    generation (:data:`GENERATION_S`, scaled).
    """  # noqa: D205
    return httpx.Client(
        base_url=server.base_url, timeout=server.scaled(GENERATION_S)
    )


def t2va_request(
    model: str, prompt: str, num_frames: int, seed: int, *, size: str
) -> dict[str, Any]:
    """A text-to-video request (one without references) for `prompt` at
    `size` (``WxH``, one the deployment serves), with exactly `num_frames`
    frames, a count the model generates, from the noise seed `seed`.
    """  # noqa: D205
    return {
        "model": model,
        "prompt": prompt,
        "size": size,
        "num_frames": num_frames,
        "seed": seed,
    }


def tokenizer(checkpoint: Checkpoint) -> PreTrainedTokenizerBase:
    """The tokenizer of the video pipeline `checkpoint`: the ``tokenizer``
    folder of its repository at its pinned revision, where a diffusers
    pipeline keeps it.

    The server offers no tokenize endpoint, so a case that sends a prompt of
    an exact token length counts the tokens with the pipeline's own
    tokenizer. The folder is resolved through the Hugging Face cache, every
    file of it fetched there unless the cache holds it or the host is
    offline (``HF_HUB_OFFLINE``), and the tokenizer is read from that
    directory: given a repository instead, the reader first looks for a
    model configuration in the folder, which a pipeline's tokenizer folder
    has none of, and an offline host cannot answer that it is absent. Skips
    the calling case when no repository publishes the checkpoint.
    """  # noqa: D205
    if checkpoint.repository is None:
        pytest.skip(checkpoint.unpublished)
    folder = (
        f"{checkpoint.folder}/tokenizer" if checkpoint.folder else "tokenizer"
    )
    snapshot = huggingface_hub.snapshot_download(
        checkpoint.repository,
        revision=checkpoint.revision,
        repo_type=checkpoint.kind,
        allow_patterns=[f"{folder}/*"],
    )
    return AutoTokenizer.from_pretrained(Path(snapshot) / folder)


#: The MiniMax-H3 benchmark description a case's exact-length prompt is cut
#: from or extended past: a two-shot paper-craft tutorial with its
#: soundscape, in the model's structured prompt format.
PROMPT = (
    "integrated_multimodal_description: [Shot 1] A 9-second 16:9 widescreen "
    "educational documentary tutorial in clean paper-textured motion-graphics "
    "design, following one illustrated paper-craft instructor at a neatly "
    "gridded tabletop. The instructor places a coral square of paper on the "
    "grid, aligns its corners, and folds it diagonally into a sharp crease; "
    'a Japanese title card reading "折り紙ランタン" appears at the upper left, '
    'while a thin animated guide line and the label "谷折り" appear beside '
    "the crease. The overhead camera makes a measured push-in as the "
    "instructor smooths the fold, with small geometric accents tracking the "
    "paper edges. [Shot 2] At 00:04.500, the same instructor unfolds the "
    "paper, rotates it a quarter turn, and presses the intersecting creases "
    "into a compact lantern shape; the camera cuts to a close three-quarter "
    "tabletop view and makes a short lateral track to reveal the dimensional "
    "form. Animated arrows trace the final fold, and the Japanese completion "
    'card "完成" settles along the lower edge as the instructor places the '
    "lantern in a small cardboard tray. overall_soundscape: Close ASMR foley "
    "records the paper's dry flex and crisp crease, fingertip taps on the "
    "matte work surface, and a soft sleeve rustle, with a quiet studio room "
    "tone underneath. A small cardboard tray clicks when the instructor sets "
    "the finished lantern down, while gentle breathing remains audible. "
    "non_diegetic_music: N/A"
)

#: The sentence repeated, whole or in part, past the end of :data:`PROMPT`.
FILLER = (
    " A coherent continuation preserves the scene, motion, lighting, and sound."
)


def exact_prompt(tokenizer: PreTrainedTokenizerBase, tokens: int) -> str:
    """:data:`PROMPT` cut or extended with :data:`FILLER` to exactly
    `tokens` tokens of `tokenizer` (special tokens not counted), so a request
    lands in the text capacity its length selects.

    The ids are cut and decoded, and the text is encoded again, since
    decoding need not preserve the count; a text that measures otherwise
    fails the calling case.
    """  # noqa: D205
    base = tokenizer.encode(PROMPT, add_special_tokens=False)
    filler = tokenizer.encode(FILLER, add_special_tokens=False)
    ids = base[:tokens]
    while len(ids) < tokens:
        ids.extend(filler[: tokens - len(ids)])

    prompt = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    measured = len(tokenizer.encode(prompt, add_special_tokens=False))
    assert measured == tokens, (measured, tokens)
    return prompt


@dataclasses.dataclass(frozen=True)
class Video:
    """What the full decoding of an MP4 found: its one video stream's
    raster, frame count, average frame rate and codec; its one audio stream's
    codec, channel count, sample rate and samples per channel; the variance
    of the video's RGB values (in bytes) and the RMS of its audio samples (as
    fractions of full scale).
    """  # noqa: D205

    width: int
    height: int
    frames: int
    fps: Fraction
    video_codec: str
    audio_codec: str
    audio_channels: int
    audio_sample_rate: int
    audio_samples: int
    video_variance: float
    audio_rms: float


def inspected(data: bytes) -> Video:
    """The MP4 `data` decoded completely. It holds exactly one video and one
    audio stream, each decoding at least one frame, with one raster and one
    audio layout and rate throughout.
    """  # noqa: D205
    with av.open(io.BytesIO(data), mode="r", format="mp4") as container:
        (video,) = container.streams.video
        (audio,) = container.streams.audio
        frames = samples = 0
        layouts = set()
        # Sums over every value: count, sum and sum of squares of the RGB
        # bytes; count and sum of squares of the audio samples.
        pixels = np.zeros(3)
        pcm = np.zeros(2)
        for frame in container.decode(video=0, audio=0):
            if isinstance(frame, av.VideoFrame):
                frames += 1
                assert (frame.width, frame.height) == (
                    video.codec_context.width,
                    video.codec_context.height,
                )
                values = frame.to_ndarray(format="rgb24").astype(np.float64)
                pixels += (values.size, values.sum(), np.square(values).sum())
                continue

            # `samples` counts per channel. An integer format is scaled to
            # full scale; a float format is at full scale already.
            samples += frame.samples
            layouts.add((len(frame.layout.channels), frame.sample_rate))
            values = frame.to_ndarray()
            scale = 1.0
            if np.issubdtype(values.dtype, np.integer):
                limits = np.iinfo(values.dtype.str)
                scale = max(abs(limits.min), limits.max)
            values = values.astype(np.float64) / scale
            pcm += (values.size, np.square(values).sum())

    assert frames > 0 and samples > 0, (frames, samples)
    ((channels, rate),) = layouts
    assert video.average_rate is not None
    mean = pixels[1] / pixels[0]
    return Video(
        width=video.codec_context.width,
        height=video.codec_context.height,
        frames=frames,
        fps=Fraction(video.average_rate),
        video_codec=video.codec_context.name,
        audio_codec=audio.codec_context.name,
        audio_channels=channels,
        audio_sample_rate=rate,
        audio_samples=samples,
        video_variance=max(0.0, pixels[2] / pixels[0] - mean**2),
        audio_rms=float(np.sqrt(pcm[1] / pcm[0])),
    )


def assert_video(
    data: bytes,
    content_type: str,
    *,
    size: str,
    frames: int,
    media_format: MediaFormat,
) -> None:
    """Hold the MP4 `data`, answered as `content_type`, to the shape its
    request resolved to and to its model's `media_format`.

    The video is `size` (``WxH``) with `frames` frames, at the format's frame
    rate in its video codec; its soundtrack is in the format's codec, channel
    count and sample rate and lasts as long as the video, to within one
    frame; and neither stream is constant, as a generated video moves and
    sounds.
    """  # noqa: D205
    assert content_type.partition(";")[0].strip() == "video/mp4", content_type
    media = inspected(data)
    width, height = (int(side) for side in size.split("x"))
    assert (media.width, media.height) == (width, height), size
    assert media.frames == frames, size
    assert media.fps == media_format.fps
    assert media.video_codec == media_format.video_codec
    assert (
        media.audio_codec,
        media.audio_channels,
        media.audio_sample_rate,
    ) == (
        media_format.audio_codec,
        media_format.audio_channels,
        media_format.audio_sample_rate,
    )
    frame_s = 1 / media_format.fps
    audio_s = media.audio_samples / media.audio_sample_rate
    assert abs(audio_s - frames * frame_s) <= frame_s
    assert media.video_variance > 0 and media.audio_rms > 0, size


def assert_generated(
    response: httpx.Response,
    *,
    size: str,
    frames: int,
    media_format: MediaFormat,
) -> None:
    """Hold the answer of ``POST /v1/videos/sync`` to its request: it
    succeeded, its headers state the shape generated (``x-video-size``,
    ``x-video-num-frames``, ``x-video-fps``), and its MP4 is that shape in
    `media_format` (:func:`assert_video`).
    """  # noqa: D205
    assert response.status_code == 200, response.text
    assert (
        response.headers["x-video-size"],
        response.headers["x-video-num-frames"],
        response.headers["x-video-fps"],
    ) == (size, str(frames), str(media_format.fps))
    assert_video(
        response.content,
        response.headers["content-type"],
        size=size,
        frames=frames,
        media_format=media_format,
    )


def decoded(data: bytes, kind: str) -> list[tuple[float | None, np.ndarray]]:
    """Every frame of the MP4 `data`'s first `kind` stream (``video`` or
    ``audio``), in decoding order: its presentation time in seconds and its
    values, a video frame's RGB pixels (``[height, width, 3]`` bytes) or an
    audio frame's samples as its sample format lays them out.
    """  # noqa: D205
    with av.open(io.BytesIO(data)) as container:
        return [
            (
                frame.time,
                frame.to_ndarray(format="rgb24")
                if kind == "video"
                else frame.to_ndarray(),
            )
            for frame in container.decode(**{kind: 0})
        ]


def assert_same_media(
    actual: bytes, expected: bytes, *, tolerance: float
) -> None:
    """The MP4s `actual` and `expected` decode to the same frames: as many of
    each stream, at the same presentation times, with video pixels (as
    fractions of full scale) and audio samples within `tolerance`, absolutely
    and relatively; a `tolerance` of zero holds them equal.
    """  # noqa: D205
    for kind in STREAMS:
        observed, reference = decoded(actual, kind), decoded(expected, kind)
        assert len(observed) == len(reference) > 0, f"{kind}: frame count"
        for index, ((time, values), (wanted_time, wanted)) in enumerate(
            zip(observed, reference, strict=True)
        ):
            assert time == wanted_time, f"{kind} frame {index}: time"
            if kind == "video":
                values = values.astype(np.float32) / 255
                wanted = wanted.astype(np.float32) / 255
            np.testing.assert_allclose(
                values,
                wanted,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=False,
                err_msg=f"{kind} frame {index}",
            )


def job(server: Server, job_id: str) -> dict[str, Any]:
    """The record of the video job `job_id` (``GET /v1/videos/{id}``): its
    ``status`` (``queued``, ``in_progress``, ``completed``, ``failed``),
    ``phase`` and ``completed_steps`` while it runs, and the shape it
    generates.
    """  # noqa: D205
    response = httpx.get(f"{server.base_url}/v1/videos/{job_id}")
    response.raise_for_status()
    return response.json()


def wait_for_job(
    server: Server,
    job_id: str,
    reached: Callable[[dict[str, Any]], bool],
    what: str,
) -> dict[str, Any]:
    """The record of the video job `job_id` once `reached` holds of it.

    The job store announces no progress of a job (``GET /v1/videos/{id}``
    has no stream or long-poll form), so the record is read every
    :data:`READ_S` seconds for at most one generation (:data:`GENERATION_S`).

    Raises:
        AssertionError: When the job failed first.
        TimeoutError: Naming `what`, when `reached` never held.
    """
    record: dict[str, Any] = {}

    def holds() -> bool:
        nonlocal record
        record = job(server, job_id)
        assert record["status"] != "failed", record
        return reached(record)

    server.wait_until(holds, GENERATION_S, what, interval_s=READ_S)
    return record


def completed_job(server: Server, job_id: str) -> dict[str, Any]:
    """The record of the video job `job_id` once it completed."""
    return wait_for_job(
        server,
        job_id,
        lambda record: record["status"] == "completed",
        f"video job {job_id} did not complete",
    )


def job_video(server: Server, job_id: str) -> httpx.Response:
    """The MP4 of the video job `job_id` (``GET /v1/videos/{id}/content``),
    once the job completed.
    """  # noqa: D205
    completed_job(server, job_id)
    response = httpx.get(f"{server.base_url}/v1/videos/{job_id}/content")
    response.raise_for_status()
    return response


#: The workers' graph dispatches by runtime mode (``graph_replay``,
#: ``graph_capture``, ``eager``), a counter of the observation store.
GRAPH_DISPATCHES = "uniserve_graph_dispatches_total"

#: How many successive reads, a second apart, must agree before the graph
#: dispatches count as settled.
SETTLED_READS = 3

#: How long the graph dispatches may take to settle, in seconds (scaled):
#: the workers' counters reach the observation store with the engine's
#: report, one each ``worker_config.report_period`` (a second by default),
#: so the calls of a finished request appear within a few periods.
SETTLE_S = 120.0


def graph_dispatches(server: Server, *modes: str) -> dict[str, float]:
    """The workers' graph dispatches of each of `modes`, as the observation
    store counts them once they stopped moving.

    The counters reach the store with the engine's periodic report, and no
    answer reports the dispatches of its own request, so the counts are read
    a second apart until :data:`SETTLED_READS` successive reads agree with
    the one before them.
    """  # noqa: D205
    reads: list[dict[str, float]] = []

    def settled() -> bool:
        reads.append(
            {
                mode: admin.counter(server, GRAPH_DISPATCHES, mode=mode)
                for mode in modes
            }
        )
        latest = reads[-SETTLED_READS - 1 :]
        return len(latest) > SETTLED_READS and all(
            counts == reads[-1] for counts in latest
        )

    server.wait_until(
        settled, SETTLE_S, "the workers' graph dispatches did not settle"
    )
    return reads[-1]
