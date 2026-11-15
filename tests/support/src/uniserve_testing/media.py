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
from typing import Any

import av
import httpx
import numpy as np
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from uniserve_eval.datasets.minimax_h3 import MiniMaxH3Dataset
from uniserve_eval.transport.video import inspect_video_bytes
from uniserve_eval.types import (
    BenchmarkPoint,
    DecodedVideo,
    LoadConfig,
    MetricDefinition,
    TaskName,
    VideoConfig,
)
from uniserve_testing import client
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
    tokenizer. Skips the calling case when no repository publishes the
    checkpoint.
    """  # noqa: D205
    if checkpoint.repository is None:
        pytest.skip(checkpoint.unpublished)
    folder = (
        f"{checkpoint.folder}/tokenizer" if checkpoint.folder else "tokenizer"
    )
    return AutoTokenizer.from_pretrained(
        checkpoint.repository, revision=checkpoint.revision, subfolder=folder
    )


def exact_prompt(tokenizer: PreTrainedTokenizerBase, tokens: int) -> str:
    """The evaluator's MiniMax-H3 benchmark prompt (its ``minimax-h3``
    dataset) cut or extended to exactly `tokens` tokens of `tokenizer`, so a
    request lands in the text capacity its length selects.
    """  # noqa: D205
    point = BenchmarkPoint(
        name="exact-prompt",
        server="uniserve",
        task=TaskName.VIDEO,
        model="MiniMax-H3",
        dataset=MiniMaxH3Dataset.name,
        # The dataset reads the prompt length alone; the metric a point must
        # protect is never read.
        metrics=(MetricDefinition(("videos_per_second",), "higher"),),
        load=LoadConfig(num_prompts=1),
        video=VideoConfig(prompt_tokens=tokens),
    )
    (example,) = MiniMaxH3Dataset(point).load(tokenizer)
    return example.prompt


def assert_video(
    data: bytes,
    content_type: str,
    *,
    size: str,
    frames: int,
    media_format: MediaFormat,
) -> DecodedVideo:
    """Hold the MP4 `data` to the shape its request resolved to and to its
    model's `media_format`, and return what its decoding found.

    The video is `size` (``WxH``) with `frames` frames, at the format's frame
    rate in its video codec; its soundtrack is in the format's codec, channel
    count and sample rate and lasts as long as the video, to within one
    frame; and neither stream is constant, as a generated video moves and
    sounds.
    """  # noqa: D205
    media = inspect_video_bytes(data, declared_mime=content_type)
    width, height = (int(side) for side in size.split("x"))
    assert (media.width, media.height) == (width, height), size
    assert media.frame_count == frames, size
    assert media.fps_numerator == media_format.fps * media.fps_denominator
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
    assert abs(media.audio_duration_s - frames * frame_s) <= frame_s
    assert media.video_variance > 0 and media.audio_rms > 0, size
    return media


def assert_generated(
    response: httpx.Response,
    *,
    size: str,
    frames: int,
    media_format: MediaFormat,
) -> DecodedVideo:
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
    return assert_video(
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
#: ``graph_capture``, ``eager``), a counter family of ``/metrics``.
GRAPH_DISPATCHES = "uniserve:worker_graph_runtime_mode_counts"

#: How many successive reads, a second apart, must agree before the graph
#: dispatches count as settled.
SETTLED_READS = 3

#: How long the graph dispatches may take to settle, in seconds (scaled):
#: the workers' counters reach ``/metrics`` with the engine's report, one
#: each ``worker_config.report_period`` (a second by default), so the calls
#: of a finished request appear within a few periods.
SETTLE_S = 120.0


def graph_dispatches(server: Server, *modes: str) -> dict[str, float]:
    """The workers' graph dispatches of each of `modes`, as ``/metrics``
    counts them once they stopped moving.

    The counters reach ``/metrics`` with the engine's periodic report, which
    states no sequence number, and no answer reports the dispatches of its
    own request, so the counts are read a second apart until
    :data:`SETTLED_READS` successive reads agree with the one before them.
    """  # noqa: D205
    reads: list[dict[str, float]] = []

    def settled() -> bool:
        reads.append(
            {
                mode: client.metric(
                    server.base_url, GRAPH_DISPATCHES, mode=mode
                )
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
