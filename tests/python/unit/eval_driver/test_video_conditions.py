"""Video requests with references reach every backend as the same work.

A row names its reference media relative to the point's reference root.
UniServe receives its body's ``input_references`` with ``file://`` URLs,
SGLang the same URLs as native H3 conditions, vLLM-Omni the same files as
uploaded parts, and FastVideo their local paths.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from PIL import Image

from tests.python.fixtures.http_stub import stub_server
from uniserve_eval.metrics import summarize
from uniserve_eval.tasks.video import VideoTask
from uniserve_eval.transport.client import send_request
from uniserve_eval.types import (
    VIDEOS_SYNC,
    BenchmarkPoint,
    DecodedVideo,
    Example,
    MetricDefinition,
    TaskName,
    VideoConfig,
    VideoShape,
)

pytestmark = pytest.mark.unit

BASE_SCHEDULE = {
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}
# FastH3 OmniRef: eight fused-block forwards over nine sigma points.
OMNIREF_POINTS = 9
VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42 reference video"
AUDIO_BYTES = b"ID3 reference voice"


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    """A reference root holding one file of each media type."""
    media = tmp_path / "inputs" / "media"
    media.mkdir(parents=True)
    Image.new("RGB", (64, 36), (200, 40, 40)).save(media / "keyframe.png")
    Image.new("RGB", (48, 64), (40, 200, 40)).save(media / "subject.png")
    (media / "clip.mp4").write_bytes(VIDEO_BYTES)
    (media / "voice.mp3").write_bytes(AUDIO_BYTES)
    return tmp_path / "inputs"


def point(inputs: Path, backend: str = "uniserve", **video):
    """A 1344x768, 24 fps point measuring ``backend`` with every field."""
    settings = {}
    if backend in ("sglang", "vllm-omni"):
        settings = {
            **BASE_SCHEDULE,
            "aspect_ratio": "auto",
            "duration_seconds": 5.0,
        }
    elif backend == "fastvideo":
        settings = {
            "num_inference_steps": BASE_SCHEDULE["num_inference_steps"],
            "duration_seconds": 5.0,
        }
    return BenchmarkPoint(
        name="referenced",
        server="video",
        task=TaskName.VIDEO,
        model="MiniMax-H3",
        dataset="jsonl",
        endpoint=VIDEOS_SYNC,
        metrics=(MetricDefinition(("videos_per_second",), "higher"),),
        video=VideoConfig(
            size="1344x768",
            fps=24,
            audio_sample_rate=32000,
            audio_channels=2,
            backend=backend,
            reference_root=str(inputs),
            **{**settings, **video},
        ),
    )


# The protocol's referenced workloads in miniature: a first keyframe (W3),
# an image with a voice (W4), and a video with its soundtrack and a voice
# (W5).
FIRST_FRAME = [
    {"type": "image", "media": "media/keyframe.png", "frame_index": 0}
]
IMAGE_AUDIO = [
    {"type": "image", "media": "media/subject.png"},
    {"type": "audio", "media": "media/voice.mp3"},
]
VIDEO_AUDIO = [
    {"type": "video", "media": "media/clip.mp4"},
    {"type": "audio", "media": "media/voice.mp3"},
]
WORKLOADS = pytest.mark.parametrize(
    "task,references",
    [("fl2va", FIRST_FRAME), ("ref2va", IMAGE_AUDIO), ("ref2va", VIDEO_AUDIO)],
    ids=["first-frame", "image-audio", "video-audio"],
)


def row(references, **fields):
    """A 124-frame row at seed 7 with ``references``."""
    return Example(
        "row",
        "scene",
        num_frames=124,
        seed=7,
        input_references=references,
        **fields,
    )


def send(request):
    """Send one request to a recording server and return what it received."""
    sent = []

    async def exercise():
        async def handler(received):
            sent.append((received.headers, await received.read()))
            return web.Response(
                body=b"mp4", headers={"content-type": "video/mp4"}
            )

        async with (
            stub_server(handler) as base_url,
            aiohttp.ClientSession() as client,
        ):
            record = await send_request(
                client, base_url, request, "row", task="video"
            )
        assert record.success
        return record

    record = asyncio.run(exercise())
    return sent[0], record


def multipart(request):
    """Decode a multipart request into its text fields and file parts."""
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + request[0]["content-type"].encode()
        + b"\r\n\r\n"
        + request[1]
    )
    fields, files = {}, []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        if filename is None:
            fields[name] = part.get_payload(decode=True).decode()
        else:
            files.append(
                (
                    name,
                    filename,
                    part.get_content_type(),
                    part.get_payload(decode=True),
                )
            )
    return fields, files


@WORKLOADS
def test_uniserve_receives_references_by_file_url(inputs, task, references):
    request = VideoTask(point(inputs)).build_request(row(references))
    received, record = send(request)

    expected = []
    for reference in references:
        item = {
            "type": reference["type"],
            "url": (inputs / reference["media"]).resolve().as_uri(),
        }
        if "frame_index" in reference:
            item["frame_index"] = reference["frame_index"]
        expected.append(item)
    # The body names no task: the server derives it from the references.
    assert json.loads(received[1]) == {
        "model": "MiniMax-H3",
        "prompt": "scene",
        "size": "1344x768",
        "num_frames": 124,
        "fps": 24,
        "seed": 7,
        "input_references": expected,
    }
    assert record.video_shape == VideoShape(1344, 768, 124, 24)


@WORKLOADS
def test_sglang_receives_the_same_files_as_native_conditions(
    inputs, task, references
):
    request = VideoTask(point(inputs, "sglang")).build_request(row(references))
    received, _ = send(request)

    body = json.loads(received[1])
    assert body["task"] == task
    assert body["target"] == {
        "short_edge": 768,
        "aspect_ratio": "auto",
        "duration_seconds": 5.0,
    }
    expected = []
    for reference in references:
        condition = {
            "type": reference["type"],
            "uri": (inputs / reference["media"]).resolve().as_uri(),
            # An anchor is a keyframe; any other item a global reference.
            "role": "keyframe" if "frame_index" in reference else "reference",
        }
        if "frame_index" in reference:
            condition["frame_index"] = reference["frame_index"]
        expected.append(condition)
    assert body["conditions"] == expected


@pytest.mark.parametrize(
    "task,references,kinds",
    [
        ("fl2va", FIRST_FRAME, ["image/png"]),
        ("ref2va", IMAGE_AUDIO, ["image/png", "audio/mpeg"]),
        ("ref2va", VIDEO_AUDIO, ["video/mp4", "audio/mpeg"]),
    ],
    ids=["first-frame", "image-audio", "video-audio"],
)
def test_vllm_omni_receives_the_same_files_as_uploaded_references(
    inputs, task, references, kinds
):
    request = VideoTask(point(inputs, "vllm-omni")).build_request(
        row(references)
    )
    received, _ = send(request)

    fields, files = multipart(received)
    assert [(name, kind) for name, _, kind, _ in files] == [
        ("input_references", kind) for kind in kinds
    ]
    assert [data for *_, data in files] == [
        (inputs / reference["media"]).read_bytes() for reference in references
    ]
    assert (int(fields["width"]), int(fields["height"])) == (1344, 768)
    assert int(fields["num_frames"]) == 124
    assert int(fields["num_inference_steps"]) == 49
    extra = json.loads(fields["extra_params"])
    assert extra["task"] == task
    assert extra["duration"] == 5.0
    # Keyframes carry their frame indices; references need none.
    assert extra.get("frame_indices") == ([0] if task == "fl2va" else None)


@pytest.mark.parametrize(
    "references,typed",
    [
        (
            IMAGE_AUDIO,
            {
                "image_reference": [{"image_url": "media/subject.png"}],
                "audio_reference": [{"audio_url": "media/voice.mp3"}],
            },
        ),
        (
            VIDEO_AUDIO,
            {
                "video_reference": [{"video_url": "media/clip.mp4"}],
                "audio_reference": [{"audio_url": "media/voice.mp3"}],
            },
        ),
    ],
    ids=["image-audio", "video-audio"],
)
def test_fastvideo_receives_reference_paths_and_its_block_count(
    inputs, references, typed
):
    config = point(
        inputs,
        "fastvideo",
        num_inference_steps=OMNIREF_POINTS,
        parallel_decoding=True,
    )
    request = VideoTask(config).build_request(row(references))
    received, _ = send(request)

    body = json.loads(received[1])
    for field, entries in typed.items():
        assert body[field] == [
            {key: str((inputs / path).resolve())}
            for entry in entries
            for key, path in entry.items()
        ]
    assert body["task"] == "ref2va"
    assert body["size"] == "1344x768"
    assert body["num_frames"] == 124
    # A parallel decoding student is asked for its fused-block forwards.
    assert body["num_inference_steps"] == OMNIREF_POINTS - 1


@pytest.mark.parametrize(
    "references,message",
    [
        # A reference states no role: an anchor is an item with frame_index.
        (
            [{"type": "image", "media": "media/subject.png", "role": "x"}],
            "unknown keys",
        ),
        ([{"type": "video_audio", "media": "media/clip.mp4"}], "unknown type"),
        ([{"type": "audio", "media": "media/clip.mp4"}], "is not audio media"),
        (
            [
                {
                    "type": "image",
                    "media": "media/keyframe.png",
                    "frame_index": 0.5,
                }
            ],
            "frame_index",
        ),
    ],
)
def test_a_reference_the_request_api_cannot_carry_is_not_sent(
    inputs, references, message
):
    with pytest.raises(ValueError, match=message):
        VideoTask(point(inputs)).build_request(row(references))


def test_references_need_the_points_reference_root(inputs):
    config = replace(
        point(inputs),
        video=replace(point(inputs).video, reference_root=None),
    )
    with pytest.raises(ValueError, match="reference_root"):
        VideoTask(config).build_request(row(IMAGE_AUDIO))


@pytest.mark.parametrize("backend", ["sglang", "vllm-omni", "fastvideo"])
def test_a_baseline_refuses_a_keyframe_no_h3_task_takes(inputs, backend):
    """A baseline names H3's task, and no task anchors a video."""
    settings = (
        {"num_inference_steps": OMNIREF_POINTS, "parallel_decoding": True}
        if backend == "fastvideo"
        else {}
    )
    anchored = [{"type": "video", "media": "media/clip.mp4", "frame_index": 0}]
    with pytest.raises(ValueError, match="anchor"):
        VideoTask(point(inputs, backend, **settings)).build_request(
            row(anchored)
        )


@pytest.mark.parametrize("backend", ["vllm-omni", "fastvideo"])
def test_a_type_grouping_backend_refuses_references_it_would_reorder(
    inputs, backend
):
    """Audio before an image would be presented after it there.

    The request is refused when it is built, before any measurement.
    """
    settings = (
        {"num_inference_steps": OMNIREF_POINTS, "parallel_decoding": True}
        if backend == "fastvideo"
        else {}
    )
    task = VideoTask(point(inputs, backend, **settings))
    with pytest.raises(ValueError, match="images, then videos, then audio"):
        task.build_request(row(IMAGE_AUDIO[::-1]))


def test_fastvideo_refuses_a_reference_start_offset_it_cannot_take(inputs):
    config = point(
        inputs,
        "fastvideo",
        num_inference_steps=OMNIREF_POINTS,
        parallel_decoding=True,
    )
    offset = [{**VIDEO_AUDIO[0], "start_time_seconds": 1.5}, VIDEO_AUDIO[1]]
    with pytest.raises(ValueError, match="start offset"):
        VideoTask(config).build_request(row(offset))


@pytest.mark.parametrize("endpoint", [VIDEOS_SYNC, "/v1/videos"])
def test_server_reported_timings_are_recorded_and_summarized(inputs, endpoint):
    """Timings come from sync headers or from the completed job."""
    stages = {"denoising_stage": 30.5, "video_decoding_stage": 2.25}

    async def exercise():
        async def handler(request):
            if request.method == "POST" and endpoint == "/v1/videos":
                return web.json_response({"id": "job", "status": "queued"})
            if request.method == "GET" and not request.path.endswith(
                "/content"
            ):
                return web.json_response(
                    {
                        "id": "job",
                        "status": "completed",
                        "inference_time_s": 33.0,
                        "stage_durations": stages,
                        "peak_storage_mb": 91000.5,
                    },
                )
            headers = {"content-type": "video/mp4"}
            if endpoint == VIDEOS_SYNC:
                headers |= {
                    "x-inference-time-s": "33.0",
                    "x-stage-durations": json.dumps(stages),
                    "x-peak-memory-mb": "91000.5",
                }
            return web.Response(body=b"mp4", headers=headers)

        config = point(
            inputs,
            "fastvideo",
            num_inference_steps=OMNIREF_POINTS,
            parallel_decoding=True,
            poll_interval_s=0.001,
        )
        config = replace(config, endpoint=endpoint)
        request = VideoTask(config).build_request(row(IMAGE_AUDIO))
        async with (
            stub_server(handler) as base_url,
            aiohttp.ClientSession() as client,
        ):
            return await send_request(
                client, base_url, request, "row", task="video"
            )

    record = asyncio.run(exercise())
    assert record.server_inference_s == 33.0
    assert record.server_stage_s == stages
    assert record.server_peak_storage_mib == 91000.5
    assert record.record_dict()["server_stage_s"] == stages

    # Summaries cover the requests whose output decoded.
    record.decoded_video = DecodedVideo(
        data=b"",
        sha256="",
        byte_size=3,
        mime="video/mp4",
        width=1344,
        height=768,
        frame_count=124,
        fps_numerator=24,
        fps_denominator=1,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=32000,
        audio_samples=round(124 / 24 * 32000),
        sample_filename="sample.mp4",
    )
    metrics = summarize([record], 40.0)
    assert metrics["server_inference_ms"]["p50"] == 33_000.0
    assert metrics["server_stage_ms"]["denoising_stage"]["p50"] == 30_500.0
    assert metrics["server_peak_storage_mib"] == 91000.5


@pytest.mark.parametrize("endpoint", [VIDEOS_SYNC, "/v1/videos"])
def test_stage_durations_alone_are_recorded_for_a_uniserve_request(
    inputs, endpoint
):
    """A server may report its stage durations without the other timings."""
    stages = {"encoding": 4.5, "denoising": 30.0, "decoding": 2.0}

    async def exercise():
        async def handler(request):
            if request.method == "POST" and endpoint == "/v1/videos":
                return web.json_response({"id": "job", "status": "queued"})
            if request.method == "GET" and not request.path.endswith(
                "/content"
            ):
                return web.json_response(
                    {
                        "id": "job",
                        "status": "completed",
                        "stage_durations": stages,
                    },
                )
            headers = {"content-type": "video/mp4"}
            if endpoint == VIDEOS_SYNC:
                headers["x-stage-durations"] = json.dumps(stages)
            return web.Response(body=b"mp4", headers=headers)

        config = replace(
            point(inputs, poll_interval_s=0.001), endpoint=endpoint
        )
        request = VideoTask(config).build_request(row(VIDEO_AUDIO))
        async with (
            stub_server(handler) as base_url,
            aiohttp.ClientSession() as client,
        ):
            return await send_request(
                client, base_url, request, "row", task="video"
            )

    record = asyncio.run(exercise())
    assert record.server_stage_s == stages
    assert record.server_inference_s is None
    assert record.server_peak_storage_mib is None
