"""MiniMax-H3 serves text-to-video, and so does a FastH3 adapter fused over it.

Serves the diffusers root's ``transformer`` on the published Ulysses-4
deployment, serving the 768p size of every aspect ratio the model names, and
requests one 124-frame ``t2va`` video at each. Every response is a complete
MP4 at the requested size whose headers report the generated shape; the
capabilities report the handshake's schedule, and a body naming a task is
refused. With ``--lora-path`` naming a FastH3 adapter, the same root serves the
student's schedule, and a fixed-seed request answers the same video twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    FASTH3_ADAPTER,
    MINIMAX_H3_CHECKPOINT,
    engine_selection,
    host_asset,
    require_uniserve_binary,
    server_process,
    t2va_request,
)
from uniserve_eval.transport.video import inspect_video_bytes

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    # Every deployment of the module serves on the host's four GPUs.
    pytest.mark.gpus(4),
]


def test_base_text_to_video_at_every_768p_size(tmp_path: Path) -> None:
    model = str(host_asset("UNISERVE_MINIMAX_H3_MODEL", MINIMAX_H3_CHECKPOINT))
    # The 768p canvas of each aspect ratio MiniMax-H3 names: 21:9, 16:9, 4:3,
    # 1:1, 3:4 and 9:16.
    sizes = [
        "1536x672",
        "1344x768",
        "1024x768",
        "768x768",
        "768x1024",
        "768x1344",
    ]
    deployment = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "minimax_h3"
        / "ulysses4.toml"
    )
    command = [
        str(require_uniserve_binary()),
        "serve",
        "--model-dir",
        model,
        "--config",
        str(deployment),
        *engine_selection(),
        "--served-model-name",
        "MiniMax-H3",
        # Five seconds bound the layouts startup prepares: two frame counts
        # at each of the six sizes and one text capacity.
        "--set",
        f"video_config.sizes={json.dumps(list(sizes))}",
        "--set",
        "video_config.max_seconds=5",
        "--max-model-len",
        "1024",
        # The default prompt-capacity ladder holds capacities below the
        # admitted length only, so the deployment states its one capacity.
        "--set",
        "video_config.text_capacities=[1024]",
    ]
    prompt = (
        "A red fox trots across a snowy meadow at sunrise, its breath "
        "steaming, while birds call from the pines."
    )
    with (
        server_process(
            command, tmp_path / "server.log", timeout_s=3600
        ) as server,
        httpx.Client(timeout=1800) as client,
    ):
        base = server.base_url
        client.base_url = base
        video = client.get("/v1/capabilities").json()["video"]["model"]
        # The base DiT generates from the prompt alone and from keyframes.
        assert video["tasks"] == ["t2va", "fl2va"]
        # The released schedule's 50 sigma points are 49 denoiser evaluations.
        assert video["num_inference_steps"] == 49
        assert video["schedule"] == {
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        }

        # The request API derives the task from the references; a body that
        # names one carries an unknown field.
        named = client.post(
            "/v1/videos/sync",
            json={
                **t2va_request("MiniMax-H3", prompt, 124, 42),
                "task": "t2va",
            },
        )
        assert named.status_code == 400
        assert "task" in named.json()["error"]["message"]

        for size in sizes:
            width, height = (int(side) for side in size.split("x"))
            # Each size states a prompt of its own: a request whose prompt's
            # encoder output is committed already pins it and runs no
            # encoding call, so it reports no `preparing` phase, and every
            # request below is to report every phase.
            response = client.post(
                "/v1/videos/sync",
                json=t2va_request(
                    "MiniMax-H3",
                    f"{prompt} The frame is {size}.",
                    124,
                    42,
                    size=size,
                ),
            )
            assert response.status_code == 200, (size, response.text)
            assert (
                response.headers["x-video-size"],
                response.headers["x-video-num-frames"],
                response.headers["x-video-fps"],
            ) == (size, "124", "24")
            media = inspect_video_bytes(
                response.content,
                declared_mime=response.headers["content-type"],
            )
            (tmp_path / f"{size}.mp4").write_bytes(response.content)
            assert (media.width, media.height) == (width, height), size
            assert media.frame_count == 124, size
            assert media.fps_numerator == 24 * media.fps_denominator
            assert media.video_codec == "h264"
            assert (media.audio_codec, media.audio_channels) == ("aac", 2)
            assert media.audio_sample_rate == 32_000
            assert abs(media.audio_duration_s - 124 / 24) <= 1 / 24
            assert media.video_variance > 0 and media.audio_rms > 0, size

            # The server reports each generation phase it observed, from
            # scheduling to completion; the phases tile the inference time.
            stages = json.loads(response.headers["x-stage-durations"])
            assert set(stages) == {
                "scheduled",
                "preparing",
                "denoising",
                "decoding",
                "finalizing",
            }, size
            assert min(stages.values()) >= 0
            assert stages["denoising"] > stages["decoding"] > 0
            inference = float(response.headers["x-inference-time-s"])
            assert abs(sum(stages.values()) - inference) <= 0.01, size


def test_a_fasth3_adapter_over_the_base_release_serves_its_student(
    tmp_path: Path,
) -> None:
    model = str(host_asset("UNISERVE_MINIMAX_H3_MODEL", MINIMAX_H3_CHECKPOINT))
    adapter = str(host_asset("UNISERVE_FASTH3_ADAPTER", FASTH3_ADAPTER))
    deployment = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "minimax_h3"
        / "ulysses4.toml"
    )
    command = [
        str(require_uniserve_binary()),
        "serve",
        "--model-dir",
        model,
        "--lora-path",
        adapter,
        "--config",
        str(deployment),
        *engine_selection(),
        "--served-model-name",
        "FastH3",
        "--set",
        'video_config.sizes=["1344x768"]',
        "--set",
        "video_config.max_seconds=5",
        "--max-model-len",
        "1024",
        "--set",
        "video_config.text_capacities=[1024]",
    ]
    prompt = (
        "A red fox trots across a snowy meadow at sunrise, its breath "
        "steaming, while birds call from the pines."
    )
    with (
        server_process(
            command, tmp_path / "server.log", timeout_s=3600
        ) as server,
        httpx.Client(timeout=1800) as client,
    ):
        base = server.base_url
        client.base_url = base
        video = client.get("/v1/capabilities").json()["video"]["model"]
        # The student is distilled from the base partition, which serves
        # text-to-video and keyframes, on four DMD rungs at the base shifts.
        assert video["tasks"] == ["t2va", "fl2va"]
        assert video["num_inference_steps"] == 4
        assert video["schedule"] == {
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        }

        # 107 frames are 4.46 seconds at 24 fps, inside the five seconds the
        # deployment prepares.
        answers = []
        for attempt in range(2):
            response = client.post(
                "/v1/videos/sync",
                json=t2va_request("FastH3", prompt, 107, 42, size="1344x768"),
            )
            assert response.status_code == 200, (attempt, response.text)
            media = inspect_video_bytes(
                response.content,
                declared_mime=response.headers["content-type"],
            )
            (tmp_path / f"fasth3-{attempt}.mp4").write_bytes(response.content)
            assert (media.width, media.height, media.frame_count) == (
                1344,
                768,
                107,
            )
            assert media.video_variance > 0 and media.audio_rms > 0
            answers.append(response.content)
        # The fused weights are the deployment's for every request: a
        # fixed-seed request is answered with the same video again.
        assert answers[0] == answers[1]
