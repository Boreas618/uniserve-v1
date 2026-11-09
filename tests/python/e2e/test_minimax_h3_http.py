"""The MiniMax-H3 base checkpoint serves text-to-video at every 768p size.

Serves the diffusers root's ``transformer`` on the published Ulysses-4
deployment, serving the 768p size of every aspect ratio the model names, and
requests one 124-frame ``t2va`` video at each. Every response is a complete
MP4 at the requested size whose headers report the generated shape; the
capabilities report the handshake's schedule, and a body naming a task is
refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    MINIMAX_H3,
    find_free_port,
    model_dir,
    require_uniserve_binary,
    server_process,
    t2va_request,
    worker_executable,
)
from uniserve_eval.transport.video import inspect_video_bytes

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("minimax_h3")]


def test_base_text_to_video_at_every_768p_size(tmp_path: Path) -> None:
    model = str(model_dir(MINIMAX_H3))
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
    port = find_free_port()
    base = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        model,
        "--worker-groups",
        str(deployment),
        "--worker-executable",
        str(worker_executable()),
        "--served-model-name",
        "MiniMax-H3",
        "--port",
        str(port),
        # Five seconds bound the layouts startup prepares: two frame counts
        # at each of the six sizes and one text capacity.
        "--set",
        f"video_config.sizes={','.join(sizes)}",
        "--set",
        "video_config.max_seconds=5",
        "--max-model-len",
        "1024",
    ]
    prompt = (
        "A red fox trots across a snowy meadow at sunrise, its breath "
        "steaming, while birds call from the pines."
    )
    with (
        server_process(command, base, tmp_path / "server.log", timeout_s=3600),
        httpx.Client(base_url=base, timeout=1800) as client,
    ):
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
            response = client.post(
                "/v1/videos/sync",
                json=t2va_request("MiniMax-H3", prompt, 124, 42, size=size),
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
                "encoding",
                "preparing",
                "denoising",
                "decoding",
                "finalizing",
            }, size
            assert min(stages.values()) >= 0
            assert stages["denoising"] > stages["decoding"] > 0
            inference = float(response.headers["x-inference-time-s"])
            assert abs(sum(stages.values()) - inference) <= 0.01, size
