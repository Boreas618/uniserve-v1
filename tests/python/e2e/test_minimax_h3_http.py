"""MiniMax-H3 serves text-to-video, and so does a FastH3 adapter fused over it.

Serves the diffusers root's ``transformer`` on the published Ulysses-4
deployment, serving the 768p size of every aspect ratio the model names, and
requests one 124-frame ``t2va`` video at each. Every response is a complete
MP4 at the requested size whose headers report the generated shape; the
capabilities report the handshake's schedule, and a body naming a task is
refused. With ``--lora-dir`` naming a FastH3 adapter's directory, the same
root serves the student's schedule, and a fixed-seed request answers the same
video twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from uniserve_testing import assets, media
from uniserve_testing.deployments import Deployments

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
]

PROMPT = (
    "A red fox trots across a snowy meadow at sunrise, its breath steaming, "
    "while birds call from the pines."
)


@pytest.mark.gpus(4)
def test_base_text_to_video_at_every_768p_size(
    deployments: Deployments, repository: Path, tmp_path: Path
) -> None:
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
    with (
        deployments.serve(
            "--served-model-name",
            "MiniMax-H3",
            "--config",
            str(repository / "configs" / "minimax_h3" / "ulysses4.toml"),
            # Five seconds bound the layouts startup prepares: two frame
            # counts at each of the six sizes and one text capacity.
            "--set",
            f"video_config.sizes={json.dumps(sizes)}",
            "--set",
            "video_config.max_seconds=5",
            "--max-model-len",
            "1024",
            # The default prompt-capacity ladder holds capacities below the
            # admitted length only, so the deployment states its one
            # capacity.
            "--set",
            "video_config.text_capacities=[1024]",
            model=assets.CHECKPOINTS["minimax-h3"].model,
        ) as server,
        media.api(server) as api,
    ):
        video = api.get("/v1/capabilities").json()["video"]["model"]
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
        named = api.post(
            "/v1/videos/sync",
            json={
                **media.t2va_request(
                    "MiniMax-H3", PROMPT, 124, 42, size=sizes[0]
                ),
                "task": "t2va",
            },
        )
        assert named.status_code == 400
        assert "task" in named.json()["error"]["message"]

        for size in sizes:
            # Each size states a prompt of its own: a request whose prompt's
            # encoder output is committed already pins it and runs no
            # encoding call, so it reports no `preparing` phase, and every
            # request below is to report every phase.
            response = api.post(
                "/v1/videos/sync",
                json=media.t2va_request(
                    "MiniMax-H3",
                    f"{PROMPT} The frame is {size}.",
                    124,
                    42,
                    size=size,
                ),
            )
            media.assert_generated(
                response, size=size, frames=124, media_format=media.MINIMAX_H3
            )
            (tmp_path / f"{size}.mp4").write_bytes(response.content)

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


@pytest.mark.gpus(4)
def test_a_fasth3_adapter_over_the_base_release_serves_its_student(
    deployments: Deployments, repository: Path, tmp_path: Path
) -> None:
    adapter = assets.adapter_directory(tmp_path / "adapter")
    with (
        deployments.serve(
            "--served-model-name",
            "FastH3",
            "--lora-dir",
            str(adapter),
            "--config",
            str(repository / "configs" / "minimax_h3" / "ulysses4.toml"),
            "--set",
            'video_config.sizes=["1344x768"]',
            "--set",
            "video_config.max_seconds=5",
            "--max-model-len",
            "1024",
            "--set",
            "video_config.text_capacities=[1024]",
            model=assets.CHECKPOINTS["minimax-h3"].model,
        ) as server,
        media.api(server) as api,
    ):
        video = api.get("/v1/capabilities").json()["video"]["model"]
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
            response = api.post(
                "/v1/videos/sync",
                json=media.t2va_request(
                    "FastH3", PROMPT, 107, 42, size="1344x768"
                ),
            )
            media.assert_generated(
                response,
                size="1344x768",
                frames=107,
                media_format=media.MINIMAX_H3,
            )
            (tmp_path / f"fasth3-{attempt}.mp4").write_bytes(response.content)
            answers.append(response.content)
        # The fused weights are the deployment's for every request: a
        # fixed-seed request is answered with the same video again.
        assert answers[0] == answers[1]
