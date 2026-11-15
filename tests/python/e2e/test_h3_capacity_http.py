"""Startup-captured capacity layouts serve every admitted H3 request shape."""

from __future__ import annotations

from pathlib import Path

import pytest
from uniserve_testing import assets, media
from uniserve_testing.deployments import Deployments

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
]

#: The canvas every request asks for, one of the default deployment's.
SIZE = "1344x768"
# Every output length the [4, 15] second API range produces at 24 fps.
FRAME_COUNTS = tuple(107 + 17 * index for index in range(16))
# Prompt lengths around and between the text capacities: the listed 1024 and
# the prompt capacity `--max-model-len`, 16384.
PROMPT_TOKENS = (1, 777, 1024, 1025, 3000, 16384, 9000, 64)
TEXT_CAPACITIES = "[1024]"


def _request(prompt: str, seconds: float, seed: int) -> dict[str, object]:
    """A t2va request at :data:`SIZE` for a duration the server aligns to
    frames.
    """  # noqa: D205
    return {
        "model": "FastH3",
        "prompt": prompt,
        "size": SIZE,
        "seconds": seconds,
        "seed": seed,
    }


@pytest.mark.gpus(4)
def test_every_admitted_duration_and_prompt_length_replays(
    deployments: Deployments, repository: Path
) -> None:
    """Serve the complete admitted range from startup-captured graphs.

    Startup captures every admitted duration at two text capacities. Each
    of the 16 output lengths is then requested once, at a fractional
    duration, with prompt lengths at, around and between the capacities.
    Every request returns complete media of its aligned length, and serving
    captures nothing. A denoiser that captures refuses to step eagerly, so a
    served request is one whose eight denoising steps replayed. Two requests
    of different layouts then run concurrently on the two request slots, and
    each matches its own isolated execution.
    """
    tokenizer = media.tokenizer(assets.CHECKPOINTS["fasth3"])

    with (
        deployments.serve(
            "--served-model-name",
            "FastH3",
            "--config",
            str(repository / "configs" / "minimax_h3" / "ulysses4.toml"),
            "--max-running-requests",
            "2",
            "--max-model-len",
            "16384",
            "--set",
            "video_config.max_seconds=15",
            "--set",
            f"video_config.text_capacities={TEXT_CAPACITIES}",
            model=assets.CHECKPOINTS["fasth3"].model,
        ) as server,
        media.api(server) as api,
    ):
        capabilities = api.get("/v1/capabilities").json()["video"]["model"]
        assert (
            capabilities["min_seconds"],
            capabilities["max_seconds"],
            capabilities["model_max_seconds"],
        ) == (4.0, 15.0, 15.0)

        for index, frames in enumerate(FRAME_COUNTS):
            # Eight frames short of the output length, which extends to it.
            seconds = (frames - 8) / 24
            tokens = PROMPT_TOKENS[index % len(PROMPT_TOKENS)]
            before = media.graph_dispatches(
                server, "graph_capture", "graph_replay"
            )
            response = api.post(
                "/v1/videos/sync",
                json=_request(
                    media.exact_prompt(tokenizer, tokens), seconds, 2000 + index
                ),
            )
            media.assert_generated(
                response,
                size=SIZE,
                frames=frames,
                media_format=media.MINIMAX_H3,
            )
            after = media.graph_dispatches(
                server, "graph_capture", "graph_replay"
            )
            assert after["graph_capture"] == before["graph_capture"], (
                seconds,
                tokens,
            )
            # Eight denoising steps, the text encoder and the refiner.
            assert after["graph_replay"] - before["graph_replay"] >= 10, (
                seconds,
                tokens,
            )

        # Two layouts in flight together keep their own slots and state.
        payloads = [
            _request(media.exact_prompt(tokenizer, 500), 7.3, 3001),
            _request(media.exact_prompt(tokenizer, 9000), 12.2, 3002),
        ]
        ids = [
            api.post("/v1/videos", json=payload).json()["id"]
            for payload in payloads
        ]
        for payload, job_id in zip(payloads, ids, strict=True):
            content = media.job_video(server, job_id)
            isolated = api.post("/v1/videos/sync", json=payload)
            isolated.raise_for_status()
            media.assert_same_media(
                content.content,
                isolated.content,
                tolerance=media.BF16_COMPOSITION,
            )
