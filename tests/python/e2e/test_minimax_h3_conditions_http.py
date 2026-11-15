"""MiniMax-H3 checkpoints serve keyframe and reference requests.

Serves the published Ulysses-4 deployments and sends the seed-42 requests
of the packaged conditioned workloads (``uniserve_eval/workloads/minimax_h3``):
the official first-frame request (W3) on the base release's ``transformer``
deployment and on a FastH3 DMD student of it, and the image-plus-audio (W4)
and official video-plus-audio (W5) reference requests on the
``transformer_ref`` deployment, with the base release's reference DiT and
with a FastH3 OmniRef export.
An OmniRef deployment that states a smaller condition capacity than the
image-plus-audio request takes refuses it, naming its rows, and then serves
the image-only request that fits. Every deployment serves under CUDA
graphs: a request with conditions denoises in a captured layout widened to
the deployment's condition capacity, and the encoding of reference media
whose shape no graph holds runs eagerly. Reference media are read from
``file://`` URLs under the deployment's ``media_config.directory``. Every
served response is a complete MP4 at the size and frame count the request
resolves to.
"""

from __future__ import annotations

import importlib.resources
import json
import shutil
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest
from uniserve_testing import assets, media
from uniserve_testing.deployments import Deployments
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
]

#: The evaluator's packaged MiniMax-H3 workloads.
WORKLOADS = (
    importlib.resources.files("uniserve_eval") / "workloads" / "minimax_h3"
)

#: The reference media the workloads name, by their paths in it.
INPUTS = assets.CHECKPOINTS["minimax-h3-inputs"]

#: The size every request below resolves to: the 16:9 keyframe's canvas, and
#: the first size the reference deployments serve.
SIZE = "1344x768"


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    """The case's media directory (``media_config.directory``), by its
    canonical path, into which :func:`_request` copies the reference media
    its requests read.
    """  # noqa: D205
    directory = (tmp_path / "inputs").resolve()
    directory.mkdir()
    return directory


def _request(workload: str, inputs: Path) -> dict[str, Any]:
    """The seed-42 request of a packaged referenced workload, its reference
    media copied into the media directory `inputs`.

    A workload's warmup manifest holds its seed-42 request: the prompt, the
    frame count and the reference media in request order, named by their
    paths in :data:`INPUTS`. The server reads a ``file://`` reference only at
    a canonical path inside its media directory, and the Hugging Face cache
    holds a file as a link into its blob store, so each medium is copied
    from the cache into `inputs`. The request states no size, so the
    server's own default holds: the first keyframe's canvas for fl2va, the
    deployment's first served size for ref2va.
    """  # noqa: D205
    manifest = WORKLOADS / f"{workload}-warmup.jsonl"
    row = json.loads(manifest.read_text().splitlines()[0])
    references = []
    for reference in row["input_references"]:
        path = inputs / reference["media"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(INPUTS.file(reference["media"]), path)
        references.append(
            {
                **{
                    key: value
                    for key, value in reference.items()
                    if key != "media"
                },
                "url": f"file://{path}",
            }
        )
    return {
        "model": "MiniMax-H3",
        "prompt": row["prompt"],
        "num_frames": row["num_frames"],
        "input_references": references,
        "seed": row["seed"],
    }


def _serve(
    deployments: Deployments,
    repository: Path,
    checkpoint: str,
    deployment: str,
    inputs: Path,
    *options: str,
) -> AbstractContextManager[Server]:
    """The deployment file `deployment` of ``configs/minimax_h3`` serving
    the checkpoint `checkpoint` (:data:`assets.CHECKPOINTS`) with `options`,
    reading reference media from `inputs`.
    """  # noqa: D205
    return deployments.serve(
        "--served-model-name",
        "MiniMax-H3",
        "--config",
        str(repository / "configs" / "minimax_h3" / deployment),
        "--set",
        f'media_config.directory="{inputs}"',
        *options,
        model=assets.CHECKPOINTS[checkpoint].model,
    )


# The official first-frame presentation holds 1935 text rows and its
# keyframe 1008 condition rows; without a stated condition capacity the
# deployment provisions the two keyframes a request may bring.
FIRST_FRAME_OPTIONS = (
    "--set",
    "video_config.max_seconds=8",
    "--max-model-len",
    "2048",
    "--set",
    "video_config.text_capacities=[2048]",
)


@pytest.mark.gpus(4)
def test_base_first_frame_request(
    deployments: Deployments, repository: Path, inputs: Path, tmp_path: Path
) -> None:
    request = _request("w3", inputs)
    with (
        _serve(
            deployments,
            repository,
            "minimax-h3",
            "ulysses4.toml",
            inputs,
            *FIRST_FRAME_OPTIONS,
        ) as server,
        media.api(server) as api,
    ):
        video = api.get("/v1/capabilities").json()["video"]["model"]
        assert set(video["tasks"]) == {"t2va", "fl2va"}

        response = api.post("/v1/videos/sync", json=request)
        (tmp_path / "fl2va.mp4").write_bytes(response.content)
        # The 16:9 keyframe resolves the 1344x768 canvas of the 192 frames
        # the row states.
        media.assert_generated(
            response, size=SIZE, frames=192, media_format=media.MINIMAX_H3
        )


@pytest.mark.gpus(4)
def test_fasth3_first_frame_request(
    deployments: Deployments, repository: Path, inputs: Path, tmp_path: Path
) -> None:
    """A FastH3 DMD student serves its partition's keyframe requests."""
    request = _request("w3", inputs)
    # The student generates 16:9 at 768p; eight seconds and a 2048-token
    # prompt capacity admit W3, and the deployment provisions the two
    # keyframes a request may bring in tiles of their own.
    with (
        _serve(
            deployments,
            repository,
            "fasth3",
            "ulysses4.toml",
            inputs,
            "--set",
            f'video_config.sizes=["{SIZE}"]',
            *FIRST_FRAME_OPTIONS,
        ) as server,
        media.api(server) as api,
    ):
        video = api.get("/v1/capabilities").json()["video"]["model"]
        assert video["tasks"] == ["t2va", "fl2va"]

        response = api.post("/v1/videos/sync", json=request)
        (tmp_path / "fl2va.mp4").write_bytes(response.content)
        media.assert_generated(
            response, size=SIZE, frames=192, media_format=media.MINIMAX_H3
        )


def _reference_requests(
    server: Server, requests: dict[str, dict[str, Any]], directory: Path
) -> None:
    """Send W4 and W5 (`requests`, by workload) to the reference deployment
    `server`; each serves complete media whose denoising steps replay
    captured graphs.
    """  # noqa: D205
    with media.api(server) as api:
        video = api.get("/v1/capabilities").json()["video"]["model"]
        assert video["tasks"] == ["ref2va"]
        steps = video["num_inference_steps"]

        for name, request in requests.items():
            before = media.graph_dispatches(server, "graph_replay")[
                "graph_replay"
            ]
            response = api.post("/v1/videos/sync", json=request)
            (directory / f"{name}.mp4").write_bytes(response.content)
            # References take the deployment's first served size, 1344x768,
            # at the 124 frames the row states.
            media.assert_generated(
                response, size=SIZE, frames=124, media_format=media.MINIMAX_H3
            )
            after = media.graph_dispatches(server, "graph_replay")[
                "graph_replay"
            ]
            assert after - before >= steps, name


# The official video-plus-audio presentation holds 6913 text rows and its
# references about 38k condition rows.
REFERENCE_OPTIONS = (
    "--set",
    "video_config.max_seconds=5",
    "--max-model-len",
    "8192",
    "--set",
    "video_config.text_capacities=[8192]",
    "--set",
    "video_config.max_condition_rows=40960",
)


@pytest.mark.gpus(4)
def test_base_reference_requests(
    deployments: Deployments, repository: Path, inputs: Path, tmp_path: Path
) -> None:
    requests = {name: _request(name, inputs) for name in ("w4", "w5")}
    with _serve(
        deployments,
        repository,
        "minimax-h3",
        "ulysses4-reference.toml",
        inputs,
        *REFERENCE_OPTIONS,
    ) as server:
        _reference_requests(server, requests, tmp_path)


# The OmniRef reference DiT packs each condition into whole 128-row tiles,
# a video's token grid in tiles of 4 latent frames by 4 by 8 tokens, so the
# same video-plus-audio request takes 47,104 rows there: its 37 x 24 x 42
# reference video fills 10 x 6 x 6 tiles and its soundtrack and audio
# reference 4 tiles each, against 38,124 rows packed densely.
OMNIREF_OPTIONS = (
    "--set",
    "video_config.max_seconds=5",
    "--max-model-len",
    "8192",
    "--set",
    "video_config.text_capacities=[8192]",
    "--set",
    "video_config.max_condition_rows=49152",
)


@pytest.mark.gpus(4)
def test_omniref_reference_requests(
    deployments: Deployments, repository: Path, inputs: Path, tmp_path: Path
) -> None:
    # The OmniRef export reads the parts it omits from the Hugging Face
    # cache at the base revision it pins.
    requests = {name: _request(name, inputs) for name in ("w4", "w5")}
    with _serve(
        deployments,
        repository,
        "fasth3-omniref",
        "ulysses4-reference.toml",
        inputs,
        *OMNIREF_OPTIONS,
    ) as server:
        _reference_requests(server, requests, tmp_path)


# The image-plus-audio request (W4) packs 7,808 condition rows on OmniRef when
# references are enlarged to the released pipeline's encoding size: its
# 2560x1440 reference image, enlarged to a 2048-pixel short edge, fills 57
# whole 128-row tiles (7,296 rows of video latents) and the five-second voice
# four (two tracks of 207 rows, 414 rows of audio latents). This deployment
# states a 7,680-row condition capacity: each of W4's condition products fits
# it and their packed rows do not. W4's image without its voice fits.
SMALL_CAPACITY_OPTIONS = (
    "--set",
    "mm_processor_kwargs.enlarge_references=true",
    "--set",
    "video_config.max_seconds=5",
    "--max-model-len",
    "8192",
    "--set",
    "video_config.text_capacities=[8192]",
    "--set",
    "video_config.max_condition_rows=7680",
)


@pytest.mark.gpus(4)
def test_omniref_refuses_conditions_above_capacity(
    deployments: Deployments, repository: Path, inputs: Path, tmp_path: Path
) -> None:
    # Each of W4's condition products fits its bound, and the server counts
    # its conditions in the denoiser's tiles and refuses it; the deployment
    # then serves its image-only form.
    packed = _request("w4", inputs)
    image_only = {
        **packed,
        "input_references": packed["input_references"][:1],
    }
    with (
        _serve(
            deployments,
            repository,
            "fasth3-omniref",
            "ulysses4-reference.toml",
            inputs,
            *SMALL_CAPACITY_OPTIONS,
        ) as server,
        media.api(server) as api,
    ):
        refused = api.post("/v1/videos/sync", json=packed)
        assert refused.status_code == 400, refused.text
        error = refused.json()["error"]
        assert error["type"] == "invalid_request_error", error
        message = error["message"]
        assert "7808" in message and "7680" in message, message
        assert "video_config.max_condition_rows" in message, message

        served = api.post("/v1/videos/sync", json=image_only)
        (tmp_path / "image_only.mp4").write_bytes(served.content)
        media.assert_generated(
            served, size=SIZE, frames=124, media_format=media.MINIMAX_H3
        )
