"""Complete media and cancellation across statically assigned H3 blocks."""

from __future__ import annotations

import http.client
import json
import urllib.parse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
from uniserve_testing import admin, assets, client, media
from uniserve_testing.deployments import Deployments, two_host_leader
from uniserve_testing.remote import RemoteHost
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
]

#: The GPUs every case leases on this host, and that a two-host case's
#: layout addresses on each of its hosts.
HOST_GPUS = 4

#: The canvas every request asks for, one of the default deployment's.
SIZE = "1344x768"

#: The request shapes the layouts serve, as (seconds, prompt tokens, frames):
#: the shortest admitted duration at the smallest text capacity, and the
#: longest at the prompt capacity, `--max-model-len`.
SHORT = (5, 1000, 124)
LONG = (15, 16384, 362)

# Each deployment below starts its own server, which takes most of its time,
# so the layouts are one per way of dividing the denoiser: the sequence
# (Ulysses, all-gather, both), the tensors, the layer range, and tensors or
# layers combined with the sequence. Other degrees of the same division run
# the same code.
#
# Four-rank layouts place the denoiser on four devices and the text encoder
# and decoders beside it, as the default deployment does.
FOUR_RANK_PARALLEL_CONFIGS = {
    "gather4": {
        "sequence_parallel": {"kind": "allgather", "allgather_degree": 4},
    },
    "hybrid": {
        "sequence_parallel": {
            "kind": "hybrid",
            "ulysses_degree": 2,
            "allgather_degree": 2,
        },
    },
    "tensor4": {"tensor_parallel_size": 4},
    "tensor2_ulysses2": {
        "tensor_parallel_size": 2,
        "sequence_parallel": {"kind": "ulysses", "ulysses_degree": 2},
    },
    "pipeline4": {"pipeline_parallel_size": 4},
}

# Eight-rank layouts place every numerical block on all eight devices of two
# hosts of four.
EIGHT_RANK_PARALLEL_CONFIGS = {
    "ulysses8": {
        "sequence_parallel": {"kind": "ulysses", "ulysses_degree": 8},
    },
    "gather8": {
        "sequence_parallel": {"kind": "allgather", "allgather_degree": 8},
    },
    "hybrid_ulysses4_gather2": {
        "sequence_parallel": {
            "kind": "hybrid",
            "ulysses_degree": 4,
            "allgather_degree": 2,
        },
    },
    "tensor8": {"tensor_parallel_size": 8},
    "tensor2_ulysses4": {
        "tensor_parallel_size": 2,
        "sequence_parallel": {"kind": "ulysses", "ulysses_degree": 4},
    },
    "pipeline2_ulysses4": {
        "pipeline_parallel_size": 2,
        "sequence_parallel": {"kind": "ulysses", "ulysses_degree": 4},
    },
}

# Ranks reach devices through these orders rather than by identity, so a
# layout that assumed rank index equals device index would fail. The
# eight-rank order permutes device pairs and never the devices inside one,
# because a host can give peer access only within its own adjacent pairs: a
# Ulysses group of two occupies one consecutive rank pair, and splitting those
# across a host's pairs would test nothing about the layout while making every
# such layout unrunnable there.
FOUR_RANK_DEVICES = [3, 1, 2, 0]
EIGHT_RANK_DENOISER_RANKS = [2, 3, 0, 1, 6, 7, 4, 5]


def _sequence_layout(degree: int) -> tuple[list[int], dict[str, Any]]:
    """The devices and blocks of a layout whose denoiser divides the
    sequence over `degree` ranks (Ulysses; none for one rank), beside a text
    encoder of the same tensor degree and the decoders, on the first
    `degree` devices of :data:`FOUR_RANK_DEVICES`.
    """  # noqa: D205
    ranks = list(range(degree))
    sequence = (
        {"kind": "ulysses", "ulysses_degree": degree}
        if degree > 1
        else {"kind": "local"}
    )
    return FOUR_RANK_DEVICES[:degree], {
        "transformer": {
            "ranks": ranks,
            "parallel_config": {"sequence_parallel": sequence},
        },
        "text_encoder": {
            "ranks": ranks,
            "parallel_config": {"tensor_parallel_size": degree},
        },
        "video_decoder": {
            "ranks": ranks,
            "parallel_config": {"distribution": "temporal_units"},
        },
        "audio_decoder": {"ranks": [0]},
    }


def _four_rank_layout(
    parallel_config: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    """The devices and blocks of a four-rank layout of the denoiser, whose
    ranks reach the devices in the order of :data:`FOUR_RANK_DEVICES`.
    """  # noqa: D205
    return [0, 1, 2, 3], {
        "transformer": {
            "ranks": FOUR_RANK_DEVICES,
            "parallel_config": parallel_config,
        },
        "text_encoder": {"ranks": [0]},
        "video_decoder": {
            "ranks": [2, 0],
            "parallel_config": {"distribution": "temporal_units"},
        },
        "audio_decoder": {"ranks": [1]},
    }


def _eight_rank_layout(
    parallel_config: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    """The devices and blocks of an eight-rank layout, every block on all
    eight devices.
    """  # noqa: D205
    ranks = list(range(8))
    return ranks, {
        "transformer": {
            "ranks": EIGHT_RANK_DENOISER_RANKS,
            "parallel_config": parallel_config,
        },
        "text_encoder": {
            "ranks": ranks,
            "parallel_config": {"tensor_parallel_size": 8},
        },
        "video_decoder": {
            "ranks": ranks,
            "parallel_config": {"distribution": "temporal_units"},
        },
        "audio_decoder": {"ranks": [0]},
    }


#: Every layout of one host's four GPUs, by name.
ONE_HOST_LAYOUTS = {
    "local": _sequence_layout(1),
    "ulysses4": _sequence_layout(4),
    **{
        name: _four_rank_layout(config)
        for name, config in FOUR_RANK_PARALLEL_CONFIGS.items()
    },
}

#: Every layout of two hosts' four GPUs each, by name.
TWO_HOST_LAYOUTS = {
    name: _eight_rank_layout(config)
    for name, config in EIGHT_RANK_PARALLEL_CONFIGS.items()
}


def _workers(
    devices: list[int],
    blocks: dict[str, Any],
    grouping: str,
    place: Callable[[int], str],
) -> list[dict[str, Any]]:
    """The worker groups of a layout: its numerical `blocks`, whose ranks
    reach `devices` by index, and the blocks every FastH3 deployment places
    beside them, in one group (``whole``) or a group per block (``split``),
    each group's ranks on the devices `place` names.
    """  # noqa: D205
    blocks = dict(blocks)
    # A FastH3 student also serves fl2va, so every deployment places the
    # latent encoder that encodes its keyframes, here on the first video
    # decoder rank; the requests below are text-to-video and never use it.
    blocks["latent_encoder"] = {"ranks": blocks["video_decoder"]["ranks"][:1]}
    # The media blocks encode and assemble on the host inside the main
    # process, which serves CUDA devices only, so they are placed on CUDA
    # ranks as the deployment files of `configs/minimax_h3` place them: each
    # decoded unit's codec slot on its decoder's rank, and the muxer and the
    # media reader on rank 0, which every layout places on the leader's host.
    blocks["video_codec"] = {
        "ranks": blocks["video_decoder"]["ranks"],
        "parallel_config": {"distribution": "temporal_units"},
    }
    blocks["muxer"] = {"ranks": [0]}
    blocks["media_reader"] = {"ranks": [0]}
    groups = {
        "whole": [tuple(blocks)],
        "split": [(name,) for name in blocks],
    }[grouping]
    workers = []
    for names in groups:
        members = sorted(
            {rank for name in names for rank in blocks[name]["ranks"]}
        )
        workers.append(
            {
                "id": names[0],
                "ranks": [place(devices[rank]) for rank in members],
                "queue_depth": 6,
                "blocks": {
                    name: {
                        "ranks": [
                            members.index(rank)
                            for rank in blocks[name]["ranks"]
                        ],
                        **blocks[name].get("parallel_config", {}),
                    }
                    for name in names
                },
            }
        )
    return workers


def _options(
    directory: Path,
    workers: list[dict[str, Any]],
    max_running_requests: int,
    shapes: tuple[tuple[int, int, int], ...],
) -> list[str]:
    """The options of a deployment of `workers` admitting
    `max_running_requests` requests of up to the longest of `shapes`.
    """  # noqa: D205
    max_seconds = max(seconds for seconds, _, _ in shapes)
    return [
        "--served-model-name",
        "MiniMax-H3",
        "--config",
        str(client.written_deployment(directory, workers)),
        "--max-running-requests",
        str(max_running_requests),
        "--max-model-len",
        "16384",
        "--set",
        f"video_config.max_seconds={max_seconds}",
        # Two text capacities bound the startup capture of every admitted
        # duration; the requests below exercise both.
        "--set",
        "video_config.text_capacities=[1024]",
        # One and two device layouts hold the whole denoiser on each of their
        # ranks and split the 15-second sequence over fewer of them, so their
        # warmup reaches a larger share of the device than the four-device
        # deployment the default share is sized for.
        "--mem-fraction-static",
        "0.92",
        "--set",
        'dtype="bfloat16"',
        "--set",
        'quantization.mode="balanced"',
    ]


def _payloads(shapes: tuple[tuple[int, int, int], ...]) -> list[dict]:
    """A text-to-video request of each of `shapes`, whose prompt is exactly
    the shape's prompt tokens long, so each request lands in its intended
    text capacity.
    """  # noqa: D205
    tokenizer = media.tokenizer(assets.CHECKPOINTS["fasth3"])
    return [
        media.t2va_request(
            "MiniMax-H3",
            media.exact_prompt(tokenizer, tokens),
            frames,
            1000,
            size=SIZE,
        )
        for _seconds, tokens, frames in shapes
    ]


def _disconnect(server: Server, payload: dict) -> None:
    """Send `payload` to ``POST /v1/videos/sync`` and close the connection
    while the engine generates it, as a client that goes away does.

    The engine announces no admission of a request, so the deployment's
    in-flight count (``GET /admin/status``) is read until it holds the
    request; the connection then closes before any answer arrives.
    """  # noqa: D205
    address = urllib.parse.urlsplit(server.base_url)
    connection = http.client.HTTPConnection(address.hostname, address.port)
    try:
        connection.request(
            "POST",
            "/v1/videos/sync",
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        server.wait_until(
            lambda: admin.status(server)["deployment"]["in_flight"] > 0,
            media.GENERATION_S,
            "the engine did not take the request",
            interval_s=media.READ_S,
        )
    finally:
        connection.close()


def _exercise(
    server: Server,
    payloads: list[dict],
    shapes: tuple[tuple[int, int, int], ...],
    directory: Path,
) -> None:
    """Serve complete media of every shape, then serve again after more
    disconnected requests than the deployment's request slots.
    """  # noqa: D205
    with media.api(server) as api:
        # Warm both decode layouts before exercising cancellation.
        for payload, (seconds, _tokens, frames) in zip(
            payloads, shapes, strict=True
        ):
            response = api.post("/v1/videos/sync", json=payload)
            media.assert_generated(
                response,
                size=SIZE,
                frames=frames,
                media_format=media.MINIMAX_H3,
            )
            (directory / f"{seconds}s.mp4").write_bytes(response.content)

        # Three disconnects exceed the two provisioned slots and require
        # their reuse.
        _seconds, _tokens, frames = shapes[0]
        for _ in range(3):
            _disconnect(server, payloads[0])
            response = api.post("/v1/videos/sync", json=payloads[0])
            media.assert_generated(
                response,
                size=SIZE,
                frames=frames,
                media_format=media.MINIMAX_H3,
            )


# A worker group per block carries every product edge between groups. The
# Ulysses-4 deployment admits one request: the worker's two state slots
# follow its queue depth, not the number of admitted requests, so one
# admitted request still leaves two. The whole denoiser on one device is
# served at the short shape alone: its long shape's activation working set
# does not fit beside it on one device.
@pytest.mark.gpus(HOST_GPUS)
@pytest.mark.parametrize(
    ("layout", "grouping", "max_running_requests", "shapes"),
    [
        pytest.param("local", "whole", 2, (SHORT,), id="local-whole"),
        *(
            pytest.param(name, "whole", 2, (SHORT, LONG), id=f"{name}-whole")
            for name in FOUR_RANK_PARALLEL_CONFIGS
        ),
        pytest.param(
            "ulysses4", "split", 1, (SHORT, LONG), id="ulysses4-split"
        ),
    ],
)
def test_block_bindings_release_cancelled_requests(
    deployments: Deployments,
    tmp_path: Path,
    layout: str,
    grouping: str,
    max_running_requests: int,
    shapes: tuple[tuple[int, int, int], ...],
) -> None:
    """A layout of one host's four GPUs serves complete media of each shape,
    and serves again after more disconnected requests than its slots.
    """  # noqa: D205
    devices, blocks = ONE_HOST_LAYOUTS[layout]
    workers = _workers(devices, blocks, grouping, lambda d: f"cuda:{d}")
    payloads = _payloads(shapes)

    with deployments.serve(
        *_options(tmp_path, workers, max_running_requests, shapes),
        model=assets.CHECKPOINTS["fasth3"].model,
    ) as server:
        _exercise(server, payloads, shapes, tmp_path)


@pytest.mark.gpus(HOST_GPUS)
@pytest.mark.parametrize(
    ("layout", "grouping"),
    [
        *((name, "whole") for name in EIGHT_RANK_PARALLEL_CONFIGS),
        ("tensor2_ulysses4", "split"),
    ],
)
def test_block_bindings_on_two_hosts_release_cancelled_requests(
    deployments: Deployments,
    peer_host: RemoteHost,
    repository: Path,
    tmp_path: Path,
    layout: str,
    grouping: str,
) -> None:
    """An eight-rank layout over two hosts of four GPUs serves complete media
    of each shape, and serves again after more disconnected requests than
    its slots.

    Its worker groups lie on all eight devices, four on each host: a product
    its ranks on one host write and its ranks on the other read is copied
    between the hosts by the reading host's Storage Server
    (docs/minimax_h3/minimax_h3.md, Two hosts). The second host's daemon
    joins this host's leader as node ``rank-1`` and runs the ranks placed on
    its devices.
    """  # noqa: D205
    shapes = (SHORT, LONG)
    devices, blocks = TWO_HOST_LAYOUTS[layout]

    def place(device: int) -> str:
        # The ranks past this host's devices run on the second host's.
        if device < HOST_GPUS:
            return f"cuda:{device}"
        return f"rank-1/cuda:{device - HOST_GPUS}"

    workers = _workers(devices, blocks, grouping, place)
    payloads = _payloads(shapes)

    with (
        deployments.serve(
            *_options(tmp_path, workers, 2, shapes),
            *two_host_leader(peer_host),
            model=assets.CHECKPOINTS["fasth3"].model,
            serving=False,
        ) as server,
        deployments.follower(peer_host, server, repository),
    ):
        _exercise(server, payloads, shapes, tmp_path)


@pytest.mark.gpus(HOST_GPUS)
def test_replicas_on_two_hosts_serve_complete_media(
    deployments: Deployments, peer_host: RemoteHost, repository: Path
) -> None:
    """`fasth3-ulysses4x2-two-host.toml` serves across two hosts of four GPUs.

    Each Ulysses4 replica, with its text encoder, decoders, codecs, muxer and
    media reader, lies on one host, so no product passes between the hosts'
    main processes, and the server places each request's stages on the
    replicas of one host (docs/minimax_h3/minimax_h3.md, Two hosts). Twice as
    many concurrent requests as the deployment admits reach both hosts'
    replicas, and each answer is complete media.
    """
    # Two resident requests on each host's replica. One five-second duration
    # and one text capacity keep the startup capture small.
    max_running_requests = 4
    payloads = [
        media.t2va_request(
            "FastH3",
            f"A river at dawn, seen from bank {index}.",
            124,
            1000 + index,
            size=SIZE,
        )
        for index in range(2 * max_running_requests)
    ]
    deployment = (
        repository
        / "configs"
        / "minimax_h3"
        / "fasth3-ulysses4x2-two-host.toml"
    )

    with (
        deployments.serve(
            "--served-model-name",
            "FastH3",
            "--config",
            str(deployment),
            "--max-running-requests",
            str(max_running_requests),
            "--max-model-len",
            "1024",
            "--set",
            "video_config.max_seconds=5",
            "--set",
            "video_config.text_capacities=[1024]",
            *two_host_leader(peer_host),
            model=assets.CHECKPOINTS["fasth3"].model,
            serving=False,
        ) as server,
        deployments.follower(peer_host, server, repository),
        media.api(server) as api,
        ThreadPoolExecutor(max_workers=len(payloads)) as pool,
    ):

        def generate(payload: dict) -> httpx.Response:
            return api.post("/v1/videos/sync", json=payload)

        for response in pool.map(generate, payloads):
            media.assert_generated(
                response, size=SIZE, frames=124, media_format=media.MINIMAX_H3
            )


@pytest.mark.gpus(HOST_GPUS)
def test_video_jobs_retain_content_and_cancel_active_work(
    deployments: Deployments, repository: Path
) -> None:
    """Exercise async ownership and reuse through the HTTP interface.

    The exercise runs on the documented four-device deployment, whose
    blocks, the media ones included, share one worker group.
    """
    with (
        deployments.serve(
            "--served-model-name",
            "FastH3",
            "--config",
            str(repository / "configs" / "minimax_h3" / "ulysses4.toml"),
            # A media deployment states the prompt length it admits; one
            # text capacity of that length bounds the startup capture of
            # every admitted duration.
            "--max-model-len",
            "1024",
            "--set",
            "video_config.text_capacities=[1024]",
            # Full-duration decoder graphs beside BF16 weights and two
            # resident request slots need more than the default share.
            "--mem-fraction-static",
            "0.92",
            model=assets.CHECKPOINTS["fasth3"].model,
        ) as server,
        media.api(server) as api,
    ):
        caps = api.get("/v1/capabilities").json()
        assert caps["model"] == "FastH3"
        video = caps["video"]
        assert video["model"]["tasks"] == ["t2va", "fl2va"]
        # The schedule counts denoiser evaluations: one per trained rung of
        # the eight-step student.
        steps = video["model"]["num_inference_steps"]
        assert steps == 8
        # The request API's fields, whatever the model.
        assert set(video["request_fields"]) == {
            "model",
            "prompt",
            "negative_prompt",
            "size",
            "seconds",
            "num_frames",
            "fps",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "priority",
            "input_references",
            "extra_params",
        }
        river = media.t2va_request("FastH3", "A river", 124, 1001, size=SIZE)
        for payload in (
            # Longer than the deployment's default 15-second capacity.
            {"model": "FastH3", "prompt": "A river", "seconds": 16},
            # A frame count the model does not generate.
            {**river, "num_frames": 125},
            # Sizes the default deployment (1344x768 and 768x1344) does not
            # serve.
            {**river, "size": "768x768"},
            {**river, "size": "848x480"},
            # A duration and a frame count together.
            {**river, "seconds": 5},
            # A field the request API does not have.
            {**river, "task": "t2va"},
        ):
            response = api.post("/v1/videos", json=payload)
            assert response.status_code == 400, (payload, response.text)
        payload = media.t2va_request(
            "FastH3",
            "A river flows through a forest, with birds singing.",
            124,
            1001,
            size=SIZE,
        )
        # Multipart fields carry the same names as text parts.
        response = api.post(
            "/v1/videos",
            files={name: (None, str(value)) for name, value in payload.items()},
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        job = media.completed_job(server, job_id)
        # The job reports the generated shape; its seconds are the generated
        # duration as the OpenAI-style string.
        assert job["size"] == SIZE
        assert job["num_frames"] == 124 and job["fps"] == 24
        assert float(job["seconds"]) == pytest.approx(124 / 24, abs=1e-3)
        assert job["completed_steps"] == job["total_steps"] == steps
        assert job["expires_at"] > job["completed_at"]
        assert job_id in {
            item["id"] for item in api.get("/v1/videos").json()["data"]
        }
        first = api.get(f"/v1/videos/{job_id}/content")
        second = api.get(f"/v1/videos/{job_id}/content")
        assert first.content == second.content
        media.assert_video(
            first.content,
            first.headers["content-type"],
            size=SIZE,
            frames=124,
            media_format=media.MINIMAX_H3,
        )
        assert api.delete(f"/v1/videos/{job_id}").json()["deleted"]
        assert api.get(f"/v1/videos/{job_id}/content").status_code == 404

        # Concurrent request storage must preserve each prompt and seed.
        # Compare both artifacts with isolated executions through the same
        # public route.
        concurrent = [
            {**payload, "seed": 1010},
            {
                **payload,
                "prompt": (
                    "A train crosses a bridge at sunrise, with birds singing."
                ),
                "seed": 1011,
            },
        ]
        ids = [
            api.post("/v1/videos", json=item).json()["id"]
            for item in concurrent
        ]
        for item, concurrent_id in zip(concurrent, ids, strict=True):
            content = media.job_video(server, concurrent_id)
            isolated = api.post("/v1/videos/sync", json=item)
            isolated.raise_for_status()
            media.assert_same_media(
                content.content,
                isolated.content,
                tolerance=media.BF16_COMPOSITION,
            )

        # Cancel more requests than the two resident slots, including a
        # genuinely active job.
        for seed in range(3):
            active = api.post(
                "/v1/videos",
                json={**payload, "num_frames": 362, "seed": seed},
            ).json()["id"]
            media.wait_for_job(
                server,
                active,
                lambda record: record["status"] == "in_progress",
                "the job never reached execution",
            )
            queued = api.post(
                "/v1/videos", json={**payload, "seed": seed + 100}
            ).json()["id"]
            assert api.delete(f"/v1/videos/{active}").status_code == 200
            assert api.delete(f"/v1/videos/{queued}").status_code == 200
            assert api.get(f"/v1/videos/{active}").status_code == 404
        reused = api.post("/v1/videos", json={**payload, "seed": 1002}).json()[
            "id"
        ]
        media.job_video(server, reused)
        sync = api.post("/v1/videos/sync", json={**payload, "seed": 1003})
        media.assert_generated(
            sync, size=SIZE, frames=124, media_format=media.MINIMAX_H3
        )
