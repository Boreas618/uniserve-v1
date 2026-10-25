"""Complete media and cancellation across statically assigned H3 components."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Iterator
from itertools import zip_longest
from pathlib import Path

import av
import httpx
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    t2va_request,
    worker_executable,
    written_deployment,
)
from uniserve_eval.datasets.minimax_h3 import MiniMaxH3Dataset
from uniserve_eval.transport.video import inspect_video_bytes
from uniserve_eval.types import (
    BenchmarkPoint,
    LoadConfig,
    MetricDefinition,
    TaskName,
    VideoConfig,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("minimax_h3")]


def _assert_media_values_close(actual: bytes, expected: bytes) -> None:
    """Compare decoded media within the BF16 model-composition error budget."""
    for kind in ("video", "audio"):
        with (
            av.open(io.BytesIO(actual)) as observed,
            av.open(io.BytesIO(expected)) as reference,
        ):
            for frame, wanted in zip_longest(
                observed.decode(**{kind: 0}), reference.decode(**{kind: 0})
            ):
                assert frame is not None and wanted is not None, (
                    f"{kind}: frame count mismatch"
                )
                assert frame.time == wanted.time, (
                    f"{kind}: presentation time mismatch"
                )
                if kind == "video":
                    values = (
                        frame.to_ndarray(format="rgb24").astype(np.float32)
                        / 255
                    )
                    reference_values = (
                        wanted.to_ndarray(format="rgb24").astype(np.float32)
                        / 255
                    )
                else:
                    values = frame.to_ndarray()
                    reference_values = wanted.to_ndarray()
                np.testing.assert_allclose(
                    values,
                    reference_values,
                    rtol=2e-2,
                    atol=2e-2,
                    equal_nan=False,
                )


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

# Eight-rank layouts place every numerical component on all eight devices,
# which is how an eight-device host, or two hosts of four, serve.
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

# Ranks reach devices through this order rather than by identity, so a layout
# that assumed rank index equals device index would fail. The order permutes
# device pairs and never the devices inside one, because a host can give peer
# access only within its own adjacent pairs: a Ulysses group of two occupies one
# consecutive rank pair, and splitting those across a host's pairs would test
# nothing about the layout while making every such layout unrunnable there.
EIGHT_RANK_DENOISER_RANKS = [2, 3, 0, 1, 6, 7, 4, 5]

# A layout needing more devices than this host has runs on two hosts when
# UNISERVE_H3_FOLLOWER names the second (an ssh destination whose checkout
# path holds the same build) and UNISERVE_H3_LEADER_ADDRESS the address of
# this host that the second reaches. The second host's daemon joins this
# host's leader as node `rank-1`, as docs/minimax_h3/minimax_h3.md's two-host
# deployments do, and runs the ranks placed on its devices.
FOLLOWER = os.environ.get("UNISERVE_H3_FOLLOWER")
LEADER_ADDRESS = os.environ.get("UNISERVE_H3_LEADER_ADDRESS")
AWAITING = "awaiting the deployment's other nodes"


@contextlib.contextmanager
def _follower(leader_log: Path, log: Path, address: str) -> Iterator[None]:
    """Start the second host's daemon once the leader logged that it awaits
    it, joining at ``address``; on exit wait until it ended, which it does
    when the leader's connection closes.
    """  # noqa: D205
    assert FOLLOWER is not None
    environment = " ".join(
        f"{name}={shlex.quote(os.environ.get(name, ''))}"
        for name in ("PATH", "LD_LIBRARY_PATH")
    )
    remote = (
        f"cd {shlex.quote(str(Path.cwd()))} && echo follower pid $$ && "
        f"exec env {environment} NCCL_NVLS_ENABLE=0 "
        f"{shlex.quote(str(require_uniserve_binary()))} serve --nnodes 2 "
        f"--node-rank 1 --dist-init-addr {address} --host-identity rank-1"
    )
    started: list[subprocess.Popen[str]] = []
    leaving = threading.Event()

    def start() -> None:
        while not leaving.is_set():
            if leader_log.exists() and AWAITING in leader_log.read_text(
                errors="replace"
            ):
                started.append(
                    subprocess.Popen(
                        ["ssh", "-o", "BatchMode=yes", FOLLOWER, remote],
                        stdout=log.open("w", encoding="utf-8"),
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                )
                return
            time.sleep(1)

    starter = threading.Thread(target=start, daemon=True)
    starter.start()
    try:
        yield
    finally:
        leaving.set()
        starter.join()
        for process in started:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pid = log.read_text().split("follower pid ")[1].split()[0]
                subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", FOLLOWER, f"kill {pid}"],
                    check=False,
                )
                process.wait(timeout=120)
                pytest.fail("the follower daemon outlived its leader")


def _eight_rank_worker_config(parallel_kind: str) -> dict:
    """Place an eight-rank layout's components on all eight devices."""
    ranks = list(range(8))
    return {
        "devices": ranks,
        "transformer": {
            "ranks": EIGHT_RANK_DENOISER_RANKS,
            "parallel_config": EIGHT_RANK_PARALLEL_CONFIGS[parallel_kind],
        },
        "text_encoder": {
            "ranks": ranks,
            "parallel_config": {"tensor_parallel_size": 8},
        },
        "video_decoder": {
            "ranks": ranks,
        },
        "audio_decoder": {"ranks": [0]},
    }


@pytest.mark.parametrize(
    ("parallel_kind", "grouping", "max_running_requests"),
    [
        (kind, "whole", 2)
        for kind in (
            "local",
            *FOUR_RANK_PARALLEL_CONFIGS,
            *EIGHT_RANK_PARALLEL_CONFIGS,
        )
    ]
    # A worker group per component carries every product edge between
    # groups. The Ulysses-4 deployment admits one request: the worker's two
    # state slots follow its queue depth, not the number of admitted
    # requests, so one admitted request still leaves two.
    + [("ulysses4", "split", 1), ("tensor2_ulysses4", "split", 2)],
)
def test_component_bindings_release_cancelled_requests(
    tmp_path: Path,
    parallel_kind: str,
    grouping: str,
    max_running_requests: int,
) -> None:
    model_value = os.environ.get("UNISERVE_H3_MODEL")
    if not model_value or not Path(model_value).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name a supported FastH3 checkpoint "
            "directory; docs/minimax_h3/minimax_h3.md lists them"
        )
    # A layout that holds the whole denoiser on one device cannot also hold a
    # 15-second activation working set: the single-device warmup exhausts a
    # 184 GiB device with 178 GB allocated. That layout is qualified at a
    # five-second maximum; every other layout takes both shapes.
    shapes = (
        ((5, 1000, 124),)
        if parallel_kind == "local"
        else ((5, 1000, 124), (15, 16384, 362))
    )
    max_video_seconds = str(max(seconds for seconds, _, _ in shapes))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    if parallel_kind in ("local", "ulysses4"):
        degree = {"local": 1, "ulysses4": 4}[parallel_kind]
        ranks = list(range(degree))
        sequence = (
            {"kind": "local"}
            if degree == 1
            else {"kind": "ulysses", "ulysses_degree": degree}
        )
        worker_config = {
            "devices": [3, 1, 2, 0][:degree],
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
            },
            "audio_decoder": {"ranks": [0]},
        }
    elif parallel_kind in FOUR_RANK_PARALLEL_CONFIGS:
        # The denoiser's ranks reach the devices in the order 3, 1, 2, 0, so
        # a layout that assumed rank index equals device index would fail.
        worker_config = {
            "devices": [0, 1, 2, 3],
            "transformer": {
                "ranks": [3, 1, 2, 0],
                "parallel_config": FOUR_RANK_PARALLEL_CONFIGS[parallel_kind],
            },
            "text_encoder": {"ranks": [0]},
            "video_decoder": {
                "ranks": [2, 0],
            },
            "audio_decoder": {"ranks": [1]},
        }
    else:
        worker_config = _eight_rank_worker_config(parallel_kind)
    # A FastH3 student also serves fl2va, so every deployment places the
    # latent encoder that encodes its keyframes, here on the first video
    # decoder rank; the requests below are text-to-video and never use it.
    worker_config["latent_encoder"] = {
        "ranks": worker_config["video_decoder"]["ranks"][:1],
    }
    devices = worker_config.pop("devices")
    visible_devices = max(devices) + 1
    local_devices = torch.cuda.device_count()
    two_hosts = (
        local_devices < visible_devices <= 2 * local_devices
        and FOLLOWER is not None
        and LEADER_ADDRESS is not None
    )
    if local_devices < visible_devices and not two_hosts:
        pytest.skip(
            f"{parallel_kind} needs {visible_devices} devices; "
            f"this host has {local_devices} and no second host is named"
        )

    def place(device: int) -> str:
        # The ranks past this host's devices run on the second host's.
        if device < local_devices:
            return f"cuda:{device}"
        return f"rank-1/cuda:{device - local_devices}"

    # Mapping a product into a consumer's device needs peer access between the
    # two. A host that grants it only within its own device pairs cannot carry
    # device products between ranks that fall in different pairs, and there the
    # host mechanism is the one those edges admit.
    device_products = not two_hosts and all(
        first == second or torch.cuda.can_device_access_peer(first, second)
        for first in devices
        for second in devices
    )
    groups = {
        "whole": [tuple(worker_config)],
        "split": [(name,) for name in worker_config],
    }[grouping]
    workers = []
    owners = {}
    for names in groups:
        members = sorted(
            {rank for name in names for rank in worker_config[name]["ranks"]}
        )
        worker_id = names[0]
        components = {
            name: {
                "ranks": [
                    members.index(rank) for rank in worker_config[name]["ranks"]
                ],
                **worker_config[name].get("parallel_config", {}),
            }
            for name in names
        }
        workers.append(
            {
                "id": worker_id,
                "ranks": [place(devices[rank]) for rank in members],
                "queue_depth": 6,
                "components": components,
            }
        )
        owners.update(dict.fromkeys(names, worker_id))
    # Media units are encoded and the artifact assembled on the CPU workers
    # of the media group, as in the shipped deployments: each CPU worker is
    # one codec slot encoding one unit of a round, a further worker muxes and
    # the last reads condition media. On two hosts each unit's codec slot is
    # on its decoder's host.
    decoder_ranks = worker_config["video_decoder"]["ranks"]
    decoder_units = len(decoder_ranks)
    media_ranks = (
        [
            place(devices[rank]).split("cuda:")[0] + "cpu"
            for rank in decoder_ranks
        ]
        + ["cpu", "cpu"]
        if two_hosts
        else [f"cpu*{decoder_units + 2}"]
    )
    workers.append(
        {
            "id": "media",
            "ranks": media_ranks,
            "queue_depth": 6,
            "components": {
                "video_codec": {
                    "ranks": list(range(decoder_units)),
                },
                "muxer": {"ranks": [decoder_units]},
                "media_reader": {"ranks": [decoder_units + 1]},
            },
        }
    )
    owners.update(
        dict.fromkeys(("video_codec", "muxer", "media_reader"), "media")
    )
    edges = {
        (owners[source], owners[destination])
        for source, destination in (
            ("media_reader", "text_encoder"),
            ("media_reader", "latent_encoder"),
            ("text_encoder", "transformer"),
            ("latent_encoder", "transformer"),
            ("transformer", "video_decoder"),
            ("transformer", "audio_decoder"),
            ("video_decoder", "video_codec"),
            ("video_codec", "muxer"),
            ("audio_decoder", "muxer"),
        )
    }
    command = [
        str(require_uniserve_binary()),
        "serve",
        model_value,
        "--served-model-name",
        "MiniMax-H3",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
        "--worker-groups",
        str(written_deployment(tmp_path, workers)),
        "--max-running-requests",
        str(max_running_requests),
        "--max-model-len",
        "16384",
        "--set",
        f"video_config.max_seconds={max_video_seconds}",
        # Two text capacities bound the startup capture of every admitted
        # duration; the requests below exercise both.
        "--set",
        "video_config.text_capacities=1024",
        # One and two device layouts hold the whole denoiser on each of their
        # ranks and split the 15-second sequence over fewer of them, so their
        # warmup reaches a larger share of the device than the four-device
        # deployment the default grant is sized for.
        "--mem-fraction-static",
        "0.92",
        "--set",
        "dtype=bfloat16",
        "--set",
        "quantization_config.mode=balanced",
    ]
    if two_hosts:
        # Products cross the hosts by the mechanisms each edge derives, as in
        # the shipped two-host deployments.
        address = f"{LEADER_ADDRESS}:{find_free_port()}"
        command += [
            "--host-identity",
            "rank-0",
            "--nnodes",
            "2",
            "--dist-init-addr",
            address,
        ]
    else:
        command += [
            "--set",
            "placement_config.transfer="
            + ",".join(
                # Device products cross workers over VMM handles where the
                # devices can map one another; the media group's CPU workers
                # have no device, so their edges carry host products over
                # shared storage only.
                f"{source}->{destination}="
                + (
                    "posix_shm"
                    if "media" in (source, destination) or not device_products
                    else "cuda_vmm+posix_shm"
                )
                for source, destination in sorted(edges)
            ),
        ]
    tokenizer = AutoTokenizer.from_pretrained(Path(model_value) / "tokenizer")
    payloads = []
    for seconds, tokens, _frames in shapes:
        # The evaluator's FastH3 prompt synthesis yields exactly `tokens`
        # prompt tokens, so each request lands in its intended text capacity.
        case = BenchmarkPoint(
            name="fast-h3-prompt",
            server="uniserve",
            task=TaskName.VIDEO,
            model="FastH3",
            dataset="minimax-h3",
            # The point only synthesizes the prompt; its metric is never read.
            metrics=(MetricDefinition(("videos_per_second",), "higher"),),
            load=LoadConfig(num_prompts=1),
            video=VideoConfig(seconds=seconds, prompt_tokens=tokens),
        )
        prompt = MiniMaxH3Dataset(case).load(tokenizer)[0].prompt
        payloads.append(t2va_request("MiniMax-H3", prompt, seconds, 1000))
    log = tmp_path / "h3-components.log"
    environment = {
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(index) for index in range(min(visible_devices, local_devices))
        )
    }
    if two_hosts:
        # NVLink SHARP is off across hosts, as in the two-host measurements.
        environment["NCCL_NVLS_ENABLE"] = "0"
    follower = (
        _follower(log, tmp_path / "h3-follower.log", address)
        if two_hosts
        else contextlib.nullcontext()
    )
    with (
        follower,
        server_process(
            command,
            base_url,
            # Startup prepares and captures every admitted duration at both
            # text capacities; two-way tensor or pipeline layouts took 9 to
            # 11 minutes to become ready on 4 x GB200.
            log,
            timeout_s=1800,
            env=environment,
        ),
    ):
        # Warm both decode geometries before exercising cancellation. Three
        # disconnects exceed the two provisioned slots and require their reuse.
        for payload, (seconds, _tokens, frames) in zip(
            payloads, shapes, strict=True
        ):
            response = httpx.post(
                f"{base_url}/v1/videos/sync",
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
            media = inspect_video_bytes(
                response.content, declared_mime=response.headers["content-type"]
            )
            assert media.frame_count == frames
            # The 16:9 target's canvas.
            assert (media.width, media.height) == (1344, 768)
            assert (media.audio_channels, media.audio_sample_rate) == (2, 32000)
            (tmp_path / f"{seconds}s.mp4").write_bytes(response.content)
            print(
                f"{parallel_kind}/{grouping}: {seconds}s complete media passed",
                flush=True,
            )

        payload = payloads[0]
        for _ in range(3):
            with httpx.Client(timeout=httpx.Timeout(30, read=0.2)) as client:
                with pytest.raises(httpx.ReadTimeout):
                    client.post(f"{base_url}/v1/videos/sync", json=payload)
            response = httpx.post(
                f"{base_url}/v1/videos/sync", json=payload, timeout=600
            )
            response.raise_for_status()
            media = inspect_video_bytes(
                response.content, declared_mime=response.headers["content-type"]
            )
            assert media.frame_count == 124
            assert (media.audio_channels, media.audio_sample_rate) == (2, 32000)


def test_video_jobs_retain_content_and_cancel_active_work(
    tmp_path: Path,
) -> None:
    """Exercise async ownership and reuse through the HTTP contract.

    The exercise runs on the documented four-device deployment, whose numerical
    components share one worker group and whose media units are encoded and
    muxed on the CPU workers of the media group.
    """
    import time

    model = os.environ.get("UNISERVE_H3_MODEL")
    if not model or not Path(model).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name a supported FastH3 checkpoint "
            "directory; docs/minimax_h3/minimax_h3.md lists them"
        )
    inference = json.loads(
        (Path(model) / "fastvideo_inference.json").read_text()
    )
    expected_steps = len(inference["dmd_denoising_steps"])
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
        "FastH3",
        "--port",
        str(port),
        # Two text capacities bound the startup capture of every admitted
        # duration.
        "--set",
        "video_config.text_capacities=1024",
        # Full-duration decoder graphs need a larger static grant on H200
        # alongside BF16 weights and two resident request slots.
        "--mem-fraction-static",
        "0.92",
        "--set",
        "cuda_graph_config.policy=full",
    ]

    def completed(client, job_id):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            response = client.get(f"/v1/videos/{job_id}")
            response.raise_for_status()
            job = response.json()
            assert job["status"] != "failed", job
            if job["status"] == "completed":
                return job
            time.sleep(0.1)
        pytest.fail("video did not complete within the request deadline")

    with (
        server_process(
            command, base, tmp_path / "video-jobs.log", timeout_s=900
        ),
        httpx.Client(base_url=base, timeout=600) as client,
    ):
        caps = client.get("/v1/capabilities").json()
        assert caps["model"] == "FastH3"
        video = caps["video"]
        assert video["tasks"] == ["t2va", "fl2va"]
        assert video["task_conditions"] == {
            "t2va": {"conditions": []},
            "fl2va": {
                "keyframe_frame_indices": [[0], [-1], [0, -1]],
                "aspect_ratio": "auto or W:H",
            },
        }
        # The default deployment prepares the 768p 16:9 and 9:16 buckets.
        assert video["canvas"]["short_edges"] == [768]
        assert video["canvas"]["aspect_ratios"] == ["16:9", "9:16"]
        assert video["canvas"]["canvases"] == [
            {
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "width": 1344,
                "height": 768,
            },
            {
                "short_edge": 768,
                "aspect_ratio": "9:16",
                "width": 768,
                "height": 1344,
            },
        ]
        # The schedule counts sigma points: one per trained rung and the
        # clean endpoint.
        schedule = video["schedule"]
        assert schedule["num_inference_steps"] == expected_steps + 1
        # A request states its work in these fields alone; the checkpoint
        # fixes the schedule.
        assert set(video["request_fields"]) == {
            "model",
            "prompt",
            "task",
            "conditions",
            "target",
            "seed",
        }
        river = t2va_request("FastH3", "A river", 5, 1001)
        for payload in (
            # Longer than the deployment's default 15-second capacity.
            {**river, "target": {**river["target"], "duration_seconds": 16}},
            # Buckets the deployment did not prepare.
            {**river, "target": {**river["target"], "aspect_ratio": "1:1"}},
            {**river, "target": {**river["target"], "short_edge": 480}},
        ):
            response = client.post("/v1/videos", json=payload)
            assert response.status_code == 400, (payload, response.text)
        payload = t2va_request(
            "FastH3",
            "A river flows through a forest, with birds singing.",
            5,
            1001,
        )
        # Multipart fields carry the same names; the target is JSON text.
        form = {**payload, "target": json.dumps(payload["target"])}
        response = client.post(
            "/v1/videos",
            files={name: (None, str(value)) for name, value in form.items()},
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        job = completed(client, job_id)
        assert job["seconds"] == 5 and job["num_frames"] == 124
        assert job["size"] == "1344x768"
        assert job["actual_seconds"] == 124 / 24
        assert job["completed_steps"] == job["total_steps"] == expected_steps
        assert job["expires_at"] > job["completed_at"]
        assert job_id in {
            item["id"] for item in client.get("/v1/videos").json()["data"]
        }
        first = client.get(f"/v1/videos/{job_id}/content")
        second = client.get(f"/v1/videos/{job_id}/content")
        assert first.content == second.content
        media = inspect_video_bytes(
            first.content, declared_mime=first.headers["content-type"]
        )
        assert media.frame_count == 124 and media.audio_channels == 2
        assert client.delete(f"/v1/videos/{job_id}").json()["deleted"]
        assert client.get(f"/v1/videos/{job_id}/content").status_code == 404

        # Concurrent request storage must preserve each prompt and seed. Compare
        # both artifacts with isolated executions through the same public route.
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
            client.post("/v1/videos", json=item).json()["id"]
            for item in concurrent
        ]
        for item, concurrent_id in zip(concurrent, ids, strict=True):
            completed(client, concurrent_id)
            content = client.get(f"/v1/videos/{concurrent_id}/content")
            isolated = client.post("/v1/videos/sync", json=item)
            isolated.raise_for_status()
            _assert_media_values_close(content.content, isolated.content)

        # Cancel more requests than the two resident slots, including a
        # genuinely active job.
        for seed in range(3):
            active = client.post(
                "/v1/videos",
                json={
                    **payload,
                    "target": {**payload["target"], "duration_seconds": 15},
                    "seed": seed,
                },
            ).json()["id"]
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                state = client.get(f"/v1/videos/{active}").json()
                if state["status"] == "in_progress":
                    break
                assert state["status"] != "failed", state
                time.sleep(0.05)
            else:
                pytest.fail("job never reached execution")
            queued = client.post(
                "/v1/videos", json={**payload, "seed": seed + 100}
            ).json()["id"]
            assert client.delete(f"/v1/videos/{active}").status_code == 200
            assert client.delete(f"/v1/videos/{queued}").status_code == 200
            assert client.get(f"/v1/videos/{active}").status_code == 404
        reused = client.post(
            "/v1/videos", json={**payload, "seed": 1002}
        ).json()["id"]
        completed(client, reused)
        assert client.get(f"/v1/videos/{reused}/content").status_code == 200
        sync = client.post("/v1/videos/sync", json={**payload, "seed": 1003})
        sync.raise_for_status()
        assert (
            inspect_video_bytes(
                sync.content, declared_mime=sync.headers["content-type"]
            ).frame_count
            == 124
        )

    # The eager policy covers decoding as well as the learned denoising steps.
    # Keep the exact checkpoint, capacity, prompt and seed from the full run.
    command[-1] = "cuda_graph_config.policy=off"
    with (
        server_process(
            command, base, tmp_path / "video-eager.log", timeout_s=900
        ),
        httpx.Client(base_url=base, timeout=600) as client,
    ):
        eager = client.post("/v1/videos/sync", json=payload)
        eager.raise_for_status()
        _assert_media_values_close(eager.content, first.content)
