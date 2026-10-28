"""A FastH3 video request preempted mid-trajectory continues exactly.

Serves a FastH3 checkpoint (``UNISERVE_H3_MODEL``) on the two GPUs the
GPU locks granted (``UNISERVE_WORLD_DEVICES``): a Ulysses-2 denoiser and a
TP2 text encoder beside the decoders, and the CPU media group. One
``t2va`` request is served uninterrupted as the reference. The same
request, as a video job, is then preempted at a denoising step boundary:

- by a sleep, which serving uses to give back device storage: the engine
  stores the trajectory in the host latent pools and the request waits in
  the hold until the wake, after which the engine restores the stored step
  and denoises the steps left;
- by a world change whose denoising computes the same (a change of the
  workers' watchdog): the serving world parks with the trajectory stored,
  and the successor restores it and continues. A FastH3 request denoises
  in seconds while a successor prepares for tens of them, so the request
  starts once the successor prepares, and the serving world's model group
  is paused (``SIGSTOP``, as fault-injection suites pause a process) with
  the request mid-trajectory and resumed once the change warms the
  successor's cache (which reads the serving scheduler's requests between
  steps) or promotes it: the park then lands at a step boundary after the
  paused calls, whatever the two durations.

Each preempted video is the reference video, frame for frame and sample
for sample. Preemption starts only after the public job status reports
that denoising has begun. The reference computes its
prompt's conditioning and stores it in the host encoder pool; the job reads
the same prompt, so each of its admissions, on the serving world and on the
one that resumes it, reads the stored conditioning instead of encoding it,
and the videos' equality shows the stored conditioning is the computed one.

A world change of the denoiser's parallel configuration, from Ulysses 2 on
both GPUs to the whole denoiser on the first, computes the same: the
successor's one denoising rank lays the pages both serving ranks stored
out in its own pages and continues from the stored step. The re-layout
moves values without computing any (its exactness, bit for bit, is the
contract ``tests/python/integration/executor/test_video_latent_relayout.py``
checks), but the steps after it run with another division of the rows and
so another order of reductions, which a parallel configuration may change
without changing what the denoiser computes: the video is not expected to
equal either configuration's uninterrupted video bit for bit. The test
asserts that the request survives the configuration change and finishes
with the reference's video and audio frame counts. The layout conversion
is checked independently through the latent restore interface.
"""

from __future__ import annotations

import functools
import io
import os
import shutil
import signal
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import av
import httpx
import numpy as np
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    t2va_request,
    worker_executable,
    written_deployment,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("minimax_h3")]

NAME = "FastH3"
PROMPT = (
    "A red fox trots across a snowy meadow at sunrise, its breath steaming, "
    "while birds call from the pines."
)
SECONDS = 4.0
SEED = 4242


def _model() -> str:
    model = os.environ.get("UNISERVE_H3_MODEL")
    if not model or not Path(model).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name a supported FastH3 checkpoint "
            "directory; docs/minimax_h3/minimax_h3.md lists them"
        )
    return model


def _devices() -> str:
    """The two GPUs the deployment serves on, as their locks granted them
    (``UNISERVE_WORLD_DEVICES``), so that other runs keep the other GPUs.
    """  # noqa: D205
    devices = os.environ.get("UNISERVE_WORLD_DEVICES", "")
    if len([device for device in devices.split(",") if device]) != 2:
        pytest.fail(
            "UNISERVE_WORLD_DEVICES must name the two GPUs whose locks "
            "this run holds"
        )
    return devices


def _deployment(
    directory: Path, denoising_ranks: int = 2, fraction: float | None = None
) -> Path:
    """Two GPUs of the model group and the CPU media group: the denoiser
    Ulysses-2 on both GPUs, or on the first alone for one denoising rank;
    `fraction` is the model group's share of each GPU
    (`mem_fraction_static`), the deployment's default without one.
    """  # noqa: D205
    directory.mkdir(parents=True, exist_ok=True)
    transformer = (
        {"sequence_parallel": {"kind": "ulysses", "ulysses_degree": 2}}
        if denoising_ranks == 2
        else {"ranks": [0]}
    )
    model = {
        "id": "model",
        "ranks": ["cuda:0-1"],
        "queue_depth": 6,
        "components": {
            "transformer": transformer,
            "text_encoder": {"tensor_parallel_size": 2},
            "video_decoder": {},
            "audio_decoder": {"distribution": "temporal_units"},
            "latent_encoder": {"distribution": "temporal_units"},
        },
    }
    if fraction is not None:
        model["mem_fraction_static"] = fraction
    return written_deployment(
        directory,
        [
            model,
            {
                "id": "media",
                "ranks": ["cpu*4"],
                "queue_depth": 6,
                "components": {
                    "video_codec": {"ranks": [0, 1]},
                    "muxer": {"ranks": [2]},
                    "media_reader": {"ranks": [3]},
                },
            },
        ],
    )


# The model group's share of each GPU while a world change of the
# denoiser's parallel configuration runs: both worlds lie on the two GPUs
# during the change, the serving Ulysses-2 world with a transformer replica
# on each, and the default share leaves the successor's second rank, which
# holds the text encoder and the decoders, less than one request slot's
# tensors beside it.
CHANGE_FRACTION = 0.85


def _stop_storage_server(socket: Path) -> None:
    """Stop the Storage Server serving `socket`, found by its command line."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in command and str(socket).encode() in command:
            os.kill(int(entry.name), signal.SIGTERM)


def _group_workers(runtime: Path, group: str) -> list[int]:
    """The worker processes of worker group `group` the deployment serving
    from `runtime` runs now, found by their command line and environment.
    """  # noqa: D205
    workers = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
            environment = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if (
            b"worker" in command
            and b"--worker-id" in command[:-1]
            and command[command.index(b"--worker-id") + 1] == group.encode()
            and f"XDG_RUNTIME_DIR={runtime}".encode() in environment
        ):
            workers.append(int(entry.name))
    return workers


@contextmanager
def _serve(
    tmp_path: Path, deployment=_deployment
) -> Iterator[tuple[str, Path, Path]]:
    """The deployment `deployment` writes, with a Storage Server of its own;
    its URL, log and runtime directory.
    """  # noqa: D205
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-", dir="/tmp"))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        _model(),
        "--served-model-name",
        NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
        "--worker-groups",
        str(deployment(tmp_path)),
        "--max-running-requests",
        "1",
        "--max-model-len",
        "1024",
        "--set",
        "video_config.aspect_ratios=16:9",
        "--set",
        f"video_config.max_seconds={SECONDS:g}",
    ]
    log = tmp_path / "server.log"
    environment = {
        "CUDA_VISIBLE_DEVICES": _devices(),
        "XDG_RUNTIME_DIR": str(runtime),
    }
    try:
        with server_process(
            command, base_url, log, timeout_s=3600, env=environment
        ):
            yield base_url, log, runtime
    finally:
        _stop_storage_server(runtime / "uniserve" / "storage.sock")
        shutil.rmtree(runtime, ignore_errors=True)


def _decoded(video: bytes) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Every decoded video frame (RGB) and audio frame of an MP4."""
    frames = {}
    for kind in ("video", "audio"):
        with av.open(io.BytesIO(video)) as container:
            frames[kind] = [
                frame.to_ndarray(format="rgb24")
                if kind == "video"
                else frame.to_ndarray()
                for frame in container.decode(**{kind: 0})
            ]
    return frames["video"], frames["audio"]


def _assert_same_video(actual: bytes, expected: bytes) -> None:
    """The two videos decode to the same frames and samples, exactly."""
    actual_video, actual_audio = _decoded(actual)
    expected_video, expected_audio = _decoded(expected)
    assert len(actual_video) == len(expected_video) > 0
    assert len(actual_audio) == len(expected_audio) > 0
    for index, (frame, wanted) in enumerate(
        zip(actual_video, expected_video, strict=True)
    ):
        assert np.array_equal(frame, wanted), f"video frame {index} differs"
    for index, (frame, wanted) in enumerate(
        zip(actual_audio, expected_audio, strict=True)
    ):
        assert np.array_equal(frame, wanted), f"audio frame {index} differs"


def _start_job(base_url: str) -> str:
    response = httpx.post(
        f"{base_url}/v1/videos",
        json=t2va_request(NAME, PROMPT, SECONDS, SEED),
        timeout=60,
    )
    assert response.status_code in (200, 202), response.text
    return response.json()["id"]


def _job(base_url: str, job: str) -> dict:
    response = httpx.get(f"{base_url}/v1/videos/{job}", timeout=60)
    response.raise_for_status()
    return response.json()


def _wait_denoising(base_url: str, job: str, steps: int) -> int:
    """Waits until the job denoised `steps` steps; returns them."""
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        state = _job(base_url, job)
        assert state["status"] not in ("failed", "completed"), state
        if state.get("phase") == "denoising" and (
            state.get("completed_steps", 0) >= steps
        ):
            return int(state["completed_steps"])
        time.sleep(0.05)
    raise TimeoutError("the job did not denoise")


# A world change's phases in their order (``GET /admin/config``).
_PHASES = ("shrink", "prepare", "warm_cache", "promote")


def _wait_phase(base_url: str, change: Future, *phases: str) -> None:
    """Waits until the world change `change` is in one of `phases`, as
    ``GET /admin/config`` reports it; fails when the change answers or
    passes the last of them before.
    """  # noqa: D205
    phase = " or ".join(phases)
    last = max(_PHASES.index(name) for name in phases)
    # A successor prepares in tens of seconds; ten minutes bounds a change
    # whose preparation waits on the paused workers.
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if change.done():
            response = change.result()
            pytest.fail(
                f"the change ended before its {phase} phase: "
                f"{response.status_code} {response.text}"
            )
        config = httpx.get(f"{base_url}/admin/config", timeout=60).json()
        current = (config.get("update") or {}).get("phase")
        if current in phases:
            return
        if current in _PHASES and _PHASES.index(current) > last:
            pytest.fail(f"the change passed its {phase} phase: {current}")
        time.sleep(0.05)
    raise TimeoutError(f"the world change did not reach its {phase} phase")


def _content(base_url: str, job: str) -> bytes:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        state = _job(base_url, job)
        if state["status"] == "completed":
            break
        assert state["status"] != "failed", state
        time.sleep(0.2)
    else:
        raise TimeoutError("the job did not complete")
    response = httpx.get(f"{base_url}/v1/videos/{job}/content", timeout=600)
    response.raise_for_status()
    return response.content


def _admin(
    base_url: str, path: str, body: dict | None = None
) -> httpx.Response:
    return httpx.post(f"{base_url}/admin/{path}", json=body or {}, timeout=1800)


def test_a_preempted_video_request_continues_its_trajectory_exactly(
    tmp_path: Path,
) -> None:
    with _serve(tmp_path) as (base_url, log, runtime):
        steps = httpx.get(f"{base_url}/v1/capabilities", timeout=60).json()[
            "video"
        ]["schedule"]["num_inference_steps"]
        assert steps >= 4, steps

        reference = httpx.post(
            f"{base_url}/v1/videos/sync",
            json=t2va_request(NAME, PROMPT, SECONDS, SEED),
            timeout=1800,
        )
        assert reference.status_code == 200, reference.text
        reference = reference.content

        # A sleep preempts the request at a step boundary; the wake resumes
        # it from the stored step.
        job = _start_job(base_url)
        _wait_denoising(base_url, job, 2)
        slept = _admin(base_url, "sleep")
        assert slept.status_code == 200, slept.text
        assert _job(base_url, job)["status"] not in ("failed", "completed")
        woke = _admin(base_url, "wake_up")
        assert woke.status_code == 200, woke.text
        _assert_same_video(_content(base_url, job), reference)

        # A world change whose denoising computes the same parks the serving
        # world at a step boundary; its successor continues the trajectory.
        # The request starts once the serving world answered the change's
        # shrink (its scheduler answers between steps, which paused workers
        # never end) and the successor prepares; the serving world's model
        # group stays paused from the request's first step until the change
        # promotes, well within the workers' five-minute watchdog.
        workers = _group_workers(runtime, "model")
        assert len(workers) == 2, workers
        with ThreadPoolExecutor(max_workers=1) as pool:
            change = pool.submit(
                httpx.put,
                f"{base_url}/admin/config",
                json={"set": ["worker_config.watchdog_timeout=6m"]},
                timeout=3600,
            )
            _wait_phase(base_url, change, "prepare")
            job = _start_job(base_url)
            _wait_denoising(base_url, job, 1)
            for worker in workers:
                os.kill(worker, signal.SIGSTOP)
            try:
                # Warming the successor's cache reads the requests the
                # serving scheduler holds, which answers between steps: the
                # workers resume once the change warms or promotes.
                _wait_phase(base_url, change, "warm_cache", "promote")
            finally:
                for worker in workers:
                    os.kill(worker, signal.SIGCONT)
            changed = change.result()
        assert changed.status_code == 200, changed.text
        result = changed.json()["result"]
        assert result["change"] == "world", result
        assert result["world"]["numerics"] == "unchanged", result
        _assert_same_video(_content(base_url, job), reference)
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text


def test_a_video_request_continues_under_another_parallel_configuration(
    tmp_path: Path,
) -> None:
    serving = functools.partial(_deployment, fraction=CHANGE_FRACTION)
    with _serve(tmp_path, serving) as (base_url, log, runtime):
        steps = httpx.get(f"{base_url}/v1/capabilities", timeout=60).json()[
            "video"
        ]["schedule"]["num_inference_steps"]
        assert steps >= 4, steps
        reference = httpx.post(
            f"{base_url}/v1/videos/sync",
            json=t2va_request(NAME, PROMPT, SECONDS, SEED),
            timeout=1800,
        )
        assert reference.status_code == 200, reference.text
        reference = reference.content

        # The change parks the Ulysses-2 world at a step boundary, as the
        # world change of the other test does; its successor denoises on
        # the first GPU alone.
        workers = _group_workers(runtime, "model")
        assert len(workers) == 2, workers
        successor = _deployment(
            tmp_path / "successor", denoising_ranks=1, fraction=CHANGE_FRACTION
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            change = pool.submit(
                httpx.put,
                f"{base_url}/admin/config",
                json={"set": [f"worker_groups={successor}"]},
                timeout=3600,
            )
            _wait_phase(base_url, change, "prepare")
            job = _start_job(base_url)
            _wait_denoising(base_url, job, 1)
            for worker in workers:
                os.kill(worker, signal.SIGSTOP)
            try:
                # Warming the successor's cache reads the requests the
                # serving scheduler holds, which answers between steps: the
                # workers resume once the change warms or promotes.
                _wait_phase(base_url, change, "warm_cache", "promote")
            finally:
                for worker in workers:
                    os.kill(worker, signal.SIGCONT)
            changed = change.result()
        assert changed.status_code == 200, changed.text
        result = changed.json()["result"]
        assert result["change"] == "world", result
        assert result["world"]["numerics"] == "unchanged", result
        video = _content(base_url, job)
        # Kept beside the server's log, for comparison with the reference.
        (tmp_path / "reference.mp4").write_bytes(reference)
        (tmp_path / "continued.mp4").write_bytes(video)

        frames, audio = _decoded(video)
        reference_frames, reference_audio = _decoded(reference)
        assert len(frames) == len(reference_frames) > 0
        assert len(audio) == len(reference_audio) > 0
        assert all(
            frame.shape == wanted.shape
            for frame, wanted in zip(frames, reference_frames, strict=True)
        )
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
