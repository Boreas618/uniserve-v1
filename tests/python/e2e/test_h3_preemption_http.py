"""A FastH3 video request preempted mid-trajectory continues exactly.

Serves a FastH3 checkpoint (``UNISERVE_H3_MODEL``) on the two GPUs the
module leases (``UNISERVE_WORLD_DEVICES``): a Ulysses-2 denoiser and a TP2
text encoder beside the decoders and the media stages. One ``t2va`` request
is served uninterrupted as the reference. The same request, as a video job,
is then preempted at a denoising step boundary, where the scheduler saves its
trajectory (``Save``) into a host block of the deployment's buffer pool under
the trajectory's key (``specs/buffer-pool.md`` §6, §13), and later restores
it (``Restore``) into the slot of the admission that continues it:

- by a sleep, which serving uses to give back device storage: the request
  waits in the hold until the wake, after which the engine restores the
  stored step and denoises the steps left;
- by an epoch change whose denoising computes the same (a change of the
  workers' watchdog): the switch promotes the target and drains the serving
  engine with a deadline of zero (``update_config.drain_s``), which ends the
  request in flight at its next step boundary with its trajectory saved; the
  daemon resubmits it to the target, which restores it and continues
  (``specs/updates.md`` §3.7). A FastH3 request denoises in seconds while a
  target prepares for tens of them, so the request starts once the target
  prepares, and the serving engine's process is paused (``SIGSTOP``, as
  fault-injection suites pause a process) after its first denoising step and
  continued once the switch begins: the drain then lands at a step boundary
  of the trajectory, whatever the two durations.

Each preempted video is the reference video, frame for frame and sample
for sample. Preemption starts only after the public job status reports
that denoising has begun. The reference computes its prompt's encoder output
and commits it under its key; the job reads the same prompt, so each of its
admissions, on the serving epoch and on the one that resumes it, pins the
committed encoder output instead of encoding it, and the videos' equality
shows the stored output is the computed one.

An epoch change of the denoiser's parallel configuration, from Ulysses 2 on
both GPUs to the whole denoiser on the first, is another layout (``shrink``)
and computes the same: the target's one denoising rank restores the
trajectory both serving ranks saved, found by its key under any layout and
converted through the copy manifest of the two layouts' encodings, and
continues from the stored step. The steps after it run with another division
of the rows and so another order of reductions, which a parallel
configuration may change without changing what the denoiser computes: the
video is not expected to equal either configuration's uninterrupted video bit
for bit. The test asserts that the request survives the configuration change
and finishes with the reference's video and audio frame counts.
"""

from __future__ import annotations

import functools
import io
import os
import signal
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
    H3_CHECKPOINT,
    admin,
    admin_status,
    apply_config,
    engine_selection,
    granted_devices,
    host_asset,
    require_uniserve_binary,
    runtime_directory,
    server_process,
    serving_engine,
    t2va_request,
    written_deployment,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.gpus(2),
]

NAME = "FastH3"
PROMPT = (
    "A red fox trots across a snowy meadow at sunrise, its breath steaming, "
    "while birds call from the pines."
)
# The shortest request duration bounds the deployment; its 107 frames are
# what each request asks for.
SECONDS = 4.0
FRAMES = 107
SEED = 4242


def _model() -> str:
    return str(host_asset("UNISERVE_H3_MODEL", H3_CHECKPOINT))


def _deployment(
    directory: Path, denoising_ranks: int = 2, fraction: float | None = None
) -> Path:
    """One group on two GPUs: the denoiser Ulysses-2 on both GPUs, or on the
    first alone for one denoising rank, beside the encoders, the decoders and
    the media stages; `fraction` is the group's share of each GPU
    (`mem_fraction_static`), the deployment's default without one.

    The media stages run on the host inside the main process, which serves
    CUDA devices only, so they are placed on the group's CUDA ranks as the
    deployment files of ``configs/minimax_h3`` place them.
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
        "blocks": {
            "transformer": transformer,
            "text_encoder": {"tensor_parallel_size": 2},
            "video_decoder": {},
            "audio_decoder": {"distribution": "temporal_units"},
            "latent_encoder": {"distribution": "temporal_units"},
            "video_codec": {},
            "muxer": {"ranks": [0]},
            "media_reader": {"ranks": [0]},
        },
    }
    if fraction is not None:
        model["mem_fraction_static"] = fraction
    return written_deployment(directory, [model])


# The model group's share of each GPU while an epoch change of the
# denoiser's parallel configuration runs: both epochs lie on the two GPUs
# during the change, the serving Ulysses-2 epoch with a transformer replica
# on each, and the default share leaves the target's second rank, which
# holds the text encoder and the decoders, less than one request slot's
# tensors beside it.
CHANGE_FRACTION = 0.85


@contextmanager
def _serve(tmp_path: Path, deployment=_deployment) -> Iterator[str]:
    """The deployment `deployment` writes, with a Storage Server of its own;
    its URL.
    """  # noqa: D205
    command = [
        str(require_uniserve_binary()),
        "serve",
        _model(),
        "--served-model-name",
        NAME,
        "--host",
        "127.0.0.1",
        *engine_selection(),
        "--worker-groups",
        str(deployment(tmp_path)),
        "--max-running-requests",
        "1",
        "--max-model-len",
        "1024",
        # The default prompt-capacity ladder holds capacities below the
        # admitted length only, so the deployment states its one capacity.
        "--set",
        "video_config.text_capacities=[1024]",
        "--set",
        'video_config.sizes=["1344x768"]',
        "--set",
        f"video_config.max_seconds={SECONDS:g}",
    ]
    log = tmp_path / "server.log"
    with runtime_directory() as runtime:
        environment = {
            # The two GPUs the module's lease granted, so that other cases
            # keep the other GPUs.
            "CUDA_VISIBLE_DEVICES": granted_devices(2),
            "XDG_RUNTIME_DIR": str(runtime),
        }
        with server_process(
            command, log, timeout_s=3600, env=environment
        ) as server:
            base_url = server.base_url
            yield base_url


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
        json=t2va_request(NAME, PROMPT, FRAMES, SEED),
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


# The drain's deadline every change of the tests states: the serving
# engine ends its running request as migratable at the drain, at the
# request's next denoising step boundary.
DRAIN_AT_ONCE = "update_config.drain_s=0"


def _wait_phase(base_url: str, change: Future, phase: str) -> None:
    """Waits until the epoch change `change` stands in `phase`, as ``GET
    /admin/status`` reports the update in flight, which the daemon answers
    from its own state while the serving engine is paused; fails when the
    change answers before.
    """  # noqa: D205
    # A target prepares in tens of seconds; ten minutes bounds a change
    # whose preparation is slow.
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if change.done():
            response = change.result()
            pytest.fail(
                f"the change ended before its {phase} phase: "
                f"{response.status_code} {response.text}"
            )
        update = admin_status(base_url)["deployment"]["update"] or {}
        if update.get("phase") == phase:
            return
        time.sleep(0.05)
    raise TimeoutError(f"the epoch change did not reach its {phase} phase")


def _change_under_a_denoising_request(
    base_url: str, *assignments: str
) -> tuple[str, dict]:
    """Applies `assignments` as an epoch change while a video job denoises
    on the serving epoch; returns the job and the update as its trial began.

    The job starts once the target prepares. The serving engine's process
    is paused after the job's first denoising step and continued once the
    switch begins, so the drain finds the job mid-trajectory.
    """  # noqa: D205
    with ThreadPoolExecutor(max_workers=1) as pool:
        change = pool.submit(
            apply_config, base_url, *assignments, DRAIN_AT_ONCE
        )
        _wait_phase(base_url, change, "preparing")
        job = _start_job(base_url)
        _wait_denoising(base_url, job, 1)
        engine = serving_engine(base_url)["pid"]
        os.kill(engine, signal.SIGSTOP)
        try:
            _wait_phase(base_url, change, "switching")
        finally:
            os.kill(engine, signal.SIGCONT)
        changed = change.result()
    assert changed.status_code == 200, changed.text
    result = changed.json()["result"]
    assert result["change"] == "epoch", result
    return job, result["update"]


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


def test_a_preempted_video_request_continues_its_trajectory_exactly(
    tmp_path: Path,
) -> None:
    with _serve(tmp_path) as base_url:
        steps = httpx.get(f"{base_url}/v1/capabilities", timeout=60).json()[
            "video"
        ]["model"]["num_inference_steps"]
        assert steps >= 4, steps

        reference = httpx.post(
            f"{base_url}/v1/videos/sync",
            json=t2va_request(NAME, PROMPT, FRAMES, SEED),
            timeout=1800,
        )
        assert reference.status_code == 200, reference.text
        reference = reference.content

        # A sleep preempts the request at a step boundary; the wake resumes
        # it from the stored step.
        job = _start_job(base_url)
        _wait_denoising(base_url, job, 2)
        slept = admin(base_url, "sleep")
        assert slept.status_code == 200, slept.text
        assert _job(base_url, job)["status"] not in ("failed", "completed")
        woke = admin(base_url, "wake_up")
        assert woke.status_code == 200, woke.text
        _assert_same_video(_content(base_url, job), reference)

        # An epoch change whose denoising computes the same keeps the layout;
        # its drain ends the request at a step boundary and the target
        # continues the trajectory.
        job, update = _change_under_a_denoising_request(
            base_url, "worker_config.watchdog_timeout=6m"
        )
        assert update["shrink"] is False, update
        _assert_same_video(_content(base_url, job), reference)
        committed = admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text


def test_a_video_request_continues_under_another_parallel_configuration(
    tmp_path: Path,
) -> None:
    serving = functools.partial(_deployment, fraction=CHANGE_FRACTION)
    with _serve(tmp_path, serving) as base_url:
        steps = httpx.get(f"{base_url}/v1/capabilities", timeout=60).json()[
            "video"
        ]["model"]["num_inference_steps"]
        assert steps >= 4, steps
        reference = httpx.post(
            f"{base_url}/v1/videos/sync",
            json=t2va_request(NAME, PROMPT, FRAMES, SEED),
            timeout=1800,
        )
        assert reference.status_code == 200, reference.text
        reference = reference.content

        # The drain ends the Ulysses-2 epoch's request at a step boundary, as
        # the epoch change of the other test does; the target, another
        # layout, denoises on the first GPU alone.
        target = _deployment(
            tmp_path / "target", denoising_ranks=1, fraction=CHANGE_FRACTION
        )
        job, update = _change_under_a_denoising_request(
            base_url, f"worker_groups={target}"
        )
        assert update["shrink"] is True, update
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
        committed = admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
