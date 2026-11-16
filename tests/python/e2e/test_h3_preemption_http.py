"""A FastH3 video request preempted mid-trajectory continues exactly.

Serves FastH3 on the two GPUs each case leases: a Ulysses-2 denoiser and a
TP2 text encoder beside the decoders and the media stages. One ``t2va``
request is served uninterrupted as the reference. The same request, as a
video job, is then preempted at a denoising step boundary, where the
scheduler saves its trajectory (``Save``) into a host block of the
deployment's buffer pool under the trajectory's key (``specs/buffer-pool.md``
§6, §13), and later restores it (``Restore``) into the slot of the admission
that continues it:

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
  of the trajectory, whatever the two durations. The admin API holds no
  pause at a step boundary (``POST /admin/pause_generation`` holds new
  requests only), so the process is paused by its signal.

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

import os
import signal
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import httpx
import pytest
from uniserve_testing import admin, assets, client, media, updates
from uniserve_testing.deployments import Deployments
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
]

NAME = "FastH3"
PROMPT = (
    "A red fox trots across a snowy meadow at sunrise, its breath steaming, "
    "while birds call from the pines."
)
# The one size the deployment serves.
SIZE = "1344x768"
# The shortest request duration bounds the deployment; its 107 frames are
# what each request asks for.
SECONDS = 4.0
FRAMES = 107
SEED = 4242

# The drain's deadline every change of the tests states: the serving
# engine ends its running request as migratable at the drain, at the
# request's next denoising step boundary.
DRAIN_AT_ONCE = "update_config.drain_s=0"

# The model group's share of each GPU while an epoch change of the
# denoiser's parallel configuration runs: both epochs lie on the two GPUs
# during the change, the serving Ulysses-2 epoch with a transformer replica
# on each, and the default share leaves the target's second rank, which
# holds the text encoder and the decoders, less than one request slot's
# tensors beside it.
CHANGE_FRACTION = 0.85


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
    model: dict[str, Any] = {
        "id": "model",
        "ranks": ["cuda:0-1"],
        "queue_depth": 6,
        "blocks": {
            "transformer": transformer,
            "text_encoder": {"tensor_parallel_size": 2},
            "video_decoder": {"distribution": "temporal_units"},
            "audio_decoder": {"distribution": "temporal_units"},
            "latent_encoder": {"distribution": "temporal_units"},
            "video_codec": {"distribution": "temporal_units"},
            "muxer": {"ranks": [0]},
            "media_reader": {"ranks": [0]},
        },
    }
    if fraction is not None:
        model["mem_fraction_static"] = fraction
    return client.written_deployment(directory, [model])


def _serve(
    deployments: Deployments, deployment: Path
) -> AbstractContextManager[Server]:
    """The deployment the file `deployment` places, admitting one request
    of up to :data:`SECONDS` at :data:`SIZE`.
    """  # noqa: D205
    return deployments.serve(
        "--served-model-name",
        NAME,
        "--config",
        str(deployment),
        "--max-running-requests",
        "1",
        "--max-model-len",
        "1024",
        # The default prompt-capacity ladder holds capacities below the
        # admitted length only, so the deployment states its one capacity.
        "--set",
        "video_config.text_capacities=[1024]",
        "--set",
        f'video_config.sizes=["{SIZE}"]',
        "--set",
        f"video_config.max_seconds={SECONDS:g}",
        model=assets.CHECKPOINTS["fasth3"].model,
    )


def _request() -> dict[str, Any]:
    """The request every case sends, as the reference and as a job."""
    return media.t2va_request(NAME, PROMPT, FRAMES, SEED, size=SIZE)


def _reference(server: Server) -> bytes:
    """The request's video served uninterrupted, once the schedule is long
    enough to be preempted mid-trajectory.
    """  # noqa: D205
    with media.api(server) as api:
        capabilities = api.get("/v1/capabilities").json()
        steps = capabilities["video"]["model"]["num_inference_steps"]
        assert steps >= 4, steps
        reference = api.post("/v1/videos/sync", json=_request())
    assert reference.status_code == 200, reference.text
    return reference.content


def _start_job(server: Server) -> str:
    response = httpx.post(f"{server.base_url}/v1/videos", json=_request())
    assert response.status_code in (200, 202), response.text
    return response.json()["id"]


def _denoised(steps: int) -> Callable[[dict[str, Any]], bool]:
    """Whether a job's record states `steps` denoising steps done; a job
    that completed first was never preempted, which fails the case.
    """  # noqa: D205

    def reached(record: dict[str, Any]) -> bool:
        assert record["status"] != "completed", record
        return (
            record.get("phase") == "denoising"
            and record["completed_steps"] >= steps
        )

    return reached


def _change_under_a_denoising_request(
    server: Server, *assignments: str, **fields: object
) -> tuple[str, dict]:
    """Applies `assignments` and the request's other `fields` (a target
    ``config``) as an epoch change while a video job denoises on the serving
    epoch; returns the job and the update as its trial began.

    The job starts once the target prepares. The serving engine's process
    is paused after the job's first denoising step and continued once the
    switch begins, so the drain finds the job mid-trajectory. The phases are
    the node's ``update`` events, which the daemon writes from its own state
    while the serving engine is paused.
    """  # noqa: D205
    after = admin.next_event(server)
    with ThreadPoolExecutor(max_workers=1) as pool:
        change = pool.submit(
            admin.apply, server, *assignments, DRAIN_AT_ONCE, **fields
        )
        admin.wait_for_phase(server, "preparing", after=after)
        job = _start_job(server)
        media.wait_for_job(server, job, _denoised(1), "the job did not denoise")
        engine = admin.serving_engine(server)["pid"]
        os.kill(engine, signal.SIGSTOP)
        try:
            admin.wait_for_phase(server, "switching", after=after)
        finally:
            os.kill(engine, signal.SIGCONT)
        changed = change.result()
    assert changed.status_code == 200, changed.text
    result = changed.json()["result"]
    assert result["change"] == "epoch", result
    return job, result["update"]


@pytest.mark.gpus(2)
def test_a_preempted_video_request_continues_its_trajectory_exactly(
    deployments: Deployments, tmp_path: Path
) -> None:
    with _serve(deployments, _deployment(tmp_path)) as server:
        reference = _reference(server)

        # A sleep preempts the request at a step boundary; the wake resumes
        # it from the stored step.
        job = _start_job(server)
        media.wait_for_job(server, job, _denoised(2), "the job did not denoise")
        slept = admin.operation(server, "sleep")
        assert slept.status_code == 200, slept.text
        assert media.job(server, job)["status"] not in ("failed", "completed")
        woke = admin.operation(server, "wake_up")
        assert woke.status_code == 200, woke.text
        media.assert_same_media(
            media.job_video(server, job).content, reference, tolerance=0
        )

        # An epoch change whose denoising computes the same keeps the layout;
        # its drain ends the request at a step boundary and the target
        # continues the trajectory.
        job, update = _change_under_a_denoising_request(
            server, 'worker_config.watchdog_timeout="6m"'
        )
        assert update["shrink"] is False, update
        media.assert_same_media(
            media.job_video(server, job).content, reference, tolerance=0
        )
        committed = admin.operation(server, "config/commit")
        assert committed.status_code == 200, committed.text


@pytest.mark.gpus(2)
def test_a_video_request_continues_under_another_parallel_configuration(
    deployments: Deployments, tmp_path: Path
) -> None:
    serving = _deployment(tmp_path, fraction=CHANGE_FRACTION)
    with _serve(deployments, serving) as server:
        reference = _reference(server)

        # The drain ends the Ulysses-2 epoch's request at a step boundary, as
        # the epoch change of the other test does; the target, another
        # layout, denoises on the first GPU alone.
        # The target is the served configuration with the target's worker
        # groups, assigned as one TOML value: with assignments alone, the
        # drain is this update's own option and the daemon's stay as served.
        target = _deployment(
            tmp_path / "target", denoising_ranks=1, fraction=CHANGE_FRACTION
        )
        groups = tomllib.loads(target.read_text())["worker_groups"]
        job, update = _change_under_a_denoising_request(
            server, f"worker_groups={updates.inline(groups)}"
        )
        assert update["shrink"] is True, update
        video = media.job_video(server, job).content
        # Kept beside the server's log, for comparison with the reference.
        (tmp_path / "reference.mp4").write_bytes(reference)
        (tmp_path / "continued.mp4").write_bytes(video)

        for kind in media.STREAMS:
            frames = media.decoded(video, kind)
            reference_frames = media.decoded(reference, kind)
            assert len(frames) == len(reference_frames) > 0, kind
            assert all(
                values.shape == wanted.shape
                for (_, values), (_, wanted) in zip(
                    frames, reference_frames, strict=True
                )
            ), kind
        committed = admin.operation(server, "config/commit")
        assert committed.status_code == 200, committed.text
