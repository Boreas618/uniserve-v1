"""A DiffusionGemma expert world keeps a standby world over MegaMoE.

Two DiffusionGemma NVFP4 schedulers share their routed experts through the
fused MegaMoE kernel, whose symmetric buffers live in an NVSHMEM heap. With
``worker_config.standby`` the failure domain keeps a standby world: once
both schedulers serve, each worker group is launched a second time at a
rendezvous of its own, captures its graphs and parks. Suspended
(``worker_config.standby_storage``), the standby frees its graph storage, its
NCCL buffers and its NVSHMEM heap, whose addresses NVSHMEM does not keep
across leaving and joining; its promotion restores the first two, joins
NVSHMEM again with new buffers and captures the graphs that run expert
steps anew.

Killing a serving rank promotes the standby world in place of a relaunch.
A seeded chat request answered before the loss is answered the same by the
promoted world.

The checkpoint directory comes from ``UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL``.
The test runs a Storage Server of its own: ``XDG_RUNTIME_DIR`` points the
daemon at a directory of the test's, and the server the daemon starts there
is stopped at the end. Two GPUs must be visible.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    granted_devices,
    require_uniserve_binary,
    server_process,
    worker_executable,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("diffusion_gemma"),
]

SERVED_MODEL = "diffusion-gemma-standby"
QUESTION = "Name the three primary colors of light, separated by commas."
SEED = 7


def _checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail(
            "UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL must name the checkpoint"
        )
    return checkpoint


def _log_lines(log: Path) -> list[str]:
    """The lines of `log` without their terminal colors."""
    colors = re.compile(r"\x1b\[[0-9;]*m")
    return [colors.sub("", line) for line in log.read_text().splitlines()]


def _await_line(log: Path, text: str, timeout_s: float) -> None:
    """Wait until the server log holds a line containing `text`."""
    deadline = time.monotonic() + timeout_s
    while not any(text in line for line in _log_lines(log)):
        if time.monotonic() > deadline:
            pytest.fail(f"the server did not log {text!r}")
        time.sleep(1)


def _serving_worker(runtime: Path) -> int:
    """The pid of a worker of the first launch of its group."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().decode().split("\0")
            environment = (entry / "environ").read_bytes().decode()
        except OSError:
            continue
        if (
            len(command) > 1
            and command[1] == "worker"
            and "--generation\x001\x00" in "\x00".join(command)
            and str(runtime) in environment
        ):
            return int(entry.name)
    pytest.fail("no serving worker runs")


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


def _ask(base_url: str) -> str:
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": QUESTION}],
            "max_completion_tokens": 32,
            "seed": SEED,
        },
        timeout=600,
    )
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]["content"]


@pytest.mark.parametrize("standby_storage", ["resident", "suspended"])
def test_a_megamoe_world_promotes_its_standby_world_when_a_rank_is_lost(
    tmp_path: Path, standby_storage: str
):
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-", dir="/tmp"))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        _checkpoint(),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
        "--max-model-len",
        "8192",
        "--data-parallel-size",
        "2",
        "--set",
        "moe_config.expert_parallel=true",
        "--set",
        "moe_config.exchange=megamoe",
        "--set",
        "worker_config.standby=true",
        "--set",
        f"worker_config.standby_storage={standby_storage}",
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": granted_devices(2),
        "XDG_RUNTIME_DIR": str(runtime),
    }
    log = tmp_path / "server.log"
    try:
        with server_process(command, base_url, log, timeout_s=1800, env=env):
            _await_line(
                log, "the standby world of the failure domain is parked", 1800
            )
            before = _ask(base_url)
            os.kill(_serving_worker(runtime), signal.SIGKILL)
            _await_line(
                log, "the failure domain's standby world was promoted", 900
            )
            after = _ask(base_url)
    finally:
        _stop_storage_server(runtime / "uniserve" / "storage.sock")
        shutil.rmtree(runtime, ignore_errors=True)

    lines = _log_lines(log)

    def logged(text: str) -> bool:
        return any(text in line for line in lines)

    assert not logged("relaunching the failure domain's worker groups")
    if standby_storage == "suspended":
        for event in (
            "the standby worker released its expert workspaces",
            "the standby worker suspended its communicators",
            "the promoted standby worker resumed its communicators",
            "the promoted standby worker captured its expert graphs again",
        ):
            assert logged(event), event
    else:
        assert logged("the standby worker keeps its graph storage resident")
    assert before and after == before
