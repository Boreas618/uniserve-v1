"""Helpers of the profiling end-to-end tests.

A Qwen3 deployment on one GPU (``UNISERVE_PROFILING_DEVICE``, default 0)
with a Storage Server of the test's own, decode load from streaming
requests, and the session calls of ``/admin/profile``.
"""

from __future__ import annotations

import os
import shutil
import signal
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    worker_executable,
)

NAME = "qwen3-profiling"
PROMPT = (
    "Count from 1 to 3000, separated by commas and spaces. "
    "Write only the numbers."
)


def checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_QWEN3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_MODEL must name the model checkpoint")
    return checkpoint


def stop_storage_server(socket: Path) -> None:
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


@contextmanager
def serve(tmp_path: Path, options: Sequence[str]) -> Iterator[str]:
    """Serve Qwen3 under ``options`` (``--set`` assignments) until the block
    ends; yields the base URL.
    """  # noqa: D205
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        checkpoint(),
        "--served-model-name",
        NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
    ]
    for option in options:
        command += ["--set", option]
    environment = {
        "CUDA_VISIBLE_DEVICES": os.environ.get(
            "UNISERVE_PROFILING_DEVICE", "0"
        ),
        "XDG_RUNTIME_DIR": str(runtime),
    }
    try:
        with server_process(
            command, base_url, tmp_path / "server.log", 900, environment
        ):
            yield base_url
    finally:
        stop_storage_server(runtime / "uniserve" / "storage.sock")
        shutil.rmtree(runtime, ignore_errors=True)


def _stream(base_url: str, stop: threading.Event) -> None:
    """Stream counts until `stop` is set: decode steps throughout."""
    while not stop.is_set():
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": NAME,
                "messages": [{"role": "user", "content": PROMPT}],
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0,
                "max_completion_tokens": 2048,
                "stream": True,
            },
            timeout=1800,
        ) as response:
            for _ in response.iter_lines():
                if stop.is_set():
                    break


class Load:
    """Requests streaming on threads of their own while the block runs."""

    def __init__(self, base_url: str, streams: int = 2) -> None:
        self.stop = threading.Event()
        self.threads = [
            threading.Thread(target=_stream, args=(base_url, self.stop))
            for _ in range(streams)
        ]

    def __enter__(self) -> Load:
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=600)


def profile(base_url: str) -> dict | None:
    response = httpx.get(f"{base_url}/admin/profile", timeout=30)
    response.raise_for_status()
    return response.json()["session"]


def start(base_url: str, request: dict) -> dict:
    response = httpx.post(
        f"{base_url}/admin/profile", json=request, timeout=600
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def ended(base_url: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        session = profile(base_url)
        if session and session["state"] == "ended":
            return session
        time.sleep(1.0)
    raise TimeoutError(f"the session did not end within {timeout_s} s")
