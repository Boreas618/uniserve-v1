"""A running head hands its deployment to a successor without failing a request.

Qwen3 streams a long greedy count over a Unix-domain socket to a client
that reads its first tokens and then pauses. A second ``uniserve serve`` of
the same command starts meanwhile, and the count takes longer to generate
than the second server takes to start. The second server succeeds the
running head (a control-plane hot update): its workers stand by for the
head's, the head quiesces, and the successor resumes the request from the
cache pages the head left, recomputing at most its last partial page. When
the client reads on, the rest of the stream arrives through the head's open
connection.

The stream decodes the tokens the head decodes for the same request alone,
and its usage charges the same prompt and completion. The head stops
accepting connections and exits once its last one closes, leaving the
socket file to the successor; a request on a new connection is answered by
the successor with the same tokens.

The count is a continuation that no near tie decides, so rounding
differences between the head's decode of the resumed tokens and the
successor's prefill of the last page cannot change it.

The test runs a Storage Server of its own: ``XDG_RUNTIME_DIR`` points both
heads at a directory of the test's, and the server the first head starts
there is stopped at the end.
"""

from __future__ import annotations

import json
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
    require_uniserve_binary,
    start_server,
    stop_server,
    worker_executable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

SERVED_MODEL = "qwen3-hot-update"
# At about 230 tokens a second, the count outlasts the successor's start,
# which captures no prefill graphs.
MAX_TOKENS = 16384
# Tokens the client reads before it pauses.
EARLY_TOKENS = 64
# The host name of requests over the Unix-domain socket.
BASE_URL = "http://uniserve"
PROMPT = (
    "Count from 1 to 5000, separated by commas and spaces. "
    "Write only the numbers."
)


def _body(stream: bool) -> dict:
    body = {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "max_completion_tokens": MAX_TOKENS,
        "logprobs": True,
        "return_tokens_as_token_ids": True,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _client(socket_path: Path) -> httpx.Client:
    """A client of the server listening on ``socket_path``."""
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=str(socket_path)), timeout=900
    )


def _wait_health(socket_path: Path, process, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"the server exited with code {process.returncode}"
            )
        try:
            with _client(socket_path) as client:
                if client.get(f"{BASE_URL}/health").status_code == 200:
                    return
        except httpx.TransportError:
            pass
        time.sleep(1)
    raise TimeoutError(f"the server did not serve within {timeout_s}s")


def _ask(socket_path: Path) -> tuple[list[str], dict]:
    """Decode the count on a connection of its own: tokens and usage."""
    with _client(socket_path) as client:
        response = client.post(
            f"{BASE_URL}/v1/chat/completions", json=_body(stream=False)
        )
    response.raise_for_status()
    reply = response.json()
    tokens = [
        entry["token"] for entry in reply["choices"][0]["logprobs"]["content"]
    ]
    return tokens, reply["usage"]


def _chunks(lines) -> tuple[list[str], dict | None]:
    """The tokens and usage of SSE ``lines``, until ``[DONE]``."""
    tokens: list[str] = []
    usage = None
    for line in lines:
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            break
        chunk = json.loads(line[6:])
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", ()):
            logprobs = choice.get("logprobs") or {}
            tokens.extend(
                entry["token"] for entry in logprobs.get("content") or ()
            )
    return tokens, usage


def _wait_for_line(log: Path, pattern: str, process, timeout_s: float) -> str:
    """The first line of ``log`` matching ``pattern``, without its colors."""
    deadline = time.monotonic() + timeout_s
    expression = re.compile(pattern)
    colors = re.compile(r"\x1b\[[0-9;]*m")
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"the server exited with code {process.returncode}"
            )
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                line = colors.sub("", line)
                if expression.search(line):
                    return line
        time.sleep(1)
    raise TimeoutError(f"no line of {log} matched {pattern!r} in {timeout_s}s")


def _stop_storage_server(socket_path: Path) -> None:
    """Stop the Storage Server serving ``socket_path``.

    The server detached from the head that started it, so it is found by
    its command line.
    """
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in command and str(socket_path).encode() in command:
            os.kill(int(entry.name), signal.SIGTERM)


def test_a_successor_head_takes_a_streaming_request_over(tmp_path: Path):
    checkpoint = os.environ.get("UNISERVE_QWEN3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_MODEL must name the model checkpoint")

    # A Unix socket path holds at most 107 bytes, so the heads' sockets live
    # in a short directory of their own.
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-", dir="/tmp"))
    listening = runtime / "http.sock"
    command = [
        str(require_uniserve_binary()),
        "serve",
        checkpoint,
        "--served-model-name",
        SERVED_MODEL,
        "--set",
        f"server_config.uds={listening}",
        "--worker-executable",
        str(worker_executable()),
        "--set",
        "dtype=bfloat16",
        "--set",
        "kv_cache_config.max_tokens=32768",
        "--set",
        "chat_config.reasoning_parser=none",
        "--set",
        "cuda_graph_config.capture_prefill=false",
    ]
    env = {"CUDA_VISIBLE_DEVICES": "0", "XDG_RUNTIME_DIR": str(runtime)}
    head_log = tmp_path / "head.log"
    successor_log = tmp_path / "successor.log"
    head = start_server(command, head_log, env)
    successor = None
    try:
        _wait_health(listening, head, timeout_s=900)
        expected, expected_usage = _ask(listening)
        assert len(expected) == MAX_TOKENS

        with _client(listening) as client:
            with client.stream(
                "POST",
                f"{BASE_URL}/v1/chat/completions",
                json=_body(stream=True),
            ) as response:
                response.raise_for_status()
                lines = response.iter_lines()
                early: list[str] = []
                for line in lines:
                    tokens, _ = _chunks([line])
                    early.extend(tokens)
                    if len(early) >= EARLY_TOKENS:
                        break

                successor = start_server(command, successor_log, env)
                took_over = _wait_for_line(
                    successor_log,
                    r"took the deployment over from the predecessor head",
                    successor,
                    timeout_s=900,
                )
                assert re.search(r"requests=1\b", took_over), (
                    f"the stream had ended at the handover: {took_over}"
                )
                _wait_for_line(
                    successor_log,
                    r"starting HTTP server",
                    successor,
                    timeout_s=60,
                )

                rest, usage = _chunks(lines)

        assert early + rest == expected
        assert usage == expected_usage

        # The head exits once its last connection closed, and leaves the
        # socket file to the successor, which answers on it.
        assert head.wait(timeout=120) == 0
        assert listening.exists()
        tokens, usage = _ask(listening)
        assert (tokens, usage) == (expected, expected_usage)
    finally:
        if successor is not None:
            stop_server(successor)
        stop_server(head)
        _stop_storage_server(runtime / "uniserve" / "storage.sock")
        shutil.rmtree(runtime, ignore_errors=True)
