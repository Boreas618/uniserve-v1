"""A worker group over two hosts survives the loss of one host's rank.

Qwen3 is served by a tensor group of two ranks: rank 0 on this host, whose main
process holds the scheduler, and rank 1 on a second host, whose main process is
a peer that registers with the scheduler node over the peer channel and runs
every batch the scheduler sends it (`docs/architecture.md`, The channel). The
deployment keeps a standby (``worker_config.standby``): once it serves, the
daemon's failure-domain rules start a second main process on each host, one
standby epoch over the same two hosts, which parks.

While a long greedy count streams, the second host's serving main process is
killed. Its connection's close ends the scheduler node's process too, the
domain's standbys are promoted together, and the request continues on the
promoted epoch: it migrates after every token it generated, and the client
reads one uninterrupted stream, which matches the count asked again afterwards.

The second host is named by the two-host variables of the suite
(docs/testing.md, Two hosts): ``UNISERVE_H3_FOLLOWER``, an ssh destination with
this checkout at the same path, and ``UNISERVE_H3_LEADER_ADDRESS``, this host's
address as the second reaches it. Without them the test is skipped. The second
host's main process resolves its modules from a Program Server the test starts
there with the packages this host's holds.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    ATTENTION_BACKEND,
    QWEN3_CHECKPOINT,
    find_free_port,
    granted_devices,
    host_asset,
    require_uniserve_binary,
    second_host,
    server_process,
    stop_storage_server,
    worker_executable,
)
from tests.python.e2e.program_server import remote_program_server

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

FOLLOWER = os.environ.get("UNISERVE_H3_FOLLOWER")
LEADER_ADDRESS = os.environ.get("UNISERVE_H3_LEADER_ADDRESS")

SERVED_MODEL = "qwen3-cross-host"
MAX_TOKENS = 16384
# Tokens the client reads before the second host's rank is killed.
EARLY_TOKENS = 64
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


def _chunk_tokens(line: str) -> tuple[list[str], dict | None, bool]:
    """The tokens and usage one SSE line carries, and whether it ends."""
    if not line.startswith("data: "):
        return [], None, False
    if line == "data: [DONE]":
        return [], None, True
    chunk = json.loads(line[6:])
    tokens = [
        entry["token"]
        for choice in chunk.get("choices", ())
        for entry in (choice.get("logprobs") or {}).get("content") or ()
    ]
    return tokens, chunk.get("usage"), False


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


def _kill_serving_peer(host: str) -> None:
    """Kill the second host's serving main process.

    It is the oldest main process this test's follower daemon started there,
    whose launch file lies in the follower's runtime directory, which
    `http_helpers.second_host` names after this process.
    """
    # The bracket keeps the pattern from matching the shell that runs it.
    pattern = f"[e]ngine --launch .*uniserve-follower-{os.getpid()}-"
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"pkill -KILL -o -f {shlex.quote(pattern)}",
        ],
        check=True,
        timeout=60,
    )


def test_a_group_over_two_hosts_promotes_its_standby_when_a_hosts_rank_is_lost(
    tmp_path: Path,
):
    if FOLLOWER is None or LEADER_ADDRESS is None:
        pytest.skip(
            "a worker group over two hosts needs UNISERVE_H3_FOLLOWER and "
            "UNISERVE_H3_LEADER_ADDRESS"
        )
    checkpoint = str(host_asset("UNISERVE_QWEN3_MODEL", QWEN3_CHECKPOINT))
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    address = f"{LEADER_ADDRESS}:{find_free_port()}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        checkpoint,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
        "--host-identity",
        "rank-0",
        "--nnodes",
        "2",
        "--dist-init-addr",
        address,
        # Rank 0 on this host and rank 1 on the second: the shorthand
        # spreads the group's ranks over this host and the hosts after it in
        # equal blocks.
        "--tensor-parallel-size",
        "2",
        "--set",
        "device_config.worker_hosts=rank-1",
        "--set",
        "dtype=bfloat16",
        "--set",
        f"attention_backend={ATTENTION_BACKEND}",
        "--set",
        "kv_cache_config.max_tokens=32768",
        "--set",
        "chat_config.reasoning_parser=none",
        "--set",
        "cuda_graph_config.prefill_token_sizes=[512]",
        "--set",
        "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
        "--set",
        "worker_config.standby=true",
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": granted_devices(1),
        "XDG_RUNTIME_DIR": str(runtime),
    }
    log = tmp_path / "server.log"
    streamed: list[str] = []
    usage = None
    try:
        with (
            remote_program_server(FOLLOWER) as follower_programs,
            second_host(
                FOLLOWER,
                tmp_path / "follower.log",
                address,
                leader_log=log,
                arguments=(
                    "--set",
                    f"program_config.socket={follower_programs}",
                ),
            ),
            server_process(command, base_url, log, timeout_s=1800, env=env),
        ):
            _await_line(
                log,
                "the standby epoch of the failure domain is parked",
                timeout_s=1800,
            )
            with httpx.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=_body(stream=True),
                timeout=1800,
            ) as response:
                response.raise_for_status()
                killed = False
                for line in response.iter_lines():
                    tokens, chunk_usage, done = _chunk_tokens(line)
                    if done:
                        break
                    streamed.extend(tokens)
                    usage = chunk_usage or usage
                    if not killed and len(streamed) >= EARLY_TOKENS:
                        _kill_serving_peer(FOLLOWER)
                        killed = True
            reply = httpx.post(
                f"{base_url}/v1/chat/completions",
                json=_body(stream=False),
                timeout=900,
            )
            reply.raise_for_status()
            answer = reply.json()
            reference = [
                entry["token"]
                for entry in answer["choices"][0]["logprobs"]["content"]
            ]
            reference_usage = answer["usage"]
    finally:
        stop_storage_server(runtime)
        storage_log = runtime / "uniserve" / "storage-server.log"
        if storage_log.is_file():
            shutil.copy(storage_log, tmp_path / "storage-server.log")
        shutil.rmtree(runtime, ignore_errors=True)

    lines = _log_lines(log)

    def logged(text: str) -> bool:
        return any(text in line for line in lines)

    assert logged("starting an engine whose worker group spans hosts")
    assert logged("the failure domain's standby epoch was promoted")
    assert not logged("the engine was lost; starting another")
    assert logged("a request migrated"), "the streaming request did not migrate"
    assert streamed == reference
    assert usage == reference_usage
