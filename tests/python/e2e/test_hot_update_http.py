"""A hot update replaces the engine under a streaming request, failing nothing.

Qwen3 streams a long greedy count to a client. While it streams, the
leader's daemon is asked to update its engine (``POST /engine/update``):
a standby engine starts beside the running one on the same GPU, mapping the
running engine's weights and cache planes without writing them; the running
engine drains, its streaming request leaving it as migratable after every
token it generated; the standby is promoted, and the daemon submits the
request again with those tokens to the new engine, which continues the
count where it stopped. The client reads one uninterrupted stream.

The stream decodes the tokens the new engine decodes for the same request
alone, and its usage charges the same prompt and completion. The drained
engine's process exits. The update reports how long requests waited for
the new engine and each stage of that pause, and the daemon logs how long
the migrated request waited for its first token on the new engine; the test
prints both.

The count is a continuation that no near tie decides, so rounding
differences between the drained engine's decode of the generated tokens and
the new engine's prefill of them cannot change it. The deployment captures
few graphs (decode batches up to 8, no prefill graphs), so that the standby
starts while the count streams.

The standby parks with the device memory of its graph pools and scratch
resident, or suspended (``worker_config.standby_memory``): copied to host
memory and released on the device at unchanged addresses, then mapped back
with its contents at the promotion, before its graphs run. Either way the
new engine's graphs decode the same count.

The deployment runs on one GPU, or as two schedulers that share their
experts (``--data-parallel-size 2`` with expert parallelism): the standby
engine's workers then form an expert world of their own beside the running
one, and a suspended standby also releases its all-to-all workspaces and
NCCL communicators, which every rank of its world restores together at the
promotion.

The test runs a Storage Server of its own: ``XDG_RUNTIME_DIR`` points the
daemon at a directory of the test's, and the server the daemon starts there
is stopped at the end. ``UNISERVE_UPDATE_EXECUTABLE`` names the ``uniserve``
executable of the build to update to, installed in a Python environment of
its own, whose engine the daemon logs as another build; without it, the
engine updates to the build its executable holds.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import tempfile
import threading
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    worker_executable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

SERVED_MODEL = "qwen3-hot-update"
MAX_TOKENS = 16384
# Tokens the client reads before it asks for the update.
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


def _ask(base_url: str) -> tuple[list[str], dict]:
    """Decode the count alone: its tokens and usage."""
    response = httpx.post(
        f"{base_url}/v1/chat/completions", json=_body(stream=False), timeout=900
    )
    response.raise_for_status()
    reply = response.json()
    tokens = [
        entry["token"] for entry in reply["choices"][0]["logprobs"]["content"]
    ]
    return tokens, reply["usage"]


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


def _checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_QWEN3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_MODEL must name the model checkpoint")
    return checkpoint


def _log_lines(log: Path) -> list[str]:
    """The lines of `log` without their terminal colors."""
    colors = re.compile(r"\x1b\[[0-9;]*m")
    return [colors.sub("", line) for line in log.read_text().splitlines()]


def _engine_builds(lines: list[str]) -> set[str]:
    """The builds of the engines that connected to the daemon."""
    connected = re.compile(r"an engine connected .*build=(\S+)")
    return {
        match.group(1) for line in lines if (match := connected.search(line))
    }


# The devices and extra options of each deployment.
TOPOLOGIES = {
    "single": ("0", ()),
    "experts": (
        "0,1",
        (
            "--data-parallel-size",
            "2",
            "--set",
            "moe_config.expert_parallel=true",
        ),
    ),
}


@pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
@pytest.mark.parametrize("standby_memory", ["resident", "suspended"])
def test_a_hot_update_continues_a_streaming_request_on_the_new_engine(
    tmp_path: Path, standby_memory: str, topology: str
):
    devices, options = TOPOLOGIES[topology]
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
        "--set",
        "dtype=bfloat16",
        "--set",
        "kv_cache_config.max_tokens=32768",
        "--set",
        "chat_config.reasoning_parser=none",
        "--set",
        "cuda_graph_config.capture_prefill=false",
        "--set",
        "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
        "--set",
        f"worker_config.standby_memory={standby_memory}",
        *options,
    ]
    env = {"CUDA_VISIBLE_DEVICES": devices, "XDG_RUNTIME_DIR": str(runtime)}
    update_body = {}
    if executable := os.environ.get("UNISERVE_UPDATE_EXECUTABLE"):
        update_body["executable"] = executable
    log = tmp_path / "server.log"
    streamed: list[str] = []
    usage = None
    after_update = 0
    update: dict = {}
    try:
        with server_process(command, base_url, log, timeout_s=900, env=env):
            with httpx.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=_body(stream=True),
                timeout=1800,
            ) as response:
                response.raise_for_status()
                updater = None
                for line in response.iter_lines():
                    tokens, chunk_usage, done = _chunk_tokens(line)
                    if done:
                        break
                    streamed.extend(tokens)
                    usage = chunk_usage or usage
                    if updater is None and len(streamed) >= EARLY_TOKENS:
                        updater = threading.Thread(
                            target=lambda: update.update(
                                httpx.post(
                                    f"{base_url}/engine/update",
                                    json=update_body,
                                    timeout=1800,
                                ).json()
                            )
                        )
                        updater.start()
                    elif updater is not None and not updater.is_alive():
                        after_update += len(tokens)
                assert updater is not None, "the stream ended before the update"
                updater.join()
            reference, reference_usage = _ask(base_url)
    finally:
        _stop_storage_server(runtime / "uniserve" / "storage.sock")
        shutil.rmtree(runtime, ignore_errors=True)

    assert "pause_seconds" in update, update
    print(
        f"hot update pause: {update['pause_seconds']:.3f} s "
        f"(drain {update['drain_seconds']:.3f} s, "
        f"close {update['close_seconds']:.3f} s, "
        f"promotion {update['promotion_seconds']:.3f} s)"
    )
    lines = _log_lines(log)
    for line in lines:
        if "a migrated request generates" in line:
            print(f"migrated request: {line.split(': ', 1)[-1]}")
    assert any("a request migrated" in line for line in lines), (
        "the streaming request did not migrate to the new engine"
    )
    assert after_update > 0, "the stream ended before the new engine served"
    assert any(
        "the drained engine's process exited" in line for line in lines
    ), "the drained engine's process did not exit"
    # The drain closes the workers, which exit once they gave up their
    # storages; none is stopped at the shutdown deadline.
    assert not any(
        "worker did not exit before the shutdown deadline" in line
        for line in lines
    ), "a drained engine's worker did not exit"
    if "executable" in update_body:
        builds = _engine_builds(lines)
        assert len(builds) == 2, f"the engines ran the builds {builds}"
    if standby_memory == "suspended":
        assert any(
            "the standby worker suspended its graph memory" in line
            for line in lines
        ), "the standby worker kept its graph memory"
        assert any(
            "the promoted standby worker restored its graph memory" in line
            for line in lines
        ), "the promoted standby worker did not restore its graph memory"
        if topology == "experts":
            for event in (
                "the standby worker suspended its expert transports",
                "the standby worker suspended its communicators",
                "the promoted standby worker restored its expert transports",
                "the promoted standby worker resumed its communicators",
            ):
                assert any(event in line for line in lines), event
    else:
        assert any(
            "the standby worker keeps its graph memory resident" in line
            for line in lines
        ), "the standby worker suspended its graph memory"
    assert streamed == reference
    assert usage == reference_usage
