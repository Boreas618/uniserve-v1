"""The host page tier keeps a prompt's cache pages across a head restart.

Qwen3 answers a long prompt twice, the head restarts, and the prompt is
asked again. The second answer reuses the prompt's pages from the device
cache; the restarted head's device cache is empty, so the pages come back
from the host page tier of the host's Storage Server, which outlives the
head. Both reuse every reusable prompt token, as the prefix cache's hit
counter reports, and both compute the same last tokens over the same cached
keys and values, so they decode the same greedy tokens.

The test runs a Storage Server of its own: ``XDG_RUNTIME_DIR`` points the
head at a socket of the test's, and the server the head starts there is
stopped at the end.
"""

from __future__ import annotations

import os
import signal
import time
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

SERVED_MODEL = "qwen3-page-tier"
BLOCK_SIZE = 64
MAX_TOKENS = 32
# About five thousand tokens of records, then a question about them.
PROMPT = (
    " ".join(
        f"Record {i}: the value of item {i * 7919 % 1000} is "
        f"{i * 104729 % 997}."
        for i in range(260)
    )
    + " Summarize the first three records in one sentence."
)


def _prefix_hits(base_url: str) -> float:
    """The prefix cache's cumulative hit tokens."""
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("uniserve:prefix_cache_hits")
    )


def _ask(base_url: str) -> tuple[int, list[str], float]:
    """Decode the prompt greedily.

    Returns its prompt tokens, the decoded tokens, and the prefix cache hit
    tokens the request added.
    """
    before = _prefix_hits(base_url)
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "max_completion_tokens": MAX_TOKENS,
            "logprobs": True,
            "return_tokens_as_token_ids": True,
        },
        timeout=300,
    )
    response.raise_for_status()
    reply = response.json()
    tokens = [
        entry["token"] for entry in reply["choices"][0]["logprobs"]["content"]
    ]
    # The scheduler's statistics reach the metrics every ten seconds.
    deadline = time.monotonic() + 30
    hits = _prefix_hits(base_url) - before
    while hits == 0 and time.monotonic() < deadline:
        time.sleep(1)
        hits = _prefix_hits(base_url) - before
    return reply["usage"]["prompt_tokens"], tokens, hits


def _stop_storage_server(socket: Path) -> None:
    """Stop the Storage Server serving `socket`.

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
        if b"serve" in command and str(socket).encode() in command:
            os.kill(int(entry.name), signal.SIGTERM)


def test_a_restarted_head_reads_a_prompts_pages_from_the_host_tier(
    tmp_path: Path,
):
    checkpoint = os.environ.get("UNISERVE_QWEN3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_MODEL must name the model checkpoint")

    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
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
        "--set",
        "dtype=bfloat16",
        "--set",
        "kv_cache_config.max_tokens=32768",
        "--set",
        f"kv_cache_config.block_size={BLOCK_SIZE}",
        "--set",
        "chat_config.reasoning_parser=none",
    ]
    env = {"CUDA_VISIBLE_DEVICES": "0", "XDG_RUNTIME_DIR": str(runtime)}
    try:
        with server_process(
            command, base_url, tmp_path / "first.log", timeout_s=900, env=env
        ):
            prompt_tokens, _, cold_hits = _ask(base_url)
            _, device, device_hits = _ask(base_url)
        restarted = tmp_path / "restarted.log"
        with server_process(
            command, base_url, restarted, timeout_s=900, env=env
        ):
            _, host, host_hits = _ask(base_url)
    finally:
        _stop_storage_server(runtime / "uniserve" / "storage.sock")

    # Reusable pages leave one prompt token to compute.
    reusable = (prompt_tokens - 1) // BLOCK_SIZE * BLOCK_SIZE
    assert (cold_hits, device_hits, host_hits) == (0, reusable, reusable)
    assert host and host == device
