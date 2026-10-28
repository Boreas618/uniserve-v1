"""The host KV cache keeps a hybrid-attention prompt's pages across a restart.

DiffusionGemma interleaves sliding-window layers, which read the 1023
tokens before each token, with full-attention layers, so its cache has a
group of each attention type. Its full-attention pages hold 64 tokens and
its sliding-window pages 32 (a token's full-attention row is half as wide,
and the groups' planes take the same bytes), so a reuse ends at a multiple
of 64 tokens. The host KV cache of the host's Storage Server, which
outlives the engine, keeps every full-attention page of a prompt and the
sliding window before each aligned boundary: every ``boundary_interval``
pages of 64 tokens (32 here, 2048 tokens, so the windows kept before them
leave gaps), the last boundary a reissue of the prompt reuses, and the last
boundary of its full pages.

The first engine answers a long prompt twice: the second answer reuses
every reusable prompt token from the device cache. The engine restarts,
with an empty device cache:

- A prompt that shares only the first records of the long one reuses its
  pages up to the last aligned boundary before they differ: the
  full-attention pages go further, but the window before each deeper
  boundary falls in a gap between the windows kept, and the reuse passes
  over those boundaries. Asked again, it reuses
  every reusable token from the device cache, and both answers are the
  same.
- The long prompt reuses every reusable token again, its window before the
  last boundary loaded from the host KV cache, and answers as the first
  engine did from its device cache.

The answers are seeded, so the same cached keys and values denoise the same
reply. The prefix cache's hit counter reports the reused tokens.

The checkpoint directory comes from ``UNISERVE_DIFFUSION_GEMMA_MODEL``. The
test runs a Storage Server of its own: ``XDG_RUNTIME_DIR`` points the
daemons at a directory of the test's, and the server the first daemon
starts there is stopped at the end.
"""

from __future__ import annotations

import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    gpu_device,
    require_uniserve_binary,
    server_process,
    worker_executable,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("diffusion_gemma"),
]

SERVED_MODEL = "diffusion-gemma-host-kv-cache"
# Tokens of a full-attention page, the largest block size: every reuse ends
# at a multiple of it.
STEP = 64
BOUNDARY_INTERVAL = 32
# Host storage the test's host KV cache may take, in GiB (per host).
HOST_GIB = 32
INTERVAL_TOKENS = BOUNDARY_INTERVAL * STEP
# The tokens of a prompt after the records it shares with another: the
# question and the chat template's end of the turn.
QUESTION_TOKENS = 64
MAX_TOKENS = 64
SEED = 7


def _records(count: int) -> str:
    return " ".join(
        f"Record {i}: the value of item {i * 7919 % 1000} is "
        f"{i * 104729 % 997}."
        for i in range(count)
    )


# About five thousand tokens of records, then a question about them.
LONG = _records(260) + " Summarize the first three records in one sentence."
# The first 160 records of the long prompt, then another question: about
# three thousand tokens shared.
SHARED = _records(160) + " Which record holds the largest value?"


def _prefix_hits(base_url: str) -> float:
    """The prefix cache's cumulative hit tokens."""
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("uniserve:prefix_cache_hits")
    )


def _ask(base_url: str, prompt: str) -> tuple[int, str, float]:
    """Answer `prompt` with the seeded reply.

    Returns its prompt tokens, the reply, and the prefix cache hit tokens
    the request added.
    """
    before = _prefix_hits(base_url)
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": MAX_TOKENS,
            "seed": SEED,
        },
        timeout=600,
    )
    assert response.status_code == 200, response.text
    reply = response.json()
    # The scheduler's statistics reach the metrics every ten seconds.
    deadline = time.monotonic() + 30
    hits = _prefix_hits(base_url) - before
    while hits == 0 and time.monotonic() < deadline:
        time.sleep(1)
        hits = _prefix_hits(base_url) - before
    content = reply["choices"][0]["message"]["content"]
    return reply["usage"]["prompt_tokens"], content, hits


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
    checkpoint = os.environ.get("UNISERVE_DIFFUSION_GEMMA_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_DIFFUSION_GEMMA_MODEL must name the checkpoint")
    return checkpoint


def _reusable(prompt_tokens: int) -> int:
    """The prompt tokens a reuse may cover: one is always computed."""
    return (prompt_tokens - 1) // STEP * STEP


def test_a_restarted_engine_loads_full_pages_and_aligned_windows(
    tmp_path: Path,
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
        "--set",
        f"kv_cache_config.boundary_interval={BOUNDARY_INTERVAL}",
        # The host KV cache holds one prompt of at most 8192 tokens; left at
        # twice the device's token capacity it would pin hundreds of GiB of
        # the host, which test runs share.
        "--set",
        f"kv_cache_config.host_gib={HOST_GIB}",
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": gpu_device(),
        "XDG_RUNTIME_DIR": str(runtime),
    }
    try:
        with server_process(
            command, base_url, tmp_path / "first.log", timeout_s=1800, env=env
        ):
            long_tokens, _, cold_hits = _ask(base_url, LONG)
            _, device, device_hits = _ask(base_url, LONG)
        with server_process(
            command,
            base_url,
            tmp_path / "restarted.log",
            timeout_s=1800,
            env=env,
        ):
            shared_tokens, shared_host, shared_host_hits = _ask(
                base_url, SHARED
            )
            _, shared_device, shared_device_hits = _ask(base_url, SHARED)
            _, host, host_hits = _ask(base_url, LONG)
    finally:
        _stop_storage_server(runtime / "uniserve" / "storage.sock")
        shutil.rmtree(runtime, ignore_errors=True)

    assert (cold_hits, device_hits) == (0, _reusable(long_tokens))

    # The shared records end before the question: the last aligned boundary
    # before that is the same wherever in its last QUESTION_TOKENS they end.
    shared_end = shared_tokens - QUESTION_TOKENS
    interval = shared_end // INTERVAL_TOKENS * INTERVAL_TOKENS
    assert shared_tokens // INTERVAL_TOKENS * INTERVAL_TOKENS == interval > 0
    assert shared_host_hits == interval
    assert shared_device_hits == _reusable(shared_tokens)
    assert shared_host and shared_host == shared_device

    assert host_hits == _reusable(long_tokens)
    assert host and host == device
