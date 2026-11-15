"""The host tier keeps a prompt's KV blocks across a restart of the engine.

Qwen3 answers a long prompt twice, the deployment is stopped and started
again, and the prompt is asked once more. Every full block the first answer
fills is committed under its key, and the scheduler's write policy stores it
to the host tier (``specs/buffer-pool.md`` §13, ``specs/scheduler-design.md``
§4.4): a host block of the deployment's buffer pool in the host's Storage
Server, which outlives the engine. The second answer pins the blocks on the
device; the restarted engine's session opens device regions that hold no
block, so its admission loads them from the host. Both reuse every block the
reuse limit admits (the prompt less its last token, in whole blocks), as the
prefix cache's counters report per tier, and both compute the last tokens over
the same cached keys and values, so they decode the same greedy tokens.

A prompt answered with one token ends with the step that fills its blocks,
and no batch follows it: the copies its stores need are sent to the executor
in a batch of their own (``specs/buffer-pool.md`` §10), and the copies still in
flight when the deployment stops are given their grace time (§20). The
restarted engine loads its blocks from the host all the same.

Each test runs a Storage Server of its own: ``XDG_RUNTIME_DIR`` points the
daemons at a directory of the test's, and the server the first daemon starts
there is stopped at the end.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    ATTENTION_BACKEND,
    engine_selection,
    granted_devices,
    qwen3_checkpoint,
    require_uniserve_binary,
    runtime_directory,
    server_process,
    tokens_found,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

SERVED_MODEL = "qwen3-host-kv-cache"
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


def _ask(
    base_url: str, prompt: str = PROMPT, max_tokens: int = MAX_TOKENS
) -> tuple[int, list[str], dict[str, float]]:
    """Decode `prompt` greedily, at most `max_tokens` tokens.

    Returns its prompt tokens, the decoded tokens, and the tokens its
    admission found cached, per tier (:func:`tokens_found`).
    """

    def ask() -> dict:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_completion_tokens": max_tokens,
                "logprobs": True,
                "return_tokens_as_token_ids": True,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    reply, found = tokens_found(base_url, ask)
    tokens = [
        entry["token"] for entry in reply["choices"][0]["logprobs"]["content"]
    ]
    return reply["usage"]["prompt_tokens"], tokens, found


def _reusable(prompt_tokens: int) -> int:
    """The tokens an admission may reuse: the reuse limit, the prompt less
    its last token in whole blocks.
    """  # noqa: D205
    return (prompt_tokens - 1) // BLOCK_SIZE * BLOCK_SIZE


def _command(checkpoint: str) -> list[str]:
    return [
        str(require_uniserve_binary()),
        "serve",
        checkpoint,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        *engine_selection(),
        "--set",
        'dtype="bfloat16"',
        "--set",
        f'attention_backend="{ATTENTION_BACKEND}"',
        "--set",
        "kv_cache_config.max_tokens=32768",
        "--set",
        f"kv_cache_config.block_size={BLOCK_SIZE}",
        "--set",
        'chat_config.reasoning_parser="none"',
    ]


def test_a_restarted_engine_loads_a_prompts_blocks_from_the_host_tier(
    tmp_path: Path,
):
    command = _command(qwen3_checkpoint())
    with runtime_directory() as runtime:
        env = {
            "CUDA_VISIBLE_DEVICES": granted_devices(1),
            "XDG_RUNTIME_DIR": str(runtime),
        }
        with server_process(
            command, tmp_path / "first.log", timeout_s=900, env=env
        ) as server:
            base_url = server.base_url
            prompt_tokens, _, cold = _ask(base_url)
            _, device, warm = _ask(base_url)
        restarted = tmp_path / "restarted.log"
        with server_process(
            command, restarted, timeout_s=900, env=env
        ) as server:
            base_url = server.base_url
            _, host, loaded = _ask(base_url)

    reusable = _reusable(prompt_tokens)
    assert reusable > 0, prompt_tokens
    # Cold, nothing is cached; warm, every reusable block is pinned on the
    # device; restarted, every one is loaded from the host tier.
    assert (cold["hits"], cold["host_hits"]) == (0, 0), cold
    assert (warm["hits"], warm["host_hits"]) == (reusable, 0), warm
    assert (loaded["hits"], loaded["host_hits"]) == (reusable, reusable), loaded
    assert host and host == device


def test_a_stopping_engine_stores_its_last_blocks_to_the_host_tier(
    tmp_path: Path,
):
    # Another prompt than the other test's, whose blocks a shared server
    # might still hold.
    prompt = "Closing. " + PROMPT
    command = _command(qwen3_checkpoint())
    with runtime_directory() as runtime:
        env = {
            "CUDA_VISIBLE_DEVICES": granted_devices(1),
            "XDG_RUNTIME_DIR": str(runtime),
        }
        with server_process(
            command, tmp_path / "first.log", timeout_s=900, env=env
        ) as server:
            base_url = server.base_url
            prompt_tokens, first, cold = _ask(base_url, prompt, 1)
        with server_process(
            command,
            tmp_path / "restarted.log",
            timeout_s=900,
            env=env,
        ) as server:
            base_url = server.base_url
            _, host, loaded = _ask(base_url, prompt, 1)

    reusable = _reusable(prompt_tokens)
    assert cold["hits"] == 0, cold
    assert (loaded["hits"], loaded["host_hits"]) == (reusable, reusable), loaded
    assert host == first
