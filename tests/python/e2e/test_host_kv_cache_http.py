"""The host tier keeps a prompt's KV blocks across a restart of the engine.

Qwen3 answers a long prompt twice, the deployment is stopped and started
again, and the prompt is asked once more. Every full block the first answer
fills is committed under its key, and the scheduler's write policy stores it
to the host tier (``specs/buffer-pool.md`` §13, ``specs/scheduler-design.md``
§4.4): a host block of the deployment's buffer pool in the host's Storage
Server, which outlives the engine. The second answer pins the blocks on the
device, and the prompt tokens it states as cached are the prompt's reuse
limit (:mod:`uniserve_testing.caching`). The restarted engine's session opens
device regions that hold no block and the deployment names no other tier, so
whatever its admission finds cached it loads from the host: it reuses as
many tokens as the second answer did, and both compute the last tokens over
the same cached keys and values, so they decode the same greedy tokens.

A prompt answered with one token ends with the step that fills its blocks,
and no batch follows it: the copies its stores need are sent to the executor
in a batch of their own (``specs/buffer-pool.md`` §10), and the copies still in
flight when the deployment stops are given their grace time (§20). The
restarted engine loads its blocks from the host all the same: it reuses as
many tokens as the same prompt asked again on it, once its own answer put
every block on the device.

Each case starts one deployment twice under one runtime directory: a
deployment is its policy package's name, and one that comes down without
``retire`` keeps its Storage Server and the storage it holds there, so the
restarted daemon finds the Storage Server the first start left, which the
case stops at its end. A second deployment on the same GPU would instead
start a Storage Server of its own, which holds no block of the first, beside
the storage the first one retains on the GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from uniserve_testing import chat
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.caching import cached_tokens
from uniserve_testing.deployments import (
    Deployments,
    runtime_directory,
    storage_status,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

#: The deployment each case starts twice, and the model name it serves.
SERVED_MODEL = "qwen3-host-kv-cache"
MAX_TOKENS = 32
# About five thousand tokens of records, then a question about them.
PROMPT = chat.long_prompt(260)


def _ask(base_url: str, prompt: str, max_tokens: int) -> dict:
    """The greedy answer to `prompt` of at most `max_tokens` tokens, each
    reported by its token id, so that two answers compare token for token.
    """  # noqa: D205
    return chat.complete(
        base_url,
        chat.body(
            SERVED_MODEL,
            prompt,
            max_tokens,
            **chat.GREEDY,
            logprobs=True,
            return_tokens_as_token_ids=True,
        ),
    )


def _storage_server(uniserve: Path, runtime: Path) -> int:
    """The process of the deployment's Storage Server under `runtime`, as its
    readiness record states it.
    """  # noqa: D205
    record = storage_status(uniserve, runtime, SERVED_MODEL)
    assert record is not None, "the deployment's Storage Server does not serve"
    return int(record["pid"])


def test_a_restarted_engine_loads_a_prompts_blocks_from_the_host_tier(
    uniserve: Path, deployments: Deployments, tmp_path: Path
):
    config = {
        "model": CHECKPOINTS["qwen3"].model,
        "served_model_name": SERVED_MODEL,
    }
    with runtime_directory(uniserve, tmp_path / "storage") as runtime:
        with deployments.serve(
            config, name=SERVED_MODEL, log="first", runtime=runtime
        ) as server:
            cold = _ask(server.base_url, PROMPT, MAX_TOKENS)
            warm = _ask(server.base_url, PROMPT, MAX_TOKENS)
            storage = _storage_server(uniserve, runtime)
        with deployments.serve(
            config, name=SERVED_MODEL, log="restarted", runtime=runtime
        ) as server:
            # The restart is served beside the Storage Server the first
            # start left, which holds the blocks it stored.
            assert _storage_server(uniserve, runtime) == storage
            loaded = _ask(server.base_url, PROMPT, MAX_TOKENS)

    # Cold, nothing is cached; warm, every reusable block is pinned on the
    # device; restarted, every one is loaded from the host tier.
    reusable = cached_tokens(warm)
    assert reusable > 0, warm["usage"]
    assert cached_tokens(cold) == 0, cold["usage"]
    assert cached_tokens(loaded) == reusable, loaded["usage"]
    assert chat.ranked_tokens(loaded)
    assert chat.ranked_tokens(loaded) == chat.ranked_tokens(warm)


def test_a_stopping_engine_stores_its_last_blocks_to_the_host_tier(
    uniserve: Path, deployments: Deployments, tmp_path: Path
):
    config = {
        "model": CHECKPOINTS["qwen3"].model,
        "served_model_name": SERVED_MODEL,
    }
    with runtime_directory(uniserve, tmp_path / "storage") as runtime:
        with deployments.serve(
            config, name=SERVED_MODEL, log="first", runtime=runtime
        ) as server:
            first = _ask(server.base_url, PROMPT, 1)
            storage = _storage_server(uniserve, runtime)
        with deployments.serve(
            config, name=SERVED_MODEL, log="restarted", runtime=runtime
        ) as server:
            # The restart is served beside the Storage Server the first
            # start left, which holds the blocks it stored.
            assert _storage_server(uniserve, runtime) == storage
            loaded = _ask(server.base_url, PROMPT, 1)
            # The prompt asked again finds every block on the device, which
            # the loaded answer either loaded or computed: its cached tokens
            # are the prompt's reuse limit.
            repeated = _ask(server.base_url, PROMPT, 1)

    reusable = cached_tokens(repeated)
    assert reusable > 0, repeated["usage"]
    assert cached_tokens(first) == 0, first["usage"]
    assert cached_tokens(loaded) == reusable, loaded["usage"]
    assert chat.ranked_tokens(loaded) == chat.ranked_tokens(first)
