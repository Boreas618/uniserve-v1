"""Captured selections honour every sampling constraint.

Qwen3 serves from CUDA graphs, prefilling a prompt of about a thousand tokens
in chunks of 256 tokens. A forward's token selection, the logprobs it reports
and the scores of its prompt are captured graphs whose inputs are tensors:
token sets as bitmasks, each request's row of logit biases, its sampler and
parameters, and each prompt row's position, target and place in its chunk's
words; a continued chunk's first token is scored from the logits its request
retained from the chunk before.

Constraints hold exactly under graphs: allowed tokens, a dominating logit
bias, reported top logprobs that rank the sampled token, and prompt scores
for every prompt token after the first.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    QWEN3_MOE,
    find_free_port,
    model_dir,
    require_uniserve_binary,
    server_process,
    worker_executable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

CHUNK = 256
# About a thousand tokens of records, then a question about them: four
# prefill chunks.
PROMPT = (
    " ".join(
        f"Record {i}: item {i * 7919 % 1000} is worth {i * 104729 % 997}."
        for i in range(80)
    )
    + " Which record is worth the most? Answer in one sentence."
)


def _checkpoint() -> str:
    checkpoint = str(model_dir(QWEN3_MOE))
    return checkpoint


@pytest.fixture(scope="module")
def graphs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    args = [
        str(require_uniserve_binary()),
        "serve",
        _checkpoint(),
        "--served-model-name",
        "qwen3-selection",
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
        f"max_num_batched_tokens={2 * CHUNK}",
        "--set",
        f"scheduler_config.chunked_prefill_size={CHUNK}",
    ]
    log = tmp_path_factory.mktemp("selection") / "server.log"
    with server_process(
        args,
        base_url,
        log,
        timeout_s=1800.0,
    ):
        yield base_url


def _chat(base_url: str, prompt: str = PROMPT, **body: Any) -> dict[str, Any]:
    model = httpx.get(f"{base_url}/v1/models", timeout=30).json()["data"][0]
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model["id"],
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 16,
            "return_tokens_as_token_ids": True,
            **body,
        },
        timeout=600,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _tokens(reply: dict[str, Any]) -> list[str]:
    return [
        entry["token"] for entry in reply["choices"][0]["logprobs"]["content"]
    ]


# Greedy, reporting five top tokens and three per prompt position.
GREEDY = {
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "logprobs": True,
    "top_logprobs": 5,
    "prompt_logprobs": 3,
}
# Seeded sampling through the general sampler.
SAMPLED = {
    "temperature": 0.8,
    "top_k": 300,
    "top_p": 0.95,
    "min_p": 0.02,
    "seed": 11,
    "logprobs": True,
    "top_logprobs": 1,
}


def test_constraints_hold_under_graphs(graphs: str) -> None:
    greedy = _chat(graphs, **GREEDY)
    tokens = _tokens(greedy)
    for entry in greedy["choices"][0]["logprobs"]["content"]:
        # Greedy over the model's distribution selects its top token.
        top = entry["top_logprobs"]
        assert top[0]["token"] == entry["token"]
        assert top[0]["logprob"] == entry["logprob"]
        assert [candidate["logprob"] for candidate in top] == sorted(
            (candidate["logprob"] for candidate in top), reverse=True
        )

    # Every prompt token after the first is scored, among at most three
    # other candidates.
    prompt = greedy["prompt_logprobs"]
    assert len(prompt) == greedy["usage"]["prompt_tokens"]
    assert prompt[0] is None
    assert all(1 <= len(position) <= 4 for position in prompt[1:])
    assert all(
        value <= 0.0 for position in prompt[1:] for value in position.values()
    )

    # Tasks restricted to two tokens select only those.
    allowed = sorted({int(token.split(":")[1]) for token in tokens[:4]})[:2]
    restricted = _chat(
        graphs,
        **SAMPLED,
        allowed_token_ids=allowed,
        ignore_eos=True,
        max_completion_tokens=8,
    )
    assert {int(token.split(":")[1]) for token in _tokens(restricted)} <= set(
        allowed
    )

    # A bias of a hundred makes its token the greedy selection.
    forced = int(tokens[-1].split(":")[1])
    biased = _chat(
        graphs,
        **GREEDY,
        logit_bias={str(forced): 100.0},
        max_completion_tokens=2,
    )
    assert _tokens(biased)[0] == f"token_id:{forced}"
