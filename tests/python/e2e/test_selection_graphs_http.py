"""Captured selections serve every sampling control as the eager path does.

Qwen3 serves the same requests twice, with CUDA graphs on one GPU and
eagerly on another, both prefilling a prompt of about a thousand tokens in
chunks of 256 tokens. Under graphs a forward's token selection, the
logprobs it reports and the scores of its prompt are captured graphs whose
inputs are tensors: token sets as bitmasks, each request's row of logit
biases, its sampler and parameters, and each prompt row's position, target
and place in its chunk's words; a continued chunk's first token is scored
from the logits its request retained from the chunk before. Eagerly the
same methods run on the builder thread.

Constraints hold exactly under graphs: allowed tokens, a dominating logit
bias, reported top logprobs that rank the sampled token, and prompt scores
for every prompt token after the first. Over a prompt extended to whole
chunks, so that every prefill forward has the same shape on both
deployments, the two then report the same prompt scores within float
rounding (``PROMPT_TOLERANCE``) and select the same tokens. A forward
padded to a captured bucket differs from its eager counterpart beyond
rounding, since the mixture-of-experts router may pick other experts for
it; so does a decode forward, whose attention runs captured kernels under
graphs and planned ones eagerly: the sampled tokens' logprobs agree within
``DECODE_TOLERANCE``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
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
# The largest difference between the two deployments' prompt scores, in
# nats, when their prefill forwards have the same shapes; below 1e-4
# measured.
PROMPT_TOLERANCE = 1e-3
# The largest difference between their sampled tokens' logprobs, in nats:
# four times the largest measured over both requests (0.061).
DECODE_TOLERANCE = 0.25
# A filler word, one token, that extends the prompt to whole chunks.
FILLER = " ok"
# The GPUs of the deployment with graphs and of the eager one.
DEVICES_ENV = "UNISERVE_SELECTION_DEVICES"


def _checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_QWEN3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_MODEL must name the model checkpoint")
    return checkpoint


def _serve(
    tmp_path_factory: pytest.TempPathFactory, policy: str, device: str
) -> Iterator[str]:
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    args = [
        str(require_uniserve_binary()),
        "serve",
        _checkpoint(),
        "--served-model-name",
        f"qwen3-selection-{policy}",
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
        "--set",
        f"cuda_graph_config.policy={policy}",
    ]
    log = tmp_path_factory.mktemp(f"selection-{policy}") / "server.log"
    with server_process(
        args,
        base_url,
        log,
        timeout_s=1800.0,
        env={"CUDA_VISIBLE_DEVICES": device},
    ):
        yield base_url


def _device(index: int) -> str:
    return os.environ.get(DEVICES_ENV, "0,1").split(",")[index]


@pytest.fixture(scope="module")
def graphs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    yield from _serve(tmp_path_factory, "full", _device(0))


@pytest.fixture(scope="module")
def eager(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    yield from _serve(tmp_path_factory, "off", _device(1))


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


@pytest.fixture(scope="module")
def aligned(graphs: str) -> str:
    """The prompt, extended with filler words to whole prefill chunks."""
    prompt = PROMPT
    for _ in range(4):
        reply = _chat(graphs, prompt, max_completion_tokens=1)
        missing = -reply["usage"]["prompt_tokens"] % CHUNK
        if missing == 0:
            return prompt
        prompt += FILLER * missing
    pytest.fail("the filler does not extend the prompt to whole chunks")


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
# Greedy over shaped logits: penalties, a top-k beyond the decode graph's
# candidates, a bias and bad words, which the general sampler applies.
SHAPED = {
    "temperature": 0,
    "top_k": 300,
    "top_p": 0.9,
    "repetition_penalty": 1.3,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "bad_words": ["the", " The"],
    "logprobs": True,
    "top_logprobs": 2,
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


@pytest.mark.parametrize("body", [GREEDY, SHAPED], ids=["greedy", "shaped"])
def test_graphs_select_and_score_as_eager_forwards_do(
    graphs: str, eager: str, aligned: str, body: dict[str, Any]
) -> None:
    captured = _chat(graphs, aligned, **body)
    called = _chat(eager, aligned, **body)

    assert _tokens(captured) == _tokens(called)
    sampled = []
    for ours, theirs in zip(
        captured["choices"][0]["logprobs"]["content"],
        called["choices"][0]["logprobs"]["content"],
        strict=True,
    ):
        sampled.append(abs(ours["logprob"] - theirs["logprob"]))
        assert (
            ours["top_logprobs"][0]["token"]
            == (theirs["top_logprobs"][0]["token"])
        )

    print(f"largest sampled logprob difference: {max(sampled):.6f}")
    assert max(sampled) <= DECODE_TOLERANCE

    if "prompt_logprobs" in body:
        scores = []
        for ours, theirs in zip(
            captured["prompt_logprobs"][1:],
            called["prompt_logprobs"][1:],
            strict=True,
        ):
            # Each position scores its prompt token on both.
            shared = ours.keys() & theirs.keys()
            assert shared
            scores.extend(abs(ours[key] - theirs[key]) for key in shared)
        print(f"largest prompt score difference: {max(scores):.6f}")
        assert max(scores) <= PROMPT_TOLERANCE
