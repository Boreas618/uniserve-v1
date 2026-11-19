"""DeepSeek-V4.1-Flash on the four GPUs of one host, against its own reference.

The deployment serves the release as four data-parallel schedulers whose ranks
share their routed experts (``data_parallel_size = 4`` with expert
parallelism): each rank holds the attention, the shared experts and the dense
weights whole, its window of 96 of every layer's 384 MXFP4 experts behind the
expert exchange, and a quarter of the n-gram tables' hash heads, whose lookups
gather over the union. Requests render through the family's port of the
checkpoint's reference encoder (``processing::deepseek_v41``); the cases ask
for chat mode (``enable_thinking`` false), whose prompt ends with
``<｜Assistant｜></think>``, so a short answer is the answer itself.

Two cases:

- **Teacher-forced log-probabilities.** The reference vectors are written by
  ``crates/models/tests/it/deepseek_v41/generate_logprobs.py`` from the
  checkpoint's own ``inference/`` code on the same four GPUs: for fixed prompts
  of 1K to 16K tokens, every position's log-probability of the next prompt
  token and its 64 most likely tokens. The case sends each prompt's text as a
  user message and asks for every prompt position's 64 most likely tokens
  (``prompt_logprobs``); the served prompt must be the reference's token ids.
  Per position it states the KL divergence of the served distribution from the
  reference's over the tokens both rank among their 64 most likely and the
  rest of the vocabulary as one outcome (a lower bound of the full divergence,
  which the remainder's mass keeps tight), the difference of the next token's
  log-probabilities, and whether both rank the same token first. The bound is
  the reference's own spread across two of its compositions (a prefill of the
  whole prompt, and the last 32 tokens decoded one at a time), which the
  generator states in its manifest: the mean divergence of every prompt within
  :data:`SPREAD_FACTOR` times the largest mean the reference shows against
  itself, and top-1 agreement at least the reference's lowest less
  :data:`TOP1_SLACK`.
- **Greedy generation.** Greedy chat completions of short questions name
  their answers, and a request asked twice is answered with the same tokens.

The vectors live under ``artifacts/deepseek_v41/reference`` of the repository
(Git-ignored); the first case is skipped without them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from safetensors.numpy import load_file
from uniserve_testing import chat
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("deepseek-v41"),
    pytest.mark.gpus(4),
]

SERVED_MODEL = "deepseek-v41-flash"

#: The reference vectors of ``generate_logprobs.py``.
REFERENCE = (
    Path(__file__).resolve().parents[3] / "artifacts/deepseek_v41/reference"
)

#: The candidates each scored prompt position states, the reference's count.
CANDIDATES = 64

#: The deployment's options: the longest reference prompt fits its context,
#: and a prefill takes chunks of 4096 tokens.
OPTIONS: dict[str, Any] = {
    "served_model_name": SERVED_MODEL,
    "data_parallel_size": 4,
    "max_model_len": 20480,
    "max_running_requests": 16,
    "max_num_batched_tokens": 4096,
    "max_logprobs": CANDIDATES,
    # The weights take 125 GiB of a GPU's 185: the default share of 0.7 does
    # not hold them with the graph pool and the KV cache.
    "mem_fraction_static": 0.95,
    "moe_config": {"expert_parallel": True},
    "cuda_graph_config": {
        "decode_batch_sizes": [1, 2, 4, 8],
        "prefill_token_sizes": [1024, 4096],
    },
}

#: How many times the reference's own largest mean divergence the served
#: mean of a prompt may reach. The reference's two compositions share every
#: kernel and differ in the order of a few reductions; the served model runs
#: other kernels, every reduction in its own order, and computes the routed
#: experts' weighting and their sum with the shared expert in another order
#: (``docs/deepseek_v41.md``). The factor was fixed from the reference's
#: spread before the served model was measured.
SPREAD_FACTOR = 4.0

#: How much lower than the reference's own lowest top-1 agreement the
#: served agreement of a prompt may be.
TOP1_SLACK = 0.02


@pytest.fixture(scope="module")
def server(module_deployments: Deployments) -> Server:
    """The module's deployment."""
    with module_deployments.serve(
        {"model": CHECKPOINTS["deepseek-v41"].model, **OPTIONS}
    ) as served:
        yield served


def _divergence(
    reference: dict[int, float], rest: float, served: dict[int, float]
) -> float:
    """KL(reference ‖ served) over the tokens both rank and the rest of the
    vocabulary as one outcome.

    `reference` and `served` map a token id to its log-probability, and
    `rest` is the log of the reference's mass outside its own ranked tokens,
    so the reference's remainder is exact; the served remainder is one less
    the served mass of the shared tokens.
    """  # noqa: D205
    shared = reference.keys() & served.keys()
    divergence = sum(
        math.exp(reference[token]) * (reference[token] - served[token])
        for token in shared
    )
    rest_p = math.exp(rest) + sum(
        math.exp(value)
        for token, value in reference.items()
        if token not in shared
    )
    rest_q = max(1.0 - sum(math.exp(served[token]) for token in shared), 1e-30)
    if rest_p > 0.0:
        divergence += rest_p * math.log(rest_p / rest_q)
    return max(divergence, 0.0)


def _served(answer: dict[str, Any]) -> list[dict[int, float]]:
    """Each scored prompt position's candidates of `answer`, by token id:
    entry ``i`` scores prompt token ``i + 1``.
    """  # noqa: D205
    return [
        {
            int(token.removeprefix("token_id:")): value
            for token, value in entry.items()
        }
        for entry in answer["prompt_logprobs"][1:]
    ]


def test_teacher_forced_log_probabilities_are_within_the_references_spread(
    server: Server,
):
    manifest_path = REFERENCE / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip(
            f"no reference vectors at {REFERENCE}: run generate_logprobs.py"
        )
    manifest = json.loads(manifest_path.read_text())
    prompts = manifest["prompts"]
    bound = SPREAD_FACTOR * max(
        prompt["decode"]["kl_mean"] for prompt in prompts
    )
    agreement = min(prompt["decode"]["top1_agreement"] for prompt in prompts)
    agreement -= TOP1_SLACK

    report = []
    for prompt in prompts:
        vectors = load_file(REFERENCE / prompt["file"])
        ids = vectors["ids"].tolist()
        answer = chat.complete(
            server.base_url,
            chat.body(
                SERVED_MODEL,
                prompt["text"],
                1,
                **chat.GREEDY,
                **chat.NO_THINKING,
                prompt_logprobs=CANDIDATES,
                return_tokens_as_token_ids=True,
                return_token_ids=True,
            ),
        )
        assert answer["prompt_token_ids"] == ids, prompt["file"]

        served = _served(answer)
        divergences, deltas, agreed = [], [], 0
        for position, candidates in enumerate(served):
            reference = dict(
                zip(
                    vectors["top_ids"][position].tolist(),
                    vectors["top_logprobs"][position].tolist(),
                    strict=True,
                )
            )
            rest = float(vectors["top_rest_logprob"][position])
            divergences.append(_divergence(reference, rest, candidates))
            target = ids[position + 1]
            deltas.append(
                abs(
                    candidates[target]
                    - float(vectors["next_logprob"][position])
                )
            )
            first = max(candidates, key=candidates.__getitem__)
            agreed += first == int(vectors["argmax"][position])

        count = len(served)
        row = {
            "prompt": prompt["file"],
            "tokens": len(ids),
            "mean_kl": sum(divergences) / count,
            "max_kl": max(divergences),
            "mean_next_delta": sum(deltas) / count,
            "max_next_delta": max(deltas),
            "top1": agreed / count,
        }
        report.append(row)
        print(json.dumps(row))

    print(json.dumps({"kl_bound": bound, "top1_bound": agreement}))
    for row in report:
        assert row["mean_kl"] <= bound, (row, bound)
        assert row["top1"] >= agreement, (row, agreement)


def test_greedy_chat_names_its_answers_and_repeats_itself(server: Server):
    questions = {
        "What is the capital of France? Answer with one word.": "paris",
        "How many legs does a spider have? Answer with a number.": "8",
        "Which planet is called the Red Planet? Answer with one word.": "mars",
    }
    for question, expected in questions.items():
        request = chat.body(
            SERVED_MODEL, question, 24, **chat.GREEDY, **chat.NO_THINKING
        )
        first = chat.complete(server.base_url, request)
        text = chat.content(first)
        print(json.dumps({"question": question, "answer": text}))
        assert expected in text.lower(), (question, text)

        again = chat.complete(server.base_url, request)
        assert chat.content(again) == text, (question, text)
