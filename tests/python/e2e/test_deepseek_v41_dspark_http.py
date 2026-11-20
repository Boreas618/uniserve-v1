"""DeepSeek-V4.1-Flash with its DSpark drafter on the four GPUs of one host.

The deployment is the package ``configs/deepseek_v41/dp4-ep4-dspark``: four
data-parallel schedulers whose ranks share their routed experts, each rank
also holding the three DSpark layers (``mtp.*``) whole. A step verifies six
tokens a request: the token committed last and the five drafts of the block
the drafter wrote after it in one forward, which the next block continues
from the drafts the verification accepts (``docs/deepseek_v41.md``).

Three cases, in this order (the first starts the deployment without
speculation, and stops it, before the module's speculating deployment starts):

- **Lossless against greedy decoding.** Greedy requests with real acceptance
  (the drafts a verification's own argmax confirms) answer as the same
  deployment without speculation answers them: the tokens are equal, or at
  their first difference each side's token is among the other side's five
  most likely, since a verification of six tokens reduces in another order
  than a decode of one (vLLM's ``check_logprobs_close``,
  :func:`uniserve_testing.chat.assert_logprobs_close`). The case reports how
  many answers are equal token for token.
- **Real acceptance length.** InferenceX's acceptance-length gate
  (``infx/bench/al_gate.py``): the 80 SPEED-Bench qualitative coding prompts
  of its golden DSpark curve (``speedbench_coding.jsonl``, its digest
  checked), at its collection settings (thinking on with the effort budget
  its renderer wrote for ``high``, 50; temperature 1.0; at most 4096 tokens;
  end-of-sequence honoured; 64 at once), and ``AL = committed tokens per
  verification that carried drafts`` from the engine's own counters. The
  verdict is the gate's: within 5% of the golden 3.51. It reports and does
  not fail the case, as the gate does: the golden curve drafts by sampling
  and verifies by block rejection, and this deployment drafts the argmax and
  verifies each draft by its own rejection.
- **Synthetic acceptance length.** ``synthetic_acceptance_length = 3.51``,
  applied with ``PUT /admin/policy`` as a step value, accepts draft ``i``
  with the rate ``clamp(2.51 - i, 0, 1)``: a verification of five drafts
  commits three tokens and a fourth with probability 0.51.

The schedulers' statistics reach the observation store (``/admin/query``) in
a report once a second, so each phase reads its counts once they held still
over several reports. The prompts live under
``artifacts/deepseek_v41/e3/dspark/speedbench`` of the repository
(Git-ignored, written by InferenceX's ``python -m infx.bench.speedbench``);
the real-acceptance case is skipped without them.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from uniserve_testing import admin, chat
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments
from uniserve_testing.packages import Config, merged
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("deepseek-v41"),
    pytest.mark.gpus(4),
]

SERVED_MODEL = "deepseek-v41-flash"

#: The repository root.
ROOT = Path(__file__).resolve().parents[3]

#: The drafts a verification holds: the drafter's block.
DRAFTS = 5

#: The package's options (``configs/deepseek_v41/dp4-ep4-dspark``), with the
#: candidates the lossless case asks beside each token.
OPTIONS: Config = {
    "model": CHECKPOINTS["deepseek-v41"].model,
    "served_model_name": SERVED_MODEL,
    "data_parallel_size": 4,
    "max_model_len": 1048576,
    "max_running_requests": 16,
    "max_num_batched_tokens": 4096,
    "max_logprobs": 5,
    "mem_fraction_static": 0.95,
    "moe_config": {"expert_parallel": True},
    "cuda_graph_config": {
        "decode_batch_sizes": [1, 2, 4, 8],
        "prefill_token_sizes": [1024, 4096],
    },
    "speculative_config": {"num_draft_tokens": DRAFTS},
}

COMMITTED = "uniserve_spec_verify_committed_tokens_total"
VERIFIED = "uniserve_spec_verify_rows_total"
#: The successive reports, a report period apart, that must state the same
#: counts before a phase's are read, and how long they may take, in seconds.
STILL_READS = 3
STILL_S = 300

#: The lossless case's greedy requests, their tokens and the likeliest
#: tokens each position ranks.
PROMPTS = [
    "Explain how a binary search works, step by step.",
    "Write a Python function that merges two sorted lists, and explain it.",
    "List the first ten prime numbers and say why each is prime.",
    "Describe the water cycle to a ten-year-old.",
    "What are the differences between TCP and UDP?",
    "Give three tips for writing readable code, with an example each.",
    "Summarize the plot of a heist story in four sentences.",
    "Translate 'good morning, how are you?' into French and Spanish.",
]
ANSWER_TOKENS = 256
TOP_LOGPROBS = 5

#: InferenceX's golden DSpark curve of DeepSeek-V4.1-Flash, thinking on, five
#: drafts (``infx/golden_al_distribution/``, the curve
#: ``deepseek-v4.1-flash__dspark__native__probabilistic-block``).
GOLDEN_AL = 3.51
#: The gate's prompts, their digest (``infx/bench/speedbench.py``
#: ``PROMPTS_SHA256``), its replay's concurrency and tokens, and its verdict's
#: band (``infx/bench/al_gate.py``).
SPEEDBENCH = (
    ROOT / "artifacts/deepseek_v41/e3/dspark/speedbench/speedbench_coding.jsonl"
)
SPEEDBENCH_SHA256 = (
    "6f1845452f1cfbc1930ea779ba6c3d5b437a9d5853eca199dd9b7619b20a0c18"
)
GATE_CONCURRENCY = 64
GATE_TOKENS = 4096
GATE_BAND = (0.95, 1.05)
#: The collection's thinking mode. Its renderer (vLLM's ``deepseek_v41``
#: encoding at b47b01cf33, InferenceX's ``encoding_dsv41.py``) wrote the
#: effort ``high`` as the budget 50, which this deployment's request side
#: writes for ``low``: the budget is sent as the number it rendered.
GATE_TEMPLATE = {"thinking": True, "reasoning_effort": 50}
GATE_SAMPLING = {"temperature": 1.0, "repetition_penalty": 1.0}
#: How long one of the gate's completions may take, in seconds: 4096 tokens
#: among 64 running requests.
GATE_COMPLETION_S = 3600.0

#: The synthetic case's requests, sent at once, and the tokens each
#: generates: one from its prompt, then whole verifications of six tokens.
SYNTHETIC_REQUESTS = 32
SYNTHETIC_TOKENS = 1 + (DRAFTS + 1) * 64


def _ask(
    server: Server, prompt: str, tokens: int, seed: int, **fields: Any
) -> dict[str, Any]:
    """The greedy answer to `prompt`, exactly `tokens` tokens long, of a
    request seeded `seed` with `fields` beside.
    """  # noqa: D205
    request = chat.body(
        SERVED_MODEL,
        prompt,
        tokens,
        **chat.GREEDY,
        chat_template_kwargs={"thinking": False},
        ignore_eos=True,
        seed=seed,
        **fields,
    )
    return chat.complete(
        server.base_url, request, server.scaled(chat.COMPLETION_S)
    )


def _ask_all(
    server: Server, prompts: list[str], tokens: int, **fields: Any
) -> list[dict[str, Any]]:
    """The answers to `prompts`, sent at once, each request seeded by its
    place among them.
    """  # noqa: D205
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        answers = pool.map(
            lambda indexed: _ask(
                server, indexed[1], tokens, indexed[0], **fields
            ),
            enumerate(prompts),
        )
        return list(answers)


def _counts(server: Server) -> tuple[float, float]:
    """The verifications that carried drafts so far, and the tokens they
    committed, once :data:`STILL_READS` successive reads, a report period
    apart, state the same counts.
    """  # noqa: D205
    last: tuple[float, float] | None = None
    held = 0

    def still() -> bool:
        nonlocal last, held
        now = (
            admin.counter(server, VERIFIED),
            admin.counter(server, COMMITTED),
        )
        held = held + 1 if now == last else 0
        last = now
        return held >= STILL_READS

    server.wait_until(still, STILL_S, "the verification counts did not settle")
    assert last is not None
    return last


def _greedy(server: Server) -> list[list[tuple[str, set[str]]]]:
    """The lossless case's answers on `server`, each to its token limit."""
    replies = _ask_all(
        server,
        PROMPTS,
        ANSWER_TOKENS,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )
    for reply in replies:
        assert reply["choices"][0]["finish_reason"] == "length", reply
    return [chat.ranked_tokens(reply) for reply in replies]


@pytest.fixture(scope="module")
def decoded(
    module_deployments: Deployments,
) -> list[list[tuple[str, set[str]]]]:
    """The lossless case's answers of the deployment without speculation,
    which stops before the module's speculating deployment starts.
    """  # noqa: D205
    options = merged(OPTIONS, {"speculative_config": {"num_draft_tokens": 0}})
    with module_deployments.serve(
        options, name="deepseek-v41-decoding"
    ) as server:
        return _greedy(server)


@pytest.fixture(scope="module")
def server(module_deployments: Deployments) -> Server:
    """The module's speculating deployment."""
    with module_deployments.serve(
        OPTIONS, name="deepseek-v41-dspark"
    ) as served:
        yield served


# The fixtures run in the order the case names them: the deployment without
# speculation answers and stops before the speculating one starts.
def test_speculation_answers_as_greedy_decoding(
    decoded: list[list[tuple[str, set[str]]]], server: Server
):
    before = _counts(server)
    speculated = _greedy(server)
    verified = _counts(server)[0] - before[0]
    assert verified > 0, "no verification carried drafts"

    equal = 0
    for index, (expected, actual) in enumerate(
        zip(decoded, speculated, strict=True)
    ):
        assert len(actual) == len(expected) == ANSWER_TOKENS, index
        chat.assert_logprobs_close(expected, actual, f"prompt {index}")
        equal += [token for token, _ in expected] == [
            token for token, _ in actual
        ]
    print(
        json.dumps(
            {"answers": len(PROMPTS), "equal": equal, "verified": verified}
        )
    )


def test_real_acceptance_length_against_the_golden_curve(server: Server):
    if not SPEEDBENCH.is_file():
        pytest.skip(
            f"no SPEED-Bench prompts at {SPEEDBENCH}: run infx.bench.speedbench"
        )
    prompts = [
        json.loads(line)["prompt"]
        for line in SPEEDBENCH.read_text(encoding="utf-8").splitlines()
    ]
    digest = hashlib.sha256(
        json.dumps(prompts, ensure_ascii=False).encode()
    ).hexdigest()
    assert digest == SPEEDBENCH_SHA256, digest

    def ask(prompt: str) -> dict[str, Any]:
        request = chat.body(
            SERVED_MODEL,
            prompt,
            GATE_TOKENS,
            chat_template_kwargs=GATE_TEMPLATE,
            **GATE_SAMPLING,
        )
        return chat.complete(
            server.base_url, request, server.scaled(GATE_COMPLETION_S)
        )

    before = _counts(server)
    with ThreadPoolExecutor(max_workers=GATE_CONCURRENCY) as pool:
        answers = list(pool.map(ask, prompts))
    after = _counts(server)

    verified = after[0] - before[0]
    committed = after[1] - before[1]
    assert verified > 0, "no verification carried drafts"
    length = committed / verified
    ratio = length / GOLDEN_AL
    verdict = (
        "below"
        if ratio < GATE_BAND[0]
        else "pass"
        if ratio <= GATE_BAND[1]
        else "above"
    )
    finished = [answer["choices"][0]["finish_reason"] for answer in answers]
    print(
        json.dumps(
            {
                "prompts": len(prompts),
                "completion_tokens": sum(
                    answer["usage"]["completion_tokens"] for answer in answers
                ),
                "finish_reasons": {
                    reason: finished.count(reason) for reason in set(finished)
                },
                "verifications": verified,
                "committed_tokens": committed,
                "acceptance_length": round(length, 4),
                "golden": GOLDEN_AL,
                "ratio": round(ratio, 4),
                "verdict": verdict,
            }
        )
    )
    # Speculation accepted drafts: a verification commits more than its
    # model's own token on average.
    assert length > 1.0, length


def test_synthetic_acceptance_commits_the_golden_length(server: Server):
    applied = admin.revision(
        admin.change(
            server,
            {"speculative_config": {"synthetic_acceptance_length": GOLDEN_AL}},
        )
    )
    assert applied["effect"] == "step", applied

    before = _counts(server)
    prompts = [
        f"Request {index}: write a long story about a lighthouse."
        for index in range(SYNTHETIC_REQUESTS)
    ]
    answers = _ask_all(server, prompts, SYNTHETIC_TOKENS)
    for answer in answers:
        assert answer["usage"]["completion_tokens"] == SYNTHETIC_TOKENS, answer
    after = _counts(server)

    verified = after[0] - before[0]
    assert verified > 0, "no verification carried drafts"
    mean = (after[1] - before[1]) / verified
    print(json.dumps({"verifications": verified, "mean_committed": mean}))

    # A verification of at least three drafts commits three tokens and a
    # fourth with probability 0.51, one Bernoulli draw: five standard errors
    # of the mean. A request's last verification may carry fewer drafts as
    # its limit nears, committing at most 2.51 fewer, once a request.
    noise = 5 * math.sqrt(0.51 * 0.49) / math.sqrt(verified)
    tail = (GOLDEN_AL - 1) * SYNTHETIC_REQUESTS / verified
    assert GOLDEN_AL - noise - tail <= mean <= GOLDEN_AL + noise, (
        mean,
        verified,
    )
