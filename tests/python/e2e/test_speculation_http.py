"""Speculation with the drafter: synthetic acceptance and the expert union.

Qwen3.5-35B-A3B is served with its multi-token-prediction drafter, three
drafts a verification (``speculative_config.num_draft_tokens``), greedily.

- Synthetic acceptance (``speculative_config.synthetic_acceptance_length``,
  ``docs/configuration.md``) accepts the draft at offset ``i`` of a
  verification with the rate ``clamp(L - 1 - i, 0, 1)`` from its request's
  draw at the draft's position, whatever the model selects there, so a
  verification that carries at least ``ceil(L - 1)`` drafts commits ``L``
  tokens on average. The case reads the engine's own statistics of the
  verifications that carried drafts: the tokens they committed
  (``uniserve_spec_verify_committed_tokens_total``) over their count
  (``uniserve_spec_verify_rows_total``). It serves ``L = 2.5`` and then moves
  the length to ``3.0`` with ``PUT /admin/policy``, a step value that the
  schedulers take at their next step without a generation; each request
  generates ``1 + 3 n`` tokens, so every verification of the second phase
  carries at least two drafts and commits exactly three.
- Two data-parallel schedulers whose ranks share their experts
  (``data_parallel_size = 2`` with ``moe_config.expert_parallel``) speculate
  across their expert union: every step of the union is the model's expert
  layers and the drafter's, once per drafter forward, on both ranks, a rank
  without a batch joining them. Greedy requests, sent at once so that both
  ranks verify in some steps and one joins in others, fail none, carry
  drafts, and answer as the same union answers them without speculation by
  the rule of vLLM's ``check_logprobs_close``
  (:func:`uniserve_testing.chat.assert_logprobs_close`): the tokens are
  equal, or at their first difference each side's token is among the other
  side's five most likely, since a verification reduces its token rows in
  another order than a decode does and may break a near tie.

The schedulers' statistics reach the observation store (``/admin/query``) in a
report once a report period (``worker_config.report_period``, a second), so
the case reads a phase's counts once they held still over several reports.
The prefix cache is off: a request whose prompt reuses a cached prefix
without its draft cache would verify no drafts (the model keeps a recurrent
state, which refills none).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from uniserve_testing import admin, chat
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments
from uniserve_testing.packages import Config, merged
from uniserve_testing.servers import Server

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

NAME = "qwen3.5-speculation"
DRAFTS = 3
#: The options every deployment of the cases states: graphs of the batches
#: the cases send, so that the model and its graph pools fit one GPU's share
#: beside each other.
OPTIONS: Config = {
    "model": CHECKPOINTS["qwen3.5"].model,
    "served_model_name": NAME,
    "max_running_requests": 64,
    "speculative_config": {"num_draft_tokens": DRAFTS},
    "kv_cache_config": {"prefix_caching": False, "host_tier": False},
    "cuda_graph_config": {
        "prefill_token_sizes": [2048],
        "decode_batch_sizes": [1, 2, 4, 8, 16, 32],
    },
}
COMMITTED = "uniserve_spec_verify_committed_tokens_total"
VERIFIED = "uniserve_spec_verify_rows_total"
#: The requests of one phase of the synthetic case, sent at once, and the
#: tokens each generates: one from its prompt, then a multiple of three.
SYNTHETIC_REQUESTS = 32
SYNTHETIC_TOKENS = 1 + 3 * 64
#: The successive reports, a report period apart, that must state the same
#: counts before a phase's are read, and how long they may take, in seconds.
STILL_READS = 3
STILL_S = 120
#: The greedy requests the expert union answers with and without speculation,
#: their tokens, the likeliest tokens each position ranks, and the union's
#: options.
PROMPTS = [
    "Explain how a binary search works, step by step.",
    "Write a short poem about the sea at night.",
    "List the first ten prime numbers and say why each is prime.",
    "Describe the water cycle to a ten-year-old.",
    "What are the differences between TCP and UDP?",
    "Give three tips for writing readable code.",
    "Summarize the plot of a heist story in four sentences.",
    "Translate 'good morning, how are you?' into French and Spanish.",
]
ANSWER_TOKENS = 96
TOP_LOGPROBS = 5
UNION: Config = {
    "data_parallel_size": 2,
    "moe_config": {"expert_parallel": True},
}


def _ask(
    server: Server, prompt: str, tokens: int, seed: int, **fields: Any
) -> dict[str, Any]:
    """The greedy answer to `prompt`, exactly `tokens` tokens long, of a
    request seeded `seed` with `fields` beside: the seed keys its synthetic
    acceptance draws.
    """  # noqa: D205
    request = chat.body(
        NAME,
        prompt,
        tokens,
        **chat.NO_THINKING,
        temperature=0,
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


def _phase(server: Server, phase: int) -> tuple[float, float]:
    """Serves one phase of the synthetic case: the verifications that
    carried drafts during it and their mean committed tokens.
    """  # noqa: D205
    before = _counts(server)
    prompts = [
        f"Request {phase}-{index}: write a long story about a lighthouse."
        for index in range(SYNTHETIC_REQUESTS)
    ]
    answers = _ask_all(server, prompts, SYNTHETIC_TOKENS)
    for answer in answers:
        assert answer["usage"]["completion_tokens"] == SYNTHETIC_TOKENS, answer

    after = _counts(server)
    verified = after[0] - before[0]
    assert verified > 0, "no verification carried drafts"
    return verified, (after[1] - before[1]) / verified


@pytest.mark.gpus(1)
def test_synthetic_acceptance_commits_its_length_and_changes_at_a_step(
    deployments: Deployments,
):
    with deployments.serve(
        merged(
            OPTIONS,
            {"speculative_config": {"synthetic_acceptance_length": 2.5}},
        )
    ) as server:
        verified, mean = _phase(server, 0)
        # A verification commits two tokens and a third with probability one
        # half, one Bernoulli draw: five standard errors of the mean, and the
        # last verification of a request, which may carry one draft and
        # commit two, at most once a request.
        noise = 5 * 0.5 / math.sqrt(verified)
        tail = 0.5 * SYNTHETIC_REQUESTS / verified
        assert abs(mean - 2.5) <= noise + tail, (mean, verified)

        # The length is a step value: the change converges in place.
        applied = admin.revision(
            admin.change(
                server,
                {"speculative_config": {"synthetic_acceptance_length": 3.0}},
            )
        )
        assert applied["effect"] == "step", applied

        # Every verification now carries at least two drafts, accepts both
        # and commits the model's token after them.
        verified, mean = _phase(server, 1)
        assert mean == 3.0, (mean, verified)


@pytest.mark.gpus(2)
def test_an_expert_union_speculates_as_it_decodes(deployments: Deployments):
    def answers(
        name: str, speculative: Config
    ) -> tuple[list[list[tuple[str, set[str]]]], float]:
        # Each deployment is named for what it serves, under which its Build
        # Server keeps what it compiles.
        options = merged(OPTIONS, UNION, {"speculative_config": speculative})
        with deployments.serve(options, name=name) as server:
            replies = _ask_all(
                server,
                PROMPTS,
                ANSWER_TOKENS,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
            )

            # No request failed.
            for reply in replies:
                assert reply["choices"][0]["finish_reason"] == "length", reply

            verified = _counts(server)[0]
            return [chat.ranked_tokens(reply) for reply in replies], verified

    speculated, verified = answers("union", {"num_draft_tokens": DRAFTS})
    decoded, _ = answers("union-decoding", {"num_draft_tokens": 0})

    # The speculating union's verifications carried drafts, and each answer
    # agrees with the union's answer without speculation.
    assert verified > 0, "no verification carried drafts"
    for index, (expected, actual) in enumerate(
        zip(decoded, speculated, strict=True)
    ):
        assert len(actual) == len(expected) == ANSWER_TOKENS, index
        chat.assert_logprobs_close(expected, actual, f"prompt {index}")
