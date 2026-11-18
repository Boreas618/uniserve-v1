"""A BF16 model serving from a KV cache of the other half precision.

Qwen3 with BF16 weights serves once with a BF16 cache and once with an FP16
cache (``kv_cache_config.dtype=float16``), with CUDA graphs captured for
prefill and decode. The FP16 deployment's attention reads and writes its KV
blocks in FP16, so its greedy outputs differ from the BF16 deployment's only
by rounding. They are compared by the rule of vLLM's
``check_logprobs_close`` (:func:`uniserve_testing.chat.assert_logprobs_close`):
the two outputs of a prompt agree token for token until their first
difference, where each output's token is among the other's ``TOP_LOGPROBS``
most likely tokens.
"""

from __future__ import annotations

import pytest
from uniserve_testing import chat
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments
from uniserve_testing.packages import Config, merged

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

SERVED_MODEL = "qwen3"
PROMPTS = (
    "Explain why the sky is blue in three sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "List five capital cities in Europe and their countries.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is the derivative of x^3 * sin(x)?",
    "Translate 'good morning, how are you?' into French and Spanish.",
    "Give three tips for writing clear technical documentation.",
    "Describe how a hash table handles collisions.",
)
MAX_TOKENS = 96
# vLLM's model correctness tests compare logprobs at five candidates.
TOP_LOGPROBS = 5


def _generate(
    deployments: Deployments, name: str, options: Config
) -> list[list[tuple[str, set[str]]]]:
    """Serve the model with `options` and decode every prompt.

    Returns each prompt's generated tokens, each with the set of its
    position's most likely tokens, as ``token_id:<id>`` strings.
    """
    config = merged(
        {
            "model": CHECKPOINTS["qwen3"].model,
            "served_model_name": SERVED_MODEL,
        },
        options,
    )
    with deployments.serve(config, name=name) as server:
        return [
            chat.ranked_tokens(
                chat.complete(
                    server.base_url,
                    chat.body(
                        SERVED_MODEL,
                        prompt,
                        MAX_TOKENS,
                        # No candidate is masked, so every reported
                        # candidate carries the model's own probability.
                        **chat.GREEDY,
                        logprobs=True,
                        top_logprobs=TOP_LOGPROBS,
                        return_tokens_as_token_ids=True,
                    ),
                )
            )
            for prompt in PROMPTS
        ]


def test_an_fp16_cache_serves_a_bf16_model_like_a_bf16_cache(
    deployments: Deployments,
):
    reference = _generate(deployments, "bfloat16-cache", {})
    converted = _generate(
        deployments,
        "float16-cache",
        {"kv_cache_config": {"dtype": "float16"}},
    )

    for prompt, expected, actual in zip(
        PROMPTS, reference, converted, strict=True
    ):
        assert actual, prompt
        chat.assert_logprobs_close(
            expected, actual, f"{prompt!r} on a BF16 and an FP16 cache"
        )
