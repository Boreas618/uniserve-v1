"""A BF16 model serving from a KV cache of the other half precision.

Qwen3 with BF16 weights serves once with a BF16 cache and once with an FP16
cache (``--kv-cache-dtype float16``), with CUDA graphs captured for prefill and
decode. The FP16 deployment's attention reads
and writes its pages in FP16, so its greedy outputs differ from the BF16
deployment's only by rounding. They are compared by the rule of vLLM's
``check_logprobs_close``: the two outputs of a prompt agree token for token
until their first difference, where each output's token is among the other's
``TOP_LOGPROBS`` most likely tokens.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
)

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
    tmp_path: Path, checkpoint: str, label: str, extra: list[str]
) -> list[list[tuple[str, set[str]]]]:
    """Serve the model with `extra` arguments and decode every prompt.

    Returns each prompt's generated tokens, each with the set of its
    position's most likely tokens, as ``token_id:<id>`` strings.
    """
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
        "--worker-python",
        sys.executable,
        "--set",
        "dtype=bfloat16",
        "--set",
        "kv_cache_config.max_tokens=16384",
        "--max-model-len",
        "4096",
        "--max-running-requests",
        "8",
        "--set",
        "kv_cache_config.block_size=64",
        "--set",
        "cuda_graph_config.capture_prefill=true",
        "--set",
        "chat_config.reasoning_parser=none",
        *extra,
    ]
    outputs = []
    with server_process(
        command,
        base_url,
        tmp_path / f"{label}.log",
        timeout_s=900,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    ):
        for prompt in PROMPTS:
            response = httpx.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": SERVED_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    # No candidate is masked, so every reported candidate
                    # carries the model's own probability.
                    "top_p": 1.0,
                    "top_k": 0,
                    "min_p": 0.0,
                    "max_completion_tokens": MAX_TOKENS,
                    "logprobs": True,
                    "top_logprobs": TOP_LOGPROBS,
                    "return_tokens_as_token_ids": True,
                },
                timeout=300,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["logprobs"]["content"]
            outputs.append(
                [
                    (
                        entry["token"],
                        {top["token"] for top in entry["top_logprobs"]},
                    )
                    for entry in content
                ]
            )
    (tmp_path / f"{label}.json").write_text(
        json.dumps([[token for token, _ in output] for output in outputs])
    )
    return outputs


def test_an_fp16_cache_serves_a_bf16_model_like_a_bf16_cache(tmp_path: Path):
    checkpoint = os.environ.get("UNISERVE_QWEN3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_MODEL must name the model checkpoint")

    reference = _generate(tmp_path, checkpoint, "bfloat16-cache", [])
    converted = _generate(
        tmp_path,
        checkpoint,
        "float16-cache",
        ["--set", "kv_cache_config.dtype=float16"],
    )

    for prompt, expected, actual in zip(
        PROMPTS, reference, converted, strict=True
    ):
        assert actual, prompt
        for position, ((token, top), (other, other_top)) in enumerate(
            zip(expected, actual)
        ):
            if token == other:
                continue
            assert token in other_top and other in top, (
                f"{prompt!r} diverges at token {position}: {token} is not "
                f"among the FP16 cache's top tokens {sorted(other_top)}, or "
                f"{other} not among the BF16 cache's {sorted(top)}"
            )
            break
