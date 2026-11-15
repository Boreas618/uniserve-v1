"""Generation on tensor, sequence and pipeline ranks listed out of order.

One worker group of two ranks serves Qwen3 with a parallel degree of two
along one axis, its ranks listed in descending device order: the order of a
group's ranks assigns each rank its position in the parallel degrees
(``docs/configuration.md``, Deployment files). Each configuration's greedy
chat outputs agree with the outputs of the same model on one device: the
tokens are equal, or at their first difference each side's token is among
the other side's five most likely tokens, as a change in reduction order may
break a near tie. The sequences are not compared past that position.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    ATTENTION_BACKEND,
    QWEN3_CHECKPOINT,
    engine_selection,
    granted_devices,
    host_asset,
    require_uniserve_binary,
    server_process,
    written_deployment,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    # The one-device reference is served once for the module, so the module
    # holds both GPUs its cases bind.
    pytest.mark.gpus(2),
]

# Candidates each generated position reports for the comparison.
TOP_LOGPROBS = 5


def _command(
    description: str,
    checkpoint: str,
    directory: Path,
    group: dict[str, Any],
) -> list[str]:
    # Qwen3 states the step budget its attention keys are built for.
    step_tokens = "1024"
    return [
        str(require_uniserve_binary()),
        "serve",
        "--model-dir",
        checkpoint,
        "--served-model-name",
        description,
        "--host",
        "127.0.0.1",
        *engine_selection(),
        "--config",
        str(written_deployment(directory, [group])),
        "--set",
        'dtype="bfloat16"',
        "--set",
        f'attention_backend="{ATTENTION_BACKEND}"',
        "--set",
        "kv_cache_config.max_tokens=8192",
        "--max-model-len",
        "4096",
        "--set",
        f"max_num_batched_tokens={step_tokens}",
        "--set",
        "scheduler_config.chunked_prefill_size=512",
        "--max-running-requests",
        "1",
    ]


def _chats(description: str) -> dict[str, list[dict[str, Any]]]:
    """The conversations a model answers: text, and an image if it reads one."""
    chats: dict[str, list[dict[str, Any]]] = {
        "text": [{"role": "user", "content": "Name a primary color."}]
    }
    return chats


def _ranked(base_url: str, description: str) -> dict[str, list[Any]]:
    """Each chat's greedy tokens with their most likely candidates."""
    ranked = {}
    for name, messages in _chats(description).items():
        # The candidates rank the model's whole distribution: a top-k or
        # nucleus cutoff from the checkpoint's generation defaults would
        # leave fewer than TOP_LOGPROBS tokens with a finite score.
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": description,
                "messages": messages,
                "temperature": 0,
                "top_k": 0,
                "top_p": 1.0,
                "max_completion_tokens": 8,
                "logprobs": True,
                "top_logprobs": TOP_LOGPROBS,
            },
            timeout=120,
        )
        response.raise_for_status()
        ranked[name] = [
            (
                position["token"],
                {candidate["token"] for candidate in position["top_logprobs"]},
            )
            for position in response.json()["choices"][0]["logprobs"]["content"]
        ]
    return ranked


def _assert_close(
    observed: list[Any], expected: list[Any], context: str
) -> None:
    for index, ((token, candidates), (wanted, alternatives)) in enumerate(
        zip(observed, expected, strict=False)
    ):
        if token == wanted:
            continue
        assert token in alternatives and wanted in candidates, (
            f"{context}: position {index} generated {token!r}, one device "
            f"{wanted!r}"
        )
        return
    assert len(observed) == len(expected), (
        f"{context}: {len(observed)} tokens, one device {len(expected)}"
    )


@pytest.fixture(scope="module")
def one_device(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str, str], dict[str, list[Any]]]:
    """Serve a model on one device once and keep its ranked chat outputs."""
    outputs: dict[str, dict[str, list[Any]]] = {}

    def ranked(description: str, checkpoint: str) -> dict[str, list[Any]]:
        if description not in outputs:
            directory = tmp_path_factory.mktemp(f"{description}-one-device")
            group = {
                "id": "model",
                "ranks": ["cuda:0"],
                "blocks": {"model": {"ranks": [0]}},
                "queue_depth": 2,
            }
            with server_process(
                _command(description, checkpoint, directory, group),
                directory / f"{description}.log",
                timeout_s=300,
                env={"CUDA_VISIBLE_DEVICES": granted_devices(1)},
            ) as server:
                base_url = server.base_url
                outputs[description] = _ranked(base_url, description)
        return outputs[description]

    return ranked


@pytest.mark.parametrize(
    ("description", "model_env"),
    [
        pytest.param(
            "qwen3", "UNISERVE_QWEN3_MODEL", marks=pytest.mark.model("qwen3")
        ),
    ],
)
@pytest.mark.parametrize("parallel_axis", ["tensor", "pipeline", "sequence"])
def test_ranks_out_of_order_generate_the_one_device_outputs(
    tmp_path: Path,
    description: str,
    model_env: str,
    parallel_axis: str,
    one_device: Callable[[str, str], dict[str, list[Any]]],
):
    checkpoint = str(host_asset(model_env, QWEN3_CHECKPOINT))
    parallel = (
        {"sequence_parallel": {"kind": "ulysses", "ulysses_degree": 2}}
        if parallel_axis == "sequence"
        else {f"{parallel_axis}_parallel_size": 2}
    )
    # The module's two GPUs, listed in descending order.
    group = {
        "id": "model",
        "ranks": ["cuda:1", "cuda:0"],
        "blocks": {"model": {"ranks": [1, 0], **parallel}},
        "queue_depth": 2,
    }
    with server_process(
        _command(description, checkpoint, tmp_path, group),
        tmp_path / f"{description}-{parallel_axis}.log",
        timeout_s=300,
        env={"CUDA_VISIBLE_DEVICES": granted_devices(2)},
    ) as server:
        base_url = server.base_url
        for name, messages in _chats(description).items():
            response = httpx.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": description,
                    "messages": messages,
                    "temperature": 0,
                    "max_completion_tokens": 8,
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            assert result["model"] == description
            assert result["usage"]["prompt_tokens"] > 0
            assert 1 <= result["usage"]["completion_tokens"] <= 8
            choice = result["choices"][0]
            assert choice["finish_reason"] in ("stop", "length"), name
            message = choice["message"]
            assert message.get("content") or message.get("reasoning")

        ranked = _ranked(base_url, description)

    expected = one_device(description, checkpoint)
    for name, tokens in ranked.items():
        _assert_close(
            tokens, expected[name], f"{description} {parallel_axis} {name}"
        )
