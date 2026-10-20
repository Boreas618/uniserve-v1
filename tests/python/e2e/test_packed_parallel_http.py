"""Public generation on ordered tensor, sequence and pipeline assignments.

Each assignment's greedy chat outputs agree with the outputs of the same
model on one device: the tokens are equal, or at their first difference each
side's token is among the other side's five most likely tokens, as a change
in reduction order may break a near tie. The sequences are not compared past
that position.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    png_size_from_b64,
    require_uniserve_binary,
    server_process,
    tiny_input_png_b64,
    written_deployment,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

# Candidates each generated position reports for the comparison.
TOP_LOGPROBS = 5


def _command(
    description: str,
    checkpoint: str,
    port: int,
    directory: Path,
    worker_config: dict[str, Any],
) -> list[str]:
    # A step writes an input image's vision block whole, and a request whose
    # block exceeds the step budget is refused. BAGEL's block holds 1371
    # tokens.
    step_tokens = {"qwen3": "1024", "bagel": "2048", "sensenova": "4096"}[
        description
    ]
    return [
        str(require_uniserve_binary()),
        "serve",
        checkpoint,
        "--served-model-name",
        description,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-python",
        sys.executable,
        "--worker-groups",
        str(written_deployment(directory, [worker_config])),
        "--set",
        "dtype=bfloat16",
        "--set",
        "kv_cache_config.max_tokens="
        + ("65536" if description == "sensenova" else "8192"),
        "--max-model-len",
        "32768" if description == "sensenova" else "4096",
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
    if description in {"bagel", "sensenova"}:
        chats["image"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the dominant color."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/png;base64,{tiny_input_png_b64()}"
                            )
                        },
                    },
                ],
            }
        ]
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
            port = find_free_port()
            base_url = f"http://127.0.0.1:{port}"
            worker_config = {
                "id": "model",
                "ranks": ["cuda:3"],
                "components": {"model": {"ranks": [0]}},
                "queue_depth": 2,
            }
            with server_process(
                _command(
                    description, checkpoint, port, directory, worker_config
                ),
                base_url,
                directory / f"{description}.log",
                timeout_s=300,
                env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            ):
                outputs[description] = _ranked(base_url, description)
        return outputs[description]

    return ranked


@pytest.mark.parametrize(
    ("description", "model_env"),
    [
        pytest.param(
            "qwen3", "UNISERVE_QWEN3_MODEL", marks=pytest.mark.model("qwen3")
        ),
        pytest.param(
            "bagel", "UNISERVE_BAGEL_MODEL", marks=pytest.mark.model("bagel")
        ),
        pytest.param(
            "sensenova",
            "UNISERVE_SENSENOVA_MODEL",
            marks=pytest.mark.model("sensenova"),
        ),
    ],
)
@pytest.mark.parametrize("parallel_axis", ["tensor", "pipeline", "sequence"])
def test_ordered_bindings_generate_outputs(
    tmp_path: Path,
    description: str,
    model_env: str,
    parallel_axis: str,
    one_device: Callable[[str, str], dict[str, list[Any]]],
):
    checkpoint = os.environ.get(model_env, "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail(f"{model_env} must name the model checkpoint directory")
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    parallel = (
        {"sequence_parallel": {"kind": "ulysses", "ulysses_degree": 2}}
        if parallel_axis == "sequence"
        else {f"{parallel_axis}_parallel_size": 2}
    )
    worker_config = {
        "id": "model",
        "ranks": [f"cuda:{device}" for device in (3, 1)],
        "components": {"model": {"ranks": [1, 0], **parallel}},
        "queue_depth": 2,
    }
    image_size = (512, 512) if description == "bagel" else (2048, 1152)
    with server_process(
        _command(description, checkpoint, port, tmp_path, worker_config),
        base_url,
        tmp_path / f"{description}-{parallel_axis}.log",
        timeout_s=300,
        env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    ):
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
            assert message.get("content") or message.get("reasoning_content")

        if description in {"bagel", "sensenova"}:
            generated = httpx.post(
                f"{base_url}/v1/images/generations",
                json={
                    "model": description,
                    "prompt": "A blue square.",
                    "size": f"{image_size[0]}x{image_size[1]}",
                    "steps": 1,
                    "seed": 7,
                },
                timeout=180,
            )
            generated.raise_for_status()
            images = generated.json()["data"]
            assert len(images) == 1
            assert png_size_from_b64(images[0]["b64_json"]) == image_size

        ranked = _ranked(base_url, description)

    expected = one_device(description, checkpoint)
    for name, tokens in ranked.items():
        _assert_close(
            tokens, expected[name], f"{description} {parallel_axis} {name}"
        )
