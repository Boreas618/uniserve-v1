"""Generation on tensor, sequence and pipeline ranks listed out of order.

One worker group of two ranks serves Qwen3 with a parallel degree of two
along one axis, its ranks listed in descending device order: the order of a
group's ranks assigns each rank its position in the parallel degrees
(``docs/configuration.md``, Deployment files). Each configuration's greedy
chat outputs agree with the outputs of the same model on one device by the
rule of vLLM's ``check_logprobs_close``
(:func:`uniserve_testing.chat.assert_logprobs_close`): the tokens are equal,
or at their first difference each side's token is among the other side's
five most likely tokens, as a change in reduction order may break a near
tie. The sequences are not compared past that position, and two outputs
that agree on every position both reached have the same length.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest
from uniserve_testing import chat, client
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    # The one-device reference is served once for the module, so the module
    # holds both GPUs its cases bind.
    pytest.mark.gpus(2),
]

SERVED_MODEL = "qwen3"
PROMPT = "Name a primary color."
MAX_TOKENS = 8
# Candidates each generated position reports for the comparison.
TOP_LOGPROBS = 5

# A position's sampled token with its most likely candidates.
Ranked = list[tuple[str, set[str]]]


def _serve(
    deployments: Deployments,
    directory: Path,
    group: dict[str, Any],
    name: str,
) -> AbstractContextManager[Server]:
    """Serve the model as the one worker group `group`, its deployment file
    in `directory` and its log named `name`, until the block ends.
    """  # noqa: D205
    return deployments.serve(
        "--served-model-name",
        SERVED_MODEL,
        "--config",
        str(client.written_deployment(directory, [group])),
        model=CHECKPOINTS["qwen3"].model,
        name=name,
    )


def _ranked(base_url: str) -> Ranked:
    """The prompt's greedy tokens with their most likely candidates.

    The candidates rank the model's whole distribution: a top-k or nucleus
    cutoff from the checkpoint's generation defaults would leave fewer than
    ``TOP_LOGPROBS`` tokens with a finite score.
    """
    answer = chat.complete(
        base_url,
        chat.body(
            SERVED_MODEL,
            PROMPT,
            MAX_TOKENS,
            **chat.GREEDY,
            logprobs=True,
            top_logprobs=TOP_LOGPROBS,
        ),
    )
    return chat.ranked_tokens(answer)


@pytest.fixture(scope="module")
def one_device(
    module_deployments: Deployments,
    tmp_path_factory: pytest.TempPathFactory,
) -> Ranked:
    """The model's ranked output on one device, served once for the module."""
    group = {
        "id": "model",
        "ranks": ["cuda:0"],
        "blocks": {"model": {"ranks": [0]}},
        "queue_depth": 2,
    }
    with _serve(
        module_deployments,
        tmp_path_factory.mktemp("one-device"),
        group,
        "one-device",
    ) as server:
        return _ranked(server.base_url)


@pytest.mark.parametrize("parallel_axis", ["tensor", "pipeline", "sequence"])
def test_ranks_out_of_order_generate_the_one_device_outputs(
    deployments: Deployments,
    tmp_path: Path,
    parallel_axis: str,
    one_device: Ranked,
):
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
    with _serve(deployments, tmp_path, group, parallel_axis) as server:
        result = chat.complete(
            server.base_url,
            chat.body(SERVED_MODEL, PROMPT, MAX_TOKENS, temperature=0),
        )
        assert result["model"] == SERVED_MODEL
        assert result["usage"]["prompt_tokens"] > 0
        assert 1 <= result["usage"]["completion_tokens"] <= MAX_TOKENS
        choice = result["choices"][0]
        assert choice["finish_reason"] in ("stop", "length")
        message = choice["message"]
        assert message.get("content") or message.get("reasoning")

        ranked = _ranked(server.base_url)

    context = f"{parallel_axis} parallel ranks against one device"
    chat.assert_logprobs_close(one_device, ranked, context)
    tokens = [token for token, _ in ranked]
    wanted = [token for token, _ in one_device]
    if tokens[: len(wanted)] == wanted[: len(tokens)]:
        # No position differs, so neither output may stop before the other.
        assert len(tokens) == len(wanted), (
            f"{context}: {len(tokens)} tokens, one device {len(wanted)}"
        )
