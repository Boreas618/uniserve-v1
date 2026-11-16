"""What the end-to-end cases ask a deployment's client API, and how they
read its answers.

One helper per question, shared by every case: a deployment's metrics
(``/metrics``, read with Prometheus' own text parser), a streamed answer's
events, and the deployment files a case writes.
"""  # noqa: D205

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import tomli_w
from prometheus_client.parser import text_string_to_metric_families


def metric(base_url: str, name: str, **labels: str) -> float:
    """The sum of the samples of the metric family `name` on ``/metrics``
    over its label sets, or over those holding every label of `labels`, such
    as ``mode="eager"``. A counter's family is named without its ``_total``
    suffix, which its samples carry.
    """  # noqa: D205
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    total = 0.0
    # The samples are matched by their own names, whatever family the parser
    # files them under: a counter's samples carry the `_total` suffix its
    # `TYPE` line does not, and the text parser then files them under a
    # family of their own name rather than the counter's.
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name not in (name, f"{name}_total"):
                continue
            if all(
                sample.labels.get(key) == value for key, value in labels.items()
            ):
                total += sample.value
    return total


def post_sse(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 600.0,
) -> list[dict[str, Any]]:
    """Every event of the streamed answer to `payload` at `endpoint`, each
    the JSON value of its ``data`` lines, and ``{"type": "sse_done"}`` for the
    ``[DONE]`` that ends an OpenAI stream.

    The answer is read as Server-Sent Events: an event's ``data:`` lines are
    joined by newlines (a single space after the colon dropped) and a blank
    line, or the end of the stream, dispatches it; comments and other fields
    are ignored.
    """  # noqa: D205
    events: list[dict[str, Any]] = []
    data: list[str] = []

    def dispatch() -> None:
        if not data:
            return
        text = "\n".join(data)
        data.clear()
        events.append(
            {"type": "sse_done"} if text == "[DONE]" else json.loads(text)
        )

    with httpx.stream(
        "POST", f"{base_url}{endpoint}", json=payload, timeout=timeout_s
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                dispatch()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").removeprefix(" "))
    dispatch()
    return events


def written_deployment(
    directory: Path, groups: list[Mapping[str, Any]]
) -> Path:
    """Write the deployment file a ``serve`` invocation is given with
    ``--config``: each entry of `groups` is one ``[[worker_groups]]`` table,
    such as ``{"id": "model", "ranks": ["cuda:0-1"], "queue_depth": 2,
    "blocks": {"model": {"tensor_parallel_size": 2}}}``.
    """  # noqa: D205
    path = directory / "deployment.toml"
    path.write_text(tomli_w.dumps({"worker_groups": list(groups)}))
    return path


def dedicated_experts(directory: Path) -> tuple[str, ...]:
    """The options of an expert epoch of two worker groups on a case's two
    GPUs: attention on the first and the routed experts on the second,
    exchanging tokens through DeepEP, whose deployment file is written into
    `directory`.
    """  # noqa: D205
    groups = [
        {
            "id": "attention",
            "ranks": ["cuda:0"],
            "queue_depth": 2,
            "blocks": {"model": {}},
        },
        {
            "id": "experts",
            "role": "experts",
            "ranks": ["cuda:1"],
            "queue_depth": 2,
        },
    ]
    return (
        "--config",
        str(written_deployment(directory, groups)),
        "--set",
        'moe_config.exchange="deepep"',
    )
