"""What the end-to-end cases ask a deployment's client API, and how they
read its answers.

One helper per question, shared by every case: a deployment's metrics
(``/metrics``, read with Prometheus' own text parser), a streamed answer's
events (the evaluator's SSE reader), and the deployment files a case writes.
"""  # noqa: D205

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import tomli_w
from prometheus_client.parser import text_string_to_metric_families

from uniserve_eval.transport.sse import iter_sse_events


def metric(base_url: str, name: str, **labels: str) -> float:
    """The sum of the samples of the metric family `name` on ``/metrics``
    over its label sets, or over those holding every label of `labels`, such
    as ``mode="eager"``. A counter's family is named without its ``_total``
    suffix, which its samples carry.
    """  # noqa: D205
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    total = 0.0
    for family in text_string_to_metric_families(text):
        if family.name != name:
            continue
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
    """Every event of the streamed answer to `payload` at `endpoint`."""
    with httpx.stream(
        "POST", f"{base_url}{endpoint}", json=payload, timeout=timeout_s
    ) as response:
        response.raise_for_status()
        return list(iter_sse_events(response.iter_lines()))


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
