"""What the end-to-end cases ask a deployment's client API, and how they
read its answers.

One helper per question, shared by every case: a streamed answer's events,
and the deployment layouts a case's packages state. What a deployment
measures is read through its admin socket (:mod:`uniserve_testing.admin`).
"""  # noqa: D205

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx


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


def layout(groups: list[Mapping[str, Any]]) -> dict[str, Any]:
    """The options of a deployment layout: each entry of `groups` is one
    ``[[worker_groups]]`` table of the package's ``config.toml``, such as
    ``{"id": "model", "ranks": ["cuda:0-1"], "queue_depth": 2, "blocks":
    {"model": {"tensor_parallel_size": 2}}}``.
    """  # noqa: D205
    return {"worker_groups": [dict(group) for group in groups]}


def dedicated_experts() -> dict[str, Any]:
    """The options of an expert epoch of two worker groups on a case's two
    GPUs: attention on the first and the routed experts on the second,
    exchanging tokens through the package that fills the ``expert_exchange``
    operator's keys, the Build Server's selection by manifest priority
    (``docs/configuration.md``, the ``[kernels]`` table).
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
    return layout(groups)
