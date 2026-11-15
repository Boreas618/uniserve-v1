"""Requests are served by the data-parallel replica the policy routes them to.

Qwen3 is served by two data-parallel schedulers whose ranks share their
experts (``--data-parallel-size 2`` with expert parallelism): one main
process, one scheduler per replica, each on its own GPU. The control thread
asks the epoch's scheduling policy where each request goes (its ``route``),
and the policy the deployment serves names the live replica with the fewest
requests in flight, rotating among equally loaded ones
(``docs/architecture.md``, The request interface). Each replica's scheduler
reports its admissions under its own ``engine`` label
(``uniserve:num_requests_admitted``).

Requests sent one at a time find both replicas idle, so the policy alternates
between them and each admits half. While one replica holds a long streaming
request, every request sent meanwhile finds the other replica with fewer in
flight, so the other replica admits all of them.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

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
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    pytest.mark.gpus(2),
]

SERVED_MODEL = "qwen3-data-parallel"
# Requests sent one at a time; an even count splits evenly.
SEQUENTIAL = 6
# Requests sent while a long one streams on one replica.
BESIDE = 4
ADMITTED = re.compile(
    r'^uniserve:num_requests_admitted_total\{.*engine="(\d+)".*\} (\S+)$'
)
RUNNING = re.compile(
    r'^uniserve:num_requests_running\{.*engine="(\d+)".*\} (\S+)$'
)


def _per_engine(base_url: str, pattern: re.Pattern[str]) -> dict[int, float]:
    """A metric's value for each replica's scheduler."""
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    values: dict[int, float] = {}
    for line in text.splitlines():
        if (found := pattern.match(line)) is not None:
            engine = int(found.group(1))
            values[engine] = values.get(engine, 0.0) + float(found.group(2))
    return values


def _quiet(base_url: str) -> dict[int, float]:
    """Each replica's admissions once nothing runs and the counts, which
    reach the metrics about every second, held for three seconds: the
    serving path's warm-up requests are counted by then.
    """  # noqa: D205
    deadline = time.monotonic() + 120
    last = _per_engine(base_url, ADMITTED)
    held = 0
    while held < 3:
        assert time.monotonic() < deadline, "the replicas did not settle"
        time.sleep(1)
        now = _per_engine(base_url, ADMITTED)
        running = any(_per_engine(base_url, RUNNING).values())
        held = held + 1 if now == last and not running else 0
        last = now
    return last


def _settled(
    base_url: str, before: dict[int, float], total: int
) -> dict[int, float]:
    """The admissions each replica added since `before`, once the schedulers'
    statistics, which reach the metrics about every second, count `total`.
    """  # noqa: D205
    deadline = time.monotonic() + 60
    while True:
        now = _per_engine(base_url, ADMITTED)
        grown = {
            engine: now.get(engine, 0.0) - before.get(engine, 0.0)
            for engine in (0, 1)
        }
        if sum(grown.values()) >= total or time.monotonic() > deadline:
            return grown
        time.sleep(1)


def _request(base_url: str, prompt: str, max_tokens: int) -> httpx.Response:
    return httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_completion_tokens": max_tokens,
            "ignore_eos": True,
        },
        timeout=900,
    )


def test_requests_are_served_by_the_replicas_the_policy_routes_them_to(
    tmp_path: Path,
):
    checkpoint = str(host_asset("UNISERVE_QWEN3_MODEL", QWEN3_CHECKPOINT))
    command = [
        str(require_uniserve_binary()),
        "serve",
        checkpoint,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        *engine_selection(),
        "--data-parallel-size",
        "2",
        "--set",
        "moe_config.expert_parallel=true",
        "--set",
        'dtype="bfloat16"',
        "--set",
        f'attention_backend="{ATTENTION_BACKEND}"',
        "--set",
        "kv_cache_config.max_tokens=32768",
        "--set",
        'chat_config.reasoning_parser="none"',
        "--set",
        "cuda_graph_config.prefill_token_sizes=[512]",
        "--set",
        "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
    ]
    env = {"CUDA_VISIBLE_DEVICES": granted_devices(2)}
    with server_process(
        command, tmp_path / "server.log", 1800, env=env
    ) as server:
        base_url = server.base_url
        # Idle replicas: the policy rotates, and each admits half.
        before = _quiet(base_url)
        for index in range(SEQUENTIAL):
            _request(base_url, f"Say the number {index}.", 4).raise_for_status()
        alternated = _settled(base_url, before, SEQUENTIAL)
        assert alternated == {0: SEQUENTIAL / 2, 1: SEQUENTIAL / 2}, alternated

        # One replica holds a long request: every request beside it is routed
        # to the other one.
        before = _quiet(base_url)
        long_reply: list[httpx.Response] = []
        streaming = threading.Thread(
            target=lambda: long_reply.append(
                _request(base_url, "Count from 1 to 3000.", 3000)
            )
        )
        streaming.start()
        deadline = time.monotonic() + 60
        while not any(_per_engine(base_url, RUNNING).values()):
            assert time.monotonic() < deadline, "the long request did not start"
            time.sleep(0.2)
        running = _per_engine(base_url, RUNNING)
        held = next(engine for engine, count in running.items() if count > 0)
        for index in range(BESIDE):
            _request(
                base_url, f"Say the letter {'abcd'[index]}.", 4
            ).raise_for_status()
        streaming.join(timeout=900)
        assert long_reply and long_reply[0].status_code == 200
        beside = _settled(base_url, before, BESIDE + 1)
        assert beside == {held: 1, 1 - held: BESIDE}, (held, beside)
