"""A prefill group transfers each request to a decode group.

Qwen3 is served by a deployment of two worker groups: a ``prefill`` group
that computes each prompt and samples its first token, and a ``decode``
group that continues the request from that token. One scheduler drives both
as separate groups of its session with the host's buffer pool: once the
prefill group's call that sampled the first token is output, the scheduler
submits a ``Transfer`` of the request's owner to the decode group, stamped by
the prefill group's executor; a task of the Transfer Server copies the
request's blocks that are not valid in the decode group's memory, device to
device, the owner gets a slot there, and the request decodes on the decode
group (``specs/buffer-pool.md`` §11, ``specs/scheduler-design.md`` §4.1). Each
case compares the greedy tokens of fixed prompts, one of several blocks with
a partial last one, with an undivided deployment's on one GPU: a prefill
group computes no decode, so the tokens after the first are the decode
group's, computed over the blocks it was transferred. Each case also reads
the scheduler's transfer counters on ``/metrics``
(``specs/scheduler-design.md`` §7.5): every prompt was transferred, none
failed, and the decode group received every block the request held when its
prompt was computed, those of its prompt and the low watermark's.

The one-host case places both groups on this host's two GPUs. The two-host
case places the decode group on a second host, named by the suite's
two-host variables (docs/testing.md, Two hosts): ``UNISERVE_H3_FOLLOWER``,
an ssh destination with this checkout at the same path, and
``UNISERVE_H3_LEADER_ADDRESS``, this host's address as the second reaches
it. Its Transfer Server reads the prefill rank's planes through NIXL, which
both hosts have installed as docs/nixl.md states, and serves on the second
host's GPU ``UNISERVE_FOLLOWER_DEVICE`` (``0`` unless stated). Without
these variables the two-host case is skipped.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.build_server import remote_build_server
from tests.python.e2e.http_helpers import (
    ATTENTION_BACKEND,
    QWEN3_CHECKPOINT,
    REPORT_SETTLE_S,
    engine_selection,
    granted_devices,
    host_asset,
    metric,
    require_uniserve_binary,
    runtime_directory,
    scaled,
    second_host,
    server_process,
    wait_until,
    written_deployment,
)

# The module's cases share two GPUs: the undivided reference serves on the
# first, and the one-host prefill/decode deployment on both.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    pytest.mark.gpus(2),
]

FOLLOWER = os.environ.get("UNISERVE_H3_FOLLOWER")
LEADER_ADDRESS = os.environ.get("UNISERVE_H3_LEADER_ADDRESS")
FOLLOWER_DEVICE = os.environ.get("UNISERVE_FOLLOWER_DEVICE", "0")

SERVED_MODEL = "qwen3-prefill-decode"
MAX_TOKENS = 48
PROMPTS = (
    "Name the planets of the solar system in order from the sun.",
    "Write a haiku about a lighthouse in winter.",
    "Explain in two sentences why the sky is blue.",
    # A prompt of several blocks, whose last block is partial.
    " ".join(
        f"Record {i}: item {i * 7919 % 1000} is {i * 104729 % 997}."
        for i in range(60)
    )
    + " Which record has the largest value?",
)

# The tokens of a KV block.
BLOCK_TOKENS = 64
# The blocks of each cache group a request holds beyond its next call: the
# low watermark (``specs/scheduler-design.md`` §4.4).
LOW_WATERMARK = 1

# The scheduler's transfer counters (``specs/scheduler-design.md`` §7.5): the
# transfers answered ``Transferred``, those answered ``Failed``, and the
# blocks the decode groups received, one per entry of each transferred
# request's tables.
TRANSFER_COUNTERS = (
    "uniserve:request_transfers",
    "uniserve:request_transfers_failed",
    "uniserve:request_transfer_blocks",
)

# A request's answer: its greedy tokens and its prompt's tokens.
Answer = tuple[list[str], int]

# Every deployment of the module serves with these options.
OPTIONS = (
    "dtype=bfloat16",
    f"attention_backend={ATTENTION_BACKEND}",
    "kv_cache_config.max_tokens=32768",
    f"kv_cache_config.block_size={BLOCK_TOKENS}",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[1024]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
)


def _checkpoint() -> str:
    return str(host_asset("UNISERVE_QWEN3_MODEL", QWEN3_CHECKPOINT))


def _command(*options: str) -> list[str]:
    command = [
        str(require_uniserve_binary()),
        "serve",
        _checkpoint(),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        *engine_selection(),
    ]
    for option in OPTIONS:
        command += ["--set", option]
    return [*command, *options]


def _split(directory: Path, decode_host: str | None) -> Path:
    """A deployment file of a prefill group on this host's first GPU and a
    decode group on the next one, or on `decode_host`'s first.
    """  # noqa: D205
    decode = "cuda:1" if decode_host is None else f"{decode_host}/cuda:0"
    return written_deployment(
        directory,
        [
            {
                "id": "prefill",
                "role": "prefill",
                "ranks": ["cuda:0"],
                "queue_depth": 2,
                "blocks": {"model": {}},
            },
            {
                "id": "decode",
                "role": "decode",
                "ranks": [decode],
                "queue_depth": 2,
                "blocks": {"model": {}},
            },
        ],
    )


def _greedy(base_url: str, prompt: str) -> Answer:
    """The tokens `prompt` decodes to greedily, and its prompt's tokens."""
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "max_completion_tokens": MAX_TOKENS,
            "logprobs": True,
            "return_tokens_as_token_ids": True,
        },
        timeout=900,
    )
    response.raise_for_status()
    reply = response.json()
    tokens = [
        entry["token"] for entry in reply["choices"][0]["logprobs"]["content"]
    ]
    return tokens, reply["usage"]["prompt_tokens"]


def _transfer_counters(base_url: str) -> tuple[float, ...]:
    """The scheduler's transfer counters, in the order of
    ``TRANSFER_COUNTERS``.
    """  # noqa: D205
    return tuple(metric(base_url, name) for name in TRANSFER_COUNTERS)


def _idle(base_url: str) -> bool:
    """Whether the schedulers reported, and reported no request running or
    waiting.
    """  # noqa: D205
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    busy = metric(base_url, "uniserve:num_requests_running") + metric(
        base_url, "uniserve:num_requests_waiting"
    )
    return "uniserve:num_requests_running{" in text and busy == 0


def _transfers(
    base_url: str, ask: Callable[[], list[Answer]]
) -> tuple[list[Answer], tuple[int, ...]]:
    """What `ask` returns, and how much each transfer counter grew over it.

    The serving path's warm-up requests may end after the server answers
    its health and are transferred too, so the counters are first read once
    a report taken after them states no request running or waiting. The
    scheduler reports once a second: the counters are read again once they
    hold a transfer per request `ask` answered and another report followed.
    """
    time.sleep(scaled(REPORT_SETTLE_S))
    wait_until(lambda: _idle(base_url), 120, "the warm-up requests did not end")
    before = _transfer_counters(base_url)
    answers = ask()
    wait_until(
        lambda: (
            sum(_transfer_counters(base_url)[:2])
            >= sum(before[:2]) + len(answers)
        ),
        30,
        "the requests' transfers were not reported",
    )
    time.sleep(scaled(REPORT_SETTLE_S))
    after = _transfer_counters(base_url)
    return answers, tuple(
        int(a - b) for a, b in zip(after, before, strict=True)
    )


def _expected(answers: list[Answer]) -> tuple[int, ...]:
    """The transfer counters' growth over `answers`: each request was
    transferred, and its decode group received the blocks the request held
    when its prompt was computed: those of its prompt and the low
    watermark's.
    """  # noqa: D205
    blocks = sum(
        math.ceil(prompt_tokens / BLOCK_TOKENS) + LOW_WATERMARK
        for _, prompt_tokens in answers
    )
    return len(answers), 0, blocks


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> list[list[str]]:
    """Each prompt's greedy tokens on an undivided deployment of one GPU."""
    directory = tmp_path_factory.mktemp("undivided")
    env = {"CUDA_VISIBLE_DEVICES": granted_devices(1)}
    with server_process(
        _command(), directory / "server.log", 1800, env=env
    ) as server:
        return [_greedy(server.base_url, prompt)[0] for prompt in PROMPTS]


def test_a_prefill_group_transfers_requests_to_a_decode_group_on_one_host(
    tmp_path: Path, reference: list[list[str]]
):
    log = tmp_path / "server.log"
    command = _command("--worker-groups", str(_split(tmp_path, None)))
    with (
        runtime_directory(tmp_path / "storage-server.log") as runtime,
        server_process(
            command,
            log,
            1800,
            env={
                "CUDA_VISIBLE_DEVICES": granted_devices(2),
                "XDG_RUNTIME_DIR": str(runtime),
            },
        ) as server,
    ):
        base_url = server.base_url
        answers, transferred = _transfers(
            base_url, lambda: [_greedy(base_url, prompt) for prompt in PROMPTS]
        )

    assert [tokens for tokens, _ in answers] == reference
    assert transferred == _expected(answers), answers


def test_a_prefill_group_transfers_requests_to_a_decode_group_on_another_host(
    tmp_path: Path, reference: list[list[str]]
):
    if FOLLOWER is None or LEADER_ADDRESS is None:
        pytest.skip(
            "a decode group on another host needs UNISERVE_H3_FOLLOWER and "
            "UNISERVE_H3_LEADER_ADDRESS"
        )
    log = tmp_path / "server.log"
    # The leader binds its join endpoint on a port it picks, and both hosts'
    # Storage Servers, which the daemons start with the deployment's
    # settings, load the installed NIXL.
    command = _command(
        "--worker-groups",
        str(_split(tmp_path, "rank-1")),
        "--host-identity",
        "rank-0",
        "--nnodes",
        "2",
        "--dist-init-addr",
        f"{LEADER_ADDRESS}:0",
    )
    with (
        runtime_directory(tmp_path / "storage-server.log") as runtime,
        remote_build_server(FOLLOWER) as follower_modules,
        server_process(
            command,
            log,
            1800,
            env={
                "CUDA_VISIBLE_DEVICES": granted_devices(1),
                "XDG_RUNTIME_DIR": str(runtime),
            },
            serving=False,
        ) as server,
        second_host(
            FOLLOWER,
            tmp_path / "follower.log",
            server,
            devices=FOLLOWER_DEVICE,
            arguments=("--set", f"build_config.socket={follower_modules}"),
        ),
    ):
        base_url = server.base_url
        answers, transferred = _transfers(
            base_url, lambda: [_greedy(base_url, prompt) for prompt in PROMPTS]
        )

    assert [tokens for tokens, _ in answers] == reference
    assert transferred == _expected(answers), answers
