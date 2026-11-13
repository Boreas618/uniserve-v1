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
group's, computed over the blocks it was transferred.

The one-host case places both groups on this host's two GPUs. The two-host
case places the decode group on a second host, named by the suite's
two-host variables (docs/testing.md, Two hosts): ``UNISERVE_H3_FOLLOWER``,
an ssh destination with this checkout at the same path, and
``UNISERVE_H3_LEADER_ADDRESS``, this host's address as the second reaches
it. Its Transfer Server reads the prefill rank's planes through NIXL, which
both hosts' Storage Servers load from ``UNISERVE_NIXL_LIBRARY``'s directory,
and serves on the second host's GPU ``UNISERVE_FOLLOWER_DEVICE`` (``0``
unless stated). Without these variables the two-host case is skipped.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.build_server import remote_build_server
from tests.python.e2e.http_helpers import (
    ATTENTION_BACKEND,
    QWEN3_CHECKPOINT,
    engine_selection,
    find_free_port,
    granted_devices,
    host_asset,
    require_uniserve_binary,
    second_host,
    server_process,
    stop_storage_server,
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
NIXL_LIBRARY = os.environ.get("UNISERVE_NIXL_LIBRARY")
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

# Every deployment of the module serves with these options.
OPTIONS = (
    "dtype=bfloat16",
    f"attention_backend={ATTENTION_BACKEND}",
    "kv_cache_config.max_tokens=32768",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[1024]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
)


def _checkpoint() -> str:
    return str(host_asset("UNISERVE_QWEN3_MODEL", QWEN3_CHECKPOINT))


def _command(port: int, *options: str) -> list[str]:
    command = [
        str(require_uniserve_binary()),
        "serve",
        _checkpoint(),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
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


def _greedy(base_url: str, prompt: str) -> list[str]:
    """The tokens `prompt` decodes to greedily."""
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
    return [
        entry["token"]
        for entry in response.json()["choices"][0]["logprobs"]["content"]
    ]


@contextmanager
def _runtime(tmp_path: Path) -> Iterator[Path]:
    """A runtime directory of its own, whose Storage Server is stopped and
    whose log is kept beside the test's at the end.
    """  # noqa: D205
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
    try:
        yield runtime
    finally:
        stop_storage_server(runtime)
        storage_log = runtime / "uniserve" / "storage-server.log"
        if storage_log.is_file():
            shutil.copy(storage_log, tmp_path / "storage-server.log")
        shutil.rmtree(runtime, ignore_errors=True)


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> list[list[str]]:
    """Each prompt's greedy tokens on an undivided deployment of one GPU."""
    directory = tmp_path_factory.mktemp("undivided")
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = {"CUDA_VISIBLE_DEVICES": granted_devices(1)}
    with server_process(
        _command(port), base_url, directory / "server.log", 1800, env=env
    ):
        return [_greedy(base_url, prompt) for prompt in PROMPTS]


def test_a_prefill_group_transfers_requests_to_a_decode_group_on_one_host(
    tmp_path: Path, reference: list[list[str]]
):
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log = tmp_path / "server.log"
    command = _command(port, "--worker-groups", str(_split(tmp_path, None)))
    with (
        _runtime(tmp_path) as runtime,
        server_process(
            command,
            base_url,
            log,
            1800,
            env={
                "CUDA_VISIBLE_DEVICES": granted_devices(2),
                "XDG_RUNTIME_DIR": str(runtime),
            },
        ),
    ):
        tokens = [_greedy(base_url, prompt) for prompt in PROMPTS]

    assert tokens == reference


def test_a_prefill_group_transfers_requests_to_a_decode_group_on_another_host(
    tmp_path: Path, reference: list[list[str]]
):
    if FOLLOWER is None or LEADER_ADDRESS is None or NIXL_LIBRARY is None:
        pytest.skip(
            "a decode group on another host needs UNISERVE_H3_FOLLOWER, "
            "UNISERVE_H3_LEADER_ADDRESS and UNISERVE_NIXL_LIBRARY"
        )
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    address = f"{LEADER_ADDRESS}:{find_free_port()}"
    log = tmp_path / "server.log"
    command = _command(
        port,
        "--worker-groups",
        str(_split(tmp_path, "rank-1")),
        "--host-identity",
        "rank-0",
        "--nnodes",
        "2",
        "--dist-init-addr",
        address,
    )
    # Both hosts' Storage Servers, which the daemons start, load NIXL from
    # the loader's search path.
    library_path = ":".join(
        [str(Path(NIXL_LIBRARY).parent), os.environ.get("LD_LIBRARY_PATH", "")]
    )
    with (
        _runtime(tmp_path) as runtime,
        remote_build_server(FOLLOWER) as follower_modules,
        second_host(
            FOLLOWER,
            tmp_path / "follower.log",
            address,
            devices=FOLLOWER_DEVICE,
            env={"LD_LIBRARY_PATH": library_path},
            arguments=("--set", f"build_config.socket={follower_modules}"),
        ),
        server_process(
            command,
            base_url,
            log,
            1800,
            env={
                "CUDA_VISIBLE_DEVICES": granted_devices(1),
                "XDG_RUNTIME_DIR": str(runtime),
                "LD_LIBRARY_PATH": library_path,
            },
        ),
    ):
        tokens = [_greedy(base_url, prompt) for prompt in PROMPTS]

    assert tokens == reference
