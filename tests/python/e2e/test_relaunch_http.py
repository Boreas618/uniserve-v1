"""A streaming request survives the loss of its worker or of its engine.

Qwen3 streams a long greedy count to a client. While it streams, a process
that serves it is killed: one test kills the worker rank, the other the
engine process. The engine relaunches a lost worker group, and the leader's
daemon starts a new engine process for a lost one; either way the request
leaves the lost workers as migratable after every token it generated, and
the daemon submits it again with those tokens once an engine serves, which
continues the count where it stopped. The client reads one uninterrupted
stream.

The stream decodes the tokens the engine decodes for the same request alone
afterwards, and its usage charges the same prompt and completion. The
relaunched workers find their weights retained at the Storage Server. The
count is a continuation that no near tie decides, so rounding differences
between the lost workers' decode of the generated tokens and the new
workers' prefill of them cannot change it. The deployment captures few
graphs (decode batches up to 8, prefill buckets of 512 tokens and the
batch bound), so that the workers start again while the count streams.

Two schedulers that share their experts (``--data-parallel-size 2`` with
expert parallelism), or attention and expert workers exchanging through
DeepEP, form one expert world, which keeps a standby world
(``worker_config.standby``): once both serve, each worker group is launched
a second time beside the first, at a rendezvous of its own, and parks with
its graphs captured, its device storage kept resident or suspended
(``worker_config.standby_storage``: suspended, it also releases its
all-to-all workspaces and NCCL communicators, which its ranks restore
together at the promotion, and frees DeepEP's windows, which cannot keep
their addresses: the promotion rebuilds them and captures the graphs that
run expert steps again). Killing a serving rank promotes that epoch in
place of a relaunch, and the request continues on it.

Which main process serves before and after a kill is what ``GET
/admin/status`` reports: the serving process's launch generation
(``generation``) and the parked standby's (``standby``), which a promotion
makes the serving one.

Each test runs a Storage Server of its own: ``XDG_RUNTIME_DIR`` points the
daemon at a directory of the test's, and the server the daemon starts there
is stopped at the end.
"""

from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    ATTENTION_BACKEND,
    admin_status,
    dedicated_experts,
    find_free_port,
    granted_devices,
    log_tail,
    qwen3_checkpoint,
    require_uniserve_binary,
    runtime_directory,
    server_process,
    wait_until,
    worker_executable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

SERVED_MODEL = "qwen3-relaunch"
MAX_TOKENS = 16384
# Tokens the client reads before a process is killed.
EARLY_TOKENS = 64
PROMPT = (
    "Count from 1 to 5000, separated by commas and spaces. "
    "Write only the numbers."
)


def _body(stream: bool) -> dict:
    body = {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "max_completion_tokens": MAX_TOKENS,
        "logprobs": True,
        "return_tokens_as_token_ids": True,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _chunk_tokens(line: str) -> tuple[list[str], dict | None, bool]:
    """The tokens and usage one SSE line carries, and whether it ends."""
    if not line.startswith("data: "):
        return [], None, False
    if line == "data: [DONE]":
        return [], None, True
    chunk = json.loads(line[6:])
    tokens = [
        entry["token"]
        for choice in chunk.get("choices", ())
        for entry in (choice.get("logprobs") or {}).get("content") or ()
    ]
    return tokens, chunk.get("usage"), False


def _processes(
    argument: str, runtime: Path, generation: int | None = None
) -> list[int]:
    """The pids of the test's `uniserve` processes running `argument`.

    A process is the test's when its environment names `runtime`; with
    `generation`, only a worker of that launch of its group counts.
    """
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().decode().split("\0")
            environment = (entry / "environ").read_bytes().decode()
        except OSError:
            continue
        launch = (
            generation is None
            or f"--generation\0{generation}\0" in "\0".join(command)
        )
        if (
            len(command) > 1
            and command[1] == argument
            and str(runtime) in environment
            and launch
        ):
            pids.append(int(entry.name))
    return pids


@dataclass
class Killed:
    """What a stream through a kill read, and what the deployment reported.

    `streamed` and `usage` are the stream's tokens and usage, `reference`
    and `reference_usage` those of the count asked again afterwards.
    `before` and `after` are the deployment of ``GET /admin/status`` before
    the kill and once the count was asked again; `alive` says whether a
    worker of the group's first launch still ran then, and `workers` how
    many of the deployment's workers ran.
    """

    streamed: list[str]
    usage: dict
    reference: list[str]
    reference_usage: dict
    before: dict
    after: dict
    alive: bool
    workers: int


def _stream_through_a_kill(
    tmp_path: Path,
    victim: str,
    *,
    options: tuple[str, ...] = (),
    devices: str | None = None,
    standby: bool = False,
) -> Killed:
    """Stream the count, killing a `victim` process after the first tokens.

    `victim` is `worker` (a rank of a group's first launch) or `engine`.
    The server runs on `devices` with the extra `options`; with `standby`,
    the stream starts once the deployment reports its standby prepared.
    """
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        qwen3_checkpoint(),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
        "--set",
        "dtype=bfloat16",
        "--set",
        f"attention_backend={ATTENTION_BACKEND}",
        "--set",
        "kv_cache_config.max_tokens=32768",
        "--set",
        "chat_config.reasoning_parser=none",
        "--set",
        "cuda_graph_config.prefill_token_sizes=[512]",
        "--set",
        "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
        *options,
    ]
    log = tmp_path / "server.log"
    streamed: list[str] = []
    usage = None
    with runtime_directory(tmp_path / "storage-server.log") as runtime:
        env = {
            "CUDA_VISIBLE_DEVICES": devices or granted_devices(1),
            "XDG_RUNTIME_DIR": str(runtime),
        }
        with server_process(command, base_url, log, timeout_s=900, env=env):
            if standby:
                wait_until(
                    lambda: (
                        admin_status(base_url)["deployment"]["standby"] or {}
                    ).get("prepared"),
                    900,
                    "the deployment reported no prepared standby",
                    context=lambda: log_tail(log),
                )
            before = admin_status(base_url)["deployment"]
            with httpx.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=_body(stream=True),
                timeout=1800,
            ) as response:
                response.raise_for_status()
                killed = False
                for line in response.iter_lines():
                    tokens, chunk_usage, done = _chunk_tokens(line)
                    if done:
                        break
                    streamed.extend(tokens)
                    usage = chunk_usage or usage
                    if not killed and len(streamed) >= EARLY_TOKENS:
                        pids = _processes(
                            victim,
                            runtime,
                            generation=1 if victim == "worker" else None,
                        )
                        assert pids, f"no {victim} process serves"
                        os.kill(pids[0], signal.SIGKILL)
                        killed = True
            reply = httpx.post(
                f"{base_url}/v1/chat/completions",
                json=_body(stream=False),
                timeout=900,
            )
            reply.raise_for_status()
            answer = reply.json()
            after = admin_status(base_url)["deployment"]
            alive = bool(_processes("worker", runtime, generation=1))
            workers = len(_processes("worker", runtime))
    return Killed(
        streamed=streamed,
        usage=usage,
        reference=[
            entry["token"]
            for entry in answer["choices"][0]["logprobs"]["content"]
        ],
        reference_usage=answer["usage"],
        before=before,
        after=after,
        alive=alive,
        workers=workers,
    )


def _continued(killed: Killed) -> None:
    """The stream read the count the deployment answers alone, with the
    same usage: it continued where the kill stopped it, and the deployment
    serves again.
    """  # noqa: D205
    assert killed.streamed == killed.reference
    assert killed.usage == killed.reference_usage
    assert killed.after["engine"] == "ready", killed.after


def test_a_request_continues_after_its_worker_rank_is_lost(tmp_path: Path):
    killed = _stream_through_a_kill(tmp_path, "worker")
    _continued(killed)
    # The engine process served throughout, and its group's ranks were
    # launched again.
    assert killed.after["generation"] == killed.before["generation"]
    assert not killed.alive, "a worker of the lost launch still runs"
    assert killed.workers > 0, "no worker serves the relaunched group"


def test_a_request_continues_after_its_engine_process_is_lost(tmp_path: Path):
    killed = _stream_through_a_kill(tmp_path, "engine")
    _continued(killed)
    # The daemon started another engine process, which serves.
    assert killed.after["generation"] not in (
        None,
        killed.before["generation"],
    ), (killed.before, killed.after)


def _expert_world(topology: str, tmp_path: Path) -> tuple[str, ...]:
    """The options of an expert world of two groups on two GPUs.

    ``replicas``: two schedulers sharing their experts over the all-to-all.
    ``dedicated``: attention and expert workers exchanging through DeepEP,
    whose windows a suspended standby frees and its promotion rebuilds.
    """
    if topology == "replicas":
        return ("--data-parallel-size", "2")
    return (
        "--worker-groups",
        str(dedicated_experts(tmp_path)),
        "--set",
        "moe_config.exchange=deepep",
    )


@pytest.mark.gpus(2)
@pytest.mark.parametrize("topology", ["replicas", "dedicated"])
@pytest.mark.parametrize("standby_storage", ["resident", "suspended"])
def test_an_expert_world_promotes_its_standby_world_when_a_rank_is_lost(
    tmp_path: Path, standby_storage: str, topology: str
):
    killed = _stream_through_a_kill(
        tmp_path,
        "worker",
        options=(
            *_expert_world(topology, tmp_path),
            "--set",
            "moe_config.expert_parallel=true",
            "--set",
            "worker_config.standby=true",
            "--set",
            f"worker_config.standby_storage={standby_storage}",
        ),
        devices=granted_devices(2),
        standby=True,
    )
    _continued(killed)
    # The parked standby serves now, rather than a world launched anew.
    standby = killed.before["standby"]
    assert killed.after["generation"] == standby["generations"][0], (
        killed.before,
        killed.after,
    )
