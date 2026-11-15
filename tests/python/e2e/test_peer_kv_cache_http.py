"""A prompt's KV blocks cross hosts through the Storage Servers' peers.

**Peer reuse across hosts.** Qwen3 is served by two single-node deployments
of the same checkpoint, served name and block size: deployment A on this
host and deployment B on a second host, whose ``kv_cache_config.peers``
names A's Storage Server. A answers a long prompt; its write policy stores
every full block it commits to A's host tier, which A's own admission then
loads back once its device copies are evicted (``POST /admin/flush_cache``):
that answer is computed over the stored blocks. B then answers the same
prompt: its admission finds the keys in neither its device nor its host
tier, and since the peers could deepen the hit by at least
``prefetch_min_tokens``, it submits a ``Prefetch`` from the peers
(``specs/scheduler-design.md`` §4.4). A's Storage Server holds the host
copies by key (``PeerHold``) while B's Transfer Server reads them through
NIXL into B's host tier, and B's admission loads them. B's
``prefix_cache_peer_hits`` grows by the prompt's full blocks, its
``storage-server list`` shows the measured link from A's host, and its
greedy tokens equal A's answer over the stored blocks, computed over the
same keys and values. A prompt A never saw brings nothing from A; once A's
Storage Server stops, B computes a prompt A stored, with the same tokens as
A computed it, and no request fails (a tier's failure never fails a
request, ``specs/buffer-pool.md`` §12.4).

**Converted reuse across hosts.** Deployment B keeps its device KV blocks in
FP8 (``kv_cache_config.dtype=float8_e4m3fn``) while A keeps BF16: a key
names the tokens and the weights that compute them, not the cache's
precision, so B's prefetch finds A's BF16 host copies, which A's Storage
Server states in the encoding A's class wrote them in; B's buffer pool reads
them into its converted-load source region (``Class::source``), and B's load
converts each into its FP8 device block (``specs/buffer-pool.md`` §13, §20),
as B's own writes encode it. B's
``prefix_cache_peer_hits`` grows by the prompt's full blocks, and its greedy
answer over the converted blocks agrees with its answer over the same
prompt's blocks it computed and wrote itself (another ``cache_salt``, asked
twice: computed cold, then pinned on its device) by the rule of vLLM's
``check_logprobs_close``: token for token until their first difference,
where each answer's token is among the other's ``TOP_LOGPROBS`` most likely
tokens. Both answers compute the prompt's last tokens over FP8 blocks; FP8
blocks written whole from a block's amax and blocks written token by token
round differently, so the two are not required to be identical.

The second host is named by the suite's two-host variables (docs/testing.md,
Two hosts): ``UNISERVE_H3_FOLLOWER``, an ssh destination with this checkout
at the same path, and ``UNISERVE_H3_LEADER_ADDRESS``, this host's address as
the second reaches it. Both hosts' Storage Servers are the test's own,
started with a TCP endpoint and NIXL (``UNISERVE_NIXL_LIBRARY``, the
``libnixl_capi.so`` of docs/nixl.md); without these variables the two-host
case is skipped. The peer endpoints are unauthenticated: the hosts share an
isolated network.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.build_server import remote_build_server
from tests.python.e2e.http_helpers import (
    ATTENTION_BACKEND,
    admin,
    engine_selection,
    find_free_port,
    granted_devices,
    qwen3_checkpoint,
    require_uniserve_binary,
    server_process,
    stop_remote_storage_server,
    stop_storage_server,
    tokens_found,
    wait_until,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

FOLLOWER = os.environ.get("UNISERVE_H3_FOLLOWER")
LEADER_ADDRESS = os.environ.get("UNISERVE_H3_LEADER_ADDRESS")
NIXL_LIBRARY = os.environ.get("UNISERVE_NIXL_LIBRARY")
# The second host's GPU the cases serve on.
FOLLOWER_DEVICE = os.environ.get("UNISERVE_FOLLOWER_DEVICE", "0")

SERVED_MODEL = "qwen3"
BLOCK_SIZE = 64
MAX_TOKENS = 32
# Blocks a prompt must fill for its peer hit to count.
PEER_BLOCKS = 16
# The candidates two greedy answers are compared at, as vLLM's model
# correctness tests compare logprobs.
TOP_LOGPROBS = 5


def _records(label: str, count: int = 260) -> str:
    """A long prompt of records under `label`, then a question about them."""
    return (
        f"{label}. "
        + " ".join(
            f"Record {i}: the value of item {i * 7919 % 1000} is "
            f"{i * 104729 % 997}."
            for i in range(count)
        )
        + " Summarize the first three records in one sentence."
    )


def _answer(base_url: str, prompt: str, **fields) -> dict:
    """The greedy answer to `prompt`, with `fields` added to the request."""
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "max_completion_tokens": MAX_TOKENS,
            "logprobs": True,
            "return_tokens_as_token_ids": True,
            **fields,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _ask(base_url: str, prompt: str) -> tuple[int, list[str]]:
    """Decode `prompt` greedily: its prompt tokens and the decoded tokens."""
    reply = _answer(base_url, prompt)
    tokens = [
        entry["token"] for entry in reply["choices"][0]["logprobs"]["content"]
    ]
    return reply["usage"]["prompt_tokens"], tokens


def _found(base_url: str, prompt: str) -> tuple[list[str], dict[str, float]]:
    """Decode `prompt` greedily: the decoded tokens, and the tokens its
    admission found cached per tier and source (:func:`tokens_found`).
    """  # noqa: D205
    (_, tokens), found = tokens_found(base_url, lambda: _ask(base_url, prompt))
    return tokens, found


def _reusable(prompt_tokens: int) -> int:
    """The tokens an admission may reuse: the prompt less its last token, in
    whole blocks.
    """  # noqa: D205
    return (prompt_tokens - 1) // BLOCK_SIZE * BLOCK_SIZE


def _stored(base_url: str, prompt: str, reusable: int) -> list[str]:
    """Waits until the host tier of the deployment at `base_url` holds every
    reusable block of `prompt`, which the deployment answered before: its
    device copies are evicted, and the prompt asked again loads every one of
    them from the host. Returns that answer, computed over the stored blocks.

    The write policy stores a block once it is committed, by a copy its
    executor runs with the next batch, so the copies may still run when the
    request has answered.
    """  # noqa: D205
    answers: list[list[str]] = []

    def loaded() -> bool:
        flushed = admin(base_url, "flush_cache")
        assert flushed.status_code == 200, flushed.text
        tokens, found = _found(base_url, prompt)
        answers.append(tokens)
        return found["host_hits"] == reusable

    wait_until(loaded, 120, "the host tier did not hold the prompt's blocks")
    return answers[-1]


def _candidates(base_url: str, prompt: str, **fields) -> list[tuple[str, set]]:
    """Decode `prompt` greedily: each decoded token with its position's
    `TOP_LOGPROBS` most likely tokens.
    """  # noqa: D205
    reply = _answer(base_url, prompt, top_logprobs=TOP_LOGPROBS, **fields)
    return [
        (entry["token"], {top["token"] for top in entry["top_logprobs"]})
        for entry in reply["choices"][0]["logprobs"]["content"]
    ]


def _logprobs_close(left: list, right: list) -> bool:
    """Whether two answers agree by vLLM's `check_logprobs_close`: token for
    token until their first difference, where each one's token is among the
    other's most likely tokens.
    """  # noqa: D205
    for (token, tops), (other, other_tops) in zip(left, right):
        if token != other:
            return token in other_tops and other in tops
    return True


def _serve_command(
    checkpoint: str, port: int, host: str, extra: tuple[str, ...] = ()
) -> list[str]:
    return [
        str(require_uniserve_binary()),
        "serve",
        checkpoint,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        host,
        "--port",
        str(port),
        *engine_selection(),
        "--set",
        "dtype=bfloat16",
        "--set",
        f"attention_backend={ATTENTION_BACKEND}",
        "--set",
        "kv_cache_config.max_tokens=32768",
        "--set",
        f"kv_cache_config.block_size={BLOCK_SIZE}",
        "--set",
        "chat_config.reasoning_parser=none",
        *extra,
    ]


def _storage_command(socket: str, tcp: str) -> list[str]:
    """The Storage Server of a test host: on `socket`, reached by other hosts
    at `tcp`, its Transfer Server reading them through NIXL.
    """  # noqa: D205
    assert NIXL_LIBRARY is not None
    return [
        str(require_uniserve_binary()),
        "storage-server",
        "serve",
        "--daemon",
        "--socket",
        socket,
        "--tcp",
        tcp,
        "--nixl-library",
        NIXL_LIBRARY,
    ]


def _environment() -> dict[str, str]:
    """The variables a test host's processes take from this test's."""
    return {
        name: os.environ.get(name, "")
        for name in (
            "PATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "MAX_JOBS",
            "FLASHINFER_WORKSPACE_BASE",
            "UCX_NET_DEVICES",
        )
    }


def _on_follower(command: str, timeout: float = 120) -> str:
    assert FOLLOWER is not None
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", FOLLOWER, command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return done.stdout


@contextmanager
def _follower_deployment(
    log: Path, port: int, devices: str, extra: list[str]
) -> Iterator[tuple[str, str]]:
    """Deployment B on the second host, its own Storage Server started first
    with a TCP endpoint and NIXL, and its own Build Server.

    Yields B's HTTP base URL and its runtime directory there; on exit it
    stops the daemon and the Storage Server, keeping their logs beside `log`.
    """  # noqa: D205
    assert FOLLOWER is not None
    address = _on_follower("hostname -I").split()[0]
    runtime = f"/tmp/uniserve-peer-{os.getpid()}-{find_free_port()}"
    socket = f"{runtime}/uniserve/storage-server.sock"
    variables = _environment()
    variables["XDG_RUNTIME_DIR"] = runtime
    variables["CUDA_VISIBLE_DEVICES"] = devices
    environment = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in variables.items()
    )
    _on_follower(
        f"mkdir -p -m 700 {runtime}/uniserve && env {environment} "
        + shlex.join(_storage_command(socket, f"{address}:0"))
    )
    with remote_build_server(FOLLOWER) as modules:
        command = _serve_command(qwen3_checkpoint(), port, "0.0.0.0") + [
            "--set",
            f"build_config.socket={modules}",
            *extra,
        ]
        remote = (
            f"cd {shlex.quote(str(Path.cwd()))} && echo follower pid $$ && "
            f"exec env {environment} {shlex.join(command)}"
        )
        process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", FOLLOWER, remote],
            stdout=log.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        base_url = f"http://{address}:{port}"
        try:
            deadline = time.monotonic() + 1800
            while True:
                try:
                    if httpx.get(f"{base_url}/health", timeout=5).is_success:
                        break
                except httpx.HTTPError:
                    pass
                assert process.poll() is None, (
                    f"deployment B ended: {log.read_text()[-4000:]}"
                )
                assert time.monotonic() < deadline, "deployment B did not start"
                time.sleep(2)
            yield base_url, runtime
        finally:
            pid = re.search(r"follower pid (\d+)", log.read_text())
            if pid is not None and process.poll() is None:
                _on_follower(f"kill -TERM -- {pid.group(1)}; true")
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
            stop_remote_storage_server(
                FOLLOWER, runtime, log.with_suffix(".storage.log")
            )


def _leader_storage() -> tuple[Path, dict[str, str], str]:
    """Deployment A's Storage Server, started on this host with a TCP
    endpoint and NIXL before A starts: its runtime directory, the
    environment A serves under, and the endpoint B's peers name.
    """  # noqa: D205
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
    socket = runtime / "uniserve" / "storage-server.sock"
    socket.parent.mkdir(mode=0o700)
    env = {
        "CUDA_VISIBLE_DEVICES": granted_devices(1),
        "XDG_RUNTIME_DIR": str(runtime),
    }
    peer = f"{LEADER_ADDRESS}:{find_free_port()}"
    subprocess.run(
        _storage_command(str(socket), peer),
        env={**os.environ, **env},
        check=True,
        timeout=120,
    )
    return runtime, env, peer


def _release_leader_storage(runtime: Path, tmp_path: Path) -> None:
    """Stops A's Storage Server and keeps its log beside the test's."""
    stop_storage_server(runtime)
    storage_log = runtime / "uniserve" / "storage-server.log"
    if storage_log.is_file():
        shutil.copy(storage_log, tmp_path / "storage-server.log")
    shutil.rmtree(runtime, ignore_errors=True)


def _skip_without_two_hosts() -> None:
    if FOLLOWER is None or LEADER_ADDRESS is None or NIXL_LIBRARY is None:
        pytest.skip(
            "peer reads across hosts need UNISERVE_H3_FOLLOWER, "
            "UNISERVE_H3_LEADER_ADDRESS and UNISERVE_NIXL_LIBRARY"
        )


def test_a_host_loads_a_prompts_blocks_from_a_peer_hosts_host_tier(
    tmp_path: Path,
):
    _skip_without_two_hosts()
    shared = _records("Shared")
    dropped = _records("Dropped")
    runtime, env, peer = _leader_storage()
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    listing = ""
    try:
        with server_process(
            _serve_command(qwen3_checkpoint(), port, "127.0.0.1"),
            base_url,
            tmp_path / "a.log",
            timeout_s=1800,
            env=env,
        ):
            # A computes both prompts and stores their full blocks to its
            # host tier. Its answer to the shared prompt over the stored
            # blocks computes only the last tokens over them, as B's does over
            # the blocks it reads: the two compute the same tokens over the
            # same cached keys and values (a prompt computed whole may round
            # otherwise, as `test_host_kv_cache_http.py` states).
            prompt_tokens, _ = _ask(base_url, shared)
            reusable = _reusable(prompt_tokens)
            a_shared = _stored(base_url, shared, reusable)
            _, a_dropped = _ask(base_url, dropped)
            _stored(base_url, dropped, reusable)

            with _follower_deployment(
                tmp_path / "b.log",
                find_free_port(),
                FOLLOWER_DEVICE,
                ["--set", f'kv_cache_config.peers=["{peer}"]'],
            ) as (b_url, b_runtime):
                # B reads the shared prompt's blocks from A.
                b_shared, shared_found = _found(b_url, shared)
                libraries = shlex.quote(os.environ.get("LD_LIBRARY_PATH", ""))
                listing = _on_follower(
                    f"env XDG_RUNTIME_DIR={b_runtime} "
                    f"LD_LIBRARY_PATH={libraries} "
                    f"{shlex.quote(str(require_uniserve_binary()))} "
                    "storage-server list"
                )

                # A prompt A never saw brings nothing from A, and B's answer
                # is A's.
                unseen = _records("Unseen")
                b_unseen, unseen_found = _found(b_url, unseen)
                _, a_unseen = _ask(base_url, unseen)
                # A's Storage Server is gone: B computes what A stored.
                stop_storage_server(runtime)
                b_dropped, dropped_found = _found(b_url, dropped)
    finally:
        _release_leader_storage(runtime, tmp_path)

    assert reusable >= PEER_BLOCKS * BLOCK_SIZE, prompt_tokens
    assert shared_found["peer_hits"] >= PEER_BLOCKS * BLOCK_SIZE, shared_found
    # Every peer hit is a host hit B loaded.
    assert shared_found["host_hits"] >= shared_found["peer_hits"], shared_found
    assert f"link {peer} host -> host" in listing, listing
    assert b_shared == a_shared
    assert unseen_found["peer_hits"] == 0, unseen_found
    assert dropped_found["peer_hits"] == 0, dropped_found
    assert b_dropped == a_dropped
    assert b_unseen == a_unseen


def test_an_fp8_host_loads_a_bf16_hosts_blocks_by_converting_them(
    tmp_path: Path,
):
    _skip_without_two_hosts()
    shared = _records("Converted")
    runtime, env, peer = _leader_storage()
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    try:
        with server_process(
            _serve_command(qwen3_checkpoint(), port, "127.0.0.1"),
            base_url,
            tmp_path / "a.log",
            timeout_s=1800,
            env=env,
        ):
            # A computes the prompt in BF16 and stores its full blocks to its
            # host tier.
            prompt_tokens, _ = _ask(base_url, shared)
            reusable = _reusable(prompt_tokens)
            bf16 = _stored(base_url, shared, reusable)

            with _follower_deployment(
                tmp_path / "b.log",
                find_free_port(),
                FOLLOWER_DEVICE,
                [
                    "--set",
                    "kv_cache_config.dtype=float8_e4m3fn",
                    "--set",
                    f'kv_cache_config.peers=["{peer}"]',
                ],
            ) as (b_url, _):
                # The same prompt on B under a salt no cache holds: computed
                # cold, then over the FP8 blocks B wrote itself.
                cold = _candidates(b_url, shared, cache_salt="local")
                local = _candidates(b_url, shared, cache_salt="local")
                converted, found = tokens_found(
                    b_url, lambda: _candidates(b_url, shared)
                )
    finally:
        _release_leader_storage(runtime, tmp_path)

    (tmp_path / "answers.json").write_text(
        json.dumps(
            {
                "bf16": bf16,
                "cold": [token for token, _ in cold],
                "local": [token for token, _ in local],
                "converted": [token for token, _ in converted],
            }
        )
    )
    assert reusable >= PEER_BLOCKS * BLOCK_SIZE, prompt_tokens
    assert found["peer_hits"] >= PEER_BLOCKS * BLOCK_SIZE, found
    assert _logprobs_close(converted, local), (converted, local)
