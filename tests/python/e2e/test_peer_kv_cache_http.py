"""Prompt pages cross hosts through the Storage Servers.

**Peer reuse across hosts.** Qwen3 is served by two single-node deployments
of the same checkpoint, served name, cache dtype and block size: deployment
A on this host and deployment B on a second host, whose
``kv_cache_config.peers`` names A's Storage Server. A answers a long prompt
twice, the second time over its cached pages; its offload commits the
prompt's pages to its host KV cache. B then answers the same prompt: its
admission finds the pages in neither its device cache nor its host pools,
pins them at A's Storage Server by key, has its own Transfer Server READ
them through NIXL, and loads them as host pages. B's
``prefix_cache_peer_hits`` grows by the prompt's full pages, its
``storage-server list`` shows the measured link from A's host, and its
greedy tokens equal A's second answer, computed over the same pages. A
prompt A never saw reads nothing from A; once A's Storage Server stops, B
computes a prompt A answered, with the same tokens, and no request fails.

**Converted reuse across hosts.** Deployment B stores its KV cache in FP8
(``kv_cache_config.dtype=float8_e4m3fn``) while A stores BF16. B's pool
admits converted loads (OD-3: a dense source into an FP8 destination): it
asks A for the prompt's pages in A's BF16 layout too, has its Transfer
Server READ them into its staging tensor, and loads them through
``page_conversion`` and ``cache_copy_region``, which encode them as B's own
writes do. B's ``prefix_cache_peer_hits`` grows by the prompt's full pages,
and its greedy answer over the converted pages agrees with its answer over
the same prompt's pages it computed and wrote itself (another ``cache_salt``,
asked twice: computed cold, then reused from its device cache) by the rule
of vLLM's ``check_logprobs_close``: token for token until their first
difference, where each answer's token is among the other's ``TOP_LOGPROBS``
most likely tokens. Both answers compute the prompt's last tokens over FP8
pages; FP8 pages written whole from a page's amax and pages written token by
token round differently, so the two are not required to be identical. A
prompt computed whole rounds otherwise again, as the first case states.

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

from tests.python.e2e.http_helpers import (
    ATTENTION_BACKEND,
    find_free_port,
    granted_devices,
    metric,
    qwen3_checkpoint,
    require_uniserve_binary,
    server_process,
    stop_storage_server,
    worker_executable,
)
from tests.python.e2e.build_server import remote_build_server

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

FOLLOWER = os.environ.get("UNISERVE_H3_FOLLOWER")
LEADER_ADDRESS = os.environ.get("UNISERVE_H3_LEADER_ADDRESS")
NIXL_LIBRARY = os.environ.get("UNISERVE_NIXL_LIBRARY")
# The second host's GPU the cases serve on.
FOLLOWER_DEVICE = os.environ.get("UNISERVE_FOLLOWER_DEVICE", "0")

SERVED_MODEL = "qwen3"
BLOCK_SIZE = 64
MAX_TOKENS = 32
# Pages a prompt must fill for its peer hit to count.
PEER_PAGES = 16
# The candidates two greedy answers are compared at, as vLLM's model
# correctness tests compare logprobs.
TOP_LOGPROBS = 5
# The prompt tokens a deployment loaded from another host's pools.
PEER_HITS = "uniserve:prefix_cache_peer_hits"


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


def _settled(base_url: str, name: str, before: float) -> float:
    """The growth of counter `name` since `before`, once the scheduler's
    statistics, which reach the metrics every ten seconds, report it.
    """  # noqa: D205
    deadline = time.monotonic() + 30
    grown = metric(base_url, name) - before
    while grown == 0 and time.monotonic() < deadline:
        time.sleep(1)
        grown = metric(base_url, name) - before
    # One more report, so a growth spread over two is counted whole.
    time.sleep(11)
    return metric(base_url, name) - before


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
        "--worker-executable",
        str(worker_executable()),
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
            "UNISERVE_BUILD_SERVER_PYTHON",
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
            kept = shlex.quote(str(log.with_suffix(".storage.log")))
            # The bracket keeps the pattern from matching the shell that runs
            # it.
            server = shlex.quote(f"[s]torage-server serve.*{socket}")
            _on_follower(
                f"pkill -TERM -f {server}; "
                f"cp {runtime}/uniserve/storage-server.log {kept} "
                f"2>/dev/null; rm -rf {runtime}; true"
            )


def _ready_pages(listing: str) -> int:
    """The committed pages the host page pools of a listing hold."""
    return sum(int(match) for match in re.findall(r"(\d+) ready", listing))


def test_a_host_loads_a_prompts_pages_from_a_peer_hosts_host_kv_cache(
    tmp_path: Path,
):
    if FOLLOWER is None or LEADER_ADDRESS is None or NIXL_LIBRARY is None:
        pytest.skip(
            "peer reads across hosts need UNISERVE_H3_FOLLOWER, "
            "UNISERVE_H3_LEADER_ADDRESS and UNISERVE_NIXL_LIBRARY"
        )
    shared = _records("Shared")
    dropped = _records("Dropped")
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
            prompt_tokens, _ = _ask(base_url, shared)
            # Asked again, A reuses the prompt's pages from its device cache
            # and computes only its last tokens over them, as B does over the
            # pages it reads: the two compute the same tokens over the same
            # cached keys and values (a prompt computed whole may round
            # otherwise, as `test_host_kv_cache_http.py` states).
            _, a_shared = _ask(base_url, shared)
            _, a_dropped = _ask(base_url, dropped)
            # A's offload commits the prompts' full pages to its host pools.
            pages = 2 * ((prompt_tokens - 1) // BLOCK_SIZE)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                listed = subprocess.run(
                    [
                        str(require_uniserve_binary()),
                        "storage-server",
                        "list",
                        "--socket",
                        str(socket),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                if _ready_pages(listed) >= pages:
                    break
                time.sleep(1)
            assert _ready_pages(listed) >= pages, listed

            with _follower_deployment(
                tmp_path / "b.log",
                find_free_port(),
                FOLLOWER_DEVICE,
                ["--set", f"kv_cache_config.peers={peer}"],
            ) as (b_url, b_runtime):
                # B reads the shared prompt's pages from A.
                peer_before = metric(b_url, PEER_HITS)
                _, b_shared = _ask(b_url, shared)
                peer_hits = _settled(b_url, PEER_HITS, peer_before)
                libraries = shlex.quote(os.environ.get("LD_LIBRARY_PATH", ""))
                listing = _on_follower(
                    f"env XDG_RUNTIME_DIR={b_runtime} "
                    f"LD_LIBRARY_PATH={libraries} "
                    f"{shlex.quote(str(require_uniserve_binary()))} "
                    "storage-server list"
                )

                # A prompt A never saw reads nothing from A, and B's answer
                # is A's.
                unseen = _records("Unseen")
                unseen_before = metric(b_url, PEER_HITS)
                _, b_unseen = _ask(b_url, unseen)
                unseen_hits = _settled(b_url, PEER_HITS, unseen_before)
                _, a_unseen = _ask(base_url, unseen)
                # A's Storage Server is gone: B computes what A served.
                stop_storage_server(runtime)
                dropped_before = metric(b_url, PEER_HITS)
                _, b_dropped = _ask(b_url, dropped)
                dropped_hits = _settled(b_url, PEER_HITS, dropped_before)
    finally:
        stop_storage_server(runtime)
        storage_log = runtime / "uniserve" / "storage-server.log"
        if storage_log.is_file():
            shutil.copy(storage_log, tmp_path / "storage-server.log")
        shutil.rmtree(runtime, ignore_errors=True)

    full_pages = (prompt_tokens - 1) // BLOCK_SIZE
    assert full_pages >= PEER_PAGES
    assert peer_hits >= PEER_PAGES * BLOCK_SIZE, peer_hits
    assert f"link {peer} host -> host" in listing, listing
    assert b_shared == a_shared
    assert unseen_hits == 0
    assert dropped_hits == 0
    assert b_dropped == a_dropped
    assert b_unseen == a_unseen


def test_an_fp8_host_loads_a_bf16_hosts_pages_by_converting_them(
    tmp_path: Path,
):
    if FOLLOWER is None or LEADER_ADDRESS is None or NIXL_LIBRARY is None:
        pytest.skip(
            "peer reads across hosts need UNISERVE_H3_FOLLOWER, "
            "UNISERVE_H3_LEADER_ADDRESS and UNISERVE_NIXL_LIBRARY"
        )
    shared = _records("Converted")
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
            # A computes the prompt in BF16 and commits its full pages to
            # its host pools.
            prompt_tokens, _ = _ask(base_url, shared)
            bf16 = _candidates(base_url, shared)
            pages = (prompt_tokens - 1) // BLOCK_SIZE
            deadline = time.monotonic() + 120
            listed = ""
            while time.monotonic() < deadline:
                listed = subprocess.run(
                    [
                        str(require_uniserve_binary()),
                        "storage-server",
                        "list",
                        "--socket",
                        str(socket),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                if _ready_pages(listed) >= pages:
                    break
                time.sleep(1)
            assert _ready_pages(listed) >= pages, listed

            with _follower_deployment(
                tmp_path / "b.log",
                find_free_port(),
                FOLLOWER_DEVICE,
                [
                    "--set",
                    "kv_cache_config.dtype=float8_e4m3fn",
                    "--set",
                    f"kv_cache_config.peers={peer}",
                ],
            ) as (b_url, _):
                # The same prompt on B under a salt no cache holds: computed
                # cold, then over the FP8 pages B wrote itself.
                cold = _candidates(b_url, shared, cache_salt="local")
                local = _candidates(b_url, shared, cache_salt="local")
                peer_before = metric(b_url, PEER_HITS)
                converted = _candidates(b_url, shared)
                peer_hits = _settled(b_url, PEER_HITS, peer_before)
    finally:
        stop_storage_server(runtime)
        storage_log = runtime / "uniserve" / "storage-server.log"
        if storage_log.is_file():
            shutil.copy(storage_log, tmp_path / "storage-server.log")
        shutil.rmtree(runtime, ignore_errors=True)

    (tmp_path / "answers.json").write_text(
        json.dumps(
            {
                "bf16": [token for token, _ in bf16],
                "cold": [token for token, _ in cold],
                "local": [token for token, _ in local],
                "converted": [token for token, _ in converted],
            }
        )
    )
    assert pages >= PEER_PAGES
    assert peer_hits >= PEER_PAGES * BLOCK_SIZE, peer_hits
    assert _logprobs_close(converted, local), (converted, local)
