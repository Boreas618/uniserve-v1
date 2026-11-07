"""A world change continues the requests in flight on a successor world.

The successor serves on trial until the operator commits it or a fault
rolls it back.

Qwen3-MoE (``UNISERVE_QWEN3_MODEL``) streams a long greedy count. While it
streams, ``PUT /admin/config`` changes the decode batch sizes the CUDA graphs
capture, which only a new world executes: a successor world starts beside
the serving one on the same GPUs and maps its weights and KV cache segments
read-only; at a step boundary the serving world parks, its streaming
request leaving it as migratable after every token it generated; the
successor imports its prefix cache index and serves on trial, and the
daemon submits the request again with those tokens. The client reads one
uninterrupted stream, equal to the count the committed world decodes for
the same request alone, with the same usage.

- A commit releases the previous world: its engine's process exits and the
  slot it held empties. The trial refuses another change meanwhile. The
  deployment runs on one GPU, as two schedulers sharing their experts
  (``--data-parallel-size 2`` with expert parallelism), or as attention and
  expert workers exchanging through DeepEP.
- A change of the executable alone, whose functions and formats stay the
  same, invalidates nothing: after the commit the successor maps every
  retained weight, a cached prompt finds as many tokens cached as before
  the change, on the device and, once fill prompts pushed its pages out of
  it, in the host KV cache, and the Storage Server shoots down no writer.
- An operator's rollback resumes the previous world, parked or started
  again on the weights it kept (``update_config.previous_world``); the
  stream continues on it, and the configuration is marked unavailable:
  applying it again is refused unless forced.

The tests print the pause of every promotion and rollback. Each runs a
Storage Server of its own: ``XDG_RUNTIME_DIR`` points the daemon at a
directory of the test's, and the server the daemon starts there is stopped
at the end.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import tomllib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    dedicated_experts,
    find_free_port,
    require_uniserve_binary,
    scaled,
    server_process,
    worker_executable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

MAX_TOKENS = 16384
# Tokens the client reads before it asks for the change.
EARLY_TOKENS = 64
PROMPT = (
    "Count from 1 to 5000, separated by commas and spaces. "
    "Write only the numbers."
)
# A prompt of several pages, whose pages the serving world publishes.
CACHED_PROMPT = "Describe the sea in one sentence. " * 40
# A prompt of several pages only a world on trial computes, which shares no
# page with the other prompts.
TRIAL_PROMPT = "List the colours of the rainbow and what each means. " * 40
# Options every deployment of the tests serves with: few captured graphs,
# so that worlds start while the count streams, and a small KV cache.
OPTIONS = (
    "dtype=bfloat16",
    "kv_cache_config.max_tokens=32768",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
)
# The change every test applies: graphs of other decode batch sizes, which
# a new world captures.
CHANGE = "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8, 16]"


def _checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_QWEN3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_MODEL must name the model checkpoint")
    return checkpoint


def _log_lines(log: Path) -> list[str]:
    """The lines of `log` without their terminal colors."""
    colors = re.compile(r"\x1b\[[0-9;]*m")
    return [colors.sub("", line) for line in log.read_text().splitlines()]


def _pids(log: Path, role: str) -> set[int]:
    """The process ids the log names for `role` (``Daemon`` or ``Engine``)."""
    pattern = re.compile(rf"\({role} pid=(\d+)\)")
    return {
        int(match.group(1))
        for line in _log_lines(log)
        if (match := pattern.search(line))
    }


def _runs(pid: int) -> bool:
    """Whether process `pid` runs; a zombie does not."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _wait_until(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise TimeoutError(f"{what} within {timeout_s} s")


def _stop_storage_server(socket: Path) -> None:
    """Stop the Storage Server serving `socket`, found by its command line."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in command and str(socket).encode() in command:
            os.kill(int(entry.name), signal.SIGTERM)


def _device() -> str:
    """The GPU a one-GPU deployment serves on (``UNISERVE_WORLD_DEVICE``,
    0 by default), so that cases run beside each other on other GPUs.
    """  # noqa: D205
    return os.environ.get("UNISERVE_WORLD_DEVICE", "0")


def _devices() -> str:
    """The two GPUs a two-GPU deployment serves on
    (``UNISERVE_WORLD_DEVICES``, ``0,1`` by default), so that cases run
    beside each other on other GPUs.
    """  # noqa: D205
    return os.environ.get("UNISERVE_WORLD_DEVICES", "0,1")


def _topology(topology: str, tmp_path: Path) -> tuple[str, tuple[str, ...]]:
    """The devices and extra options of a deployment.

    ``single``: one GPU. ``experts``: two schedulers sharing their experts
    over the all-to-all. ``dedicated``: attention and expert workers
    exchanging through DeepEP.
    """
    if topology == "single":
        return _device(), ()
    if topology == "experts":
        return _devices(), (
            "--data-parallel-size",
            "2",
            "--set",
            "moe_config.expert_parallel=true",
        )
    return _devices(), (
        "--worker-groups",
        str(dedicated_experts(tmp_path)),
        "--set",
        "moe_config.expert_parallel=true",
        "--set",
        "moe_config.exchange=deepep",
    )


@contextmanager
def _serve(
    tmp_path: Path,
    name: str,
    devices: str | None = None,
    options: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    storage: tuple[str, ...] | None = None,
) -> Iterator[tuple[str, Path]]:
    """A deployment serving `name` on `devices`, its URL and its log.

    With `storage`, a Storage Server started by hand with those options
    serves the deployment, which starts none of its own.
    """
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-", dir="/tmp"))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        _checkpoint(),
        "--served-model-name",
        name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
    ]
    for option in OPTIONS:
        command += ["--set", option]
    command += options
    log = tmp_path / "server.log"
    environment = {
        "CUDA_VISIBLE_DEVICES": devices or _device(),
        "XDG_RUNTIME_DIR": str(runtime),
        **(env or {}),
    }
    try:
        if storage is not None:
            socket = runtime / "uniserve" / "storage.sock"
            socket.parent.mkdir(mode=0o700)
            subprocess.run(
                [
                    str(require_uniserve_binary()),
                    "storage",
                    "serve",
                    "--socket",
                    str(socket),
                    "--daemon",
                    *storage,
                ],
                env={**os.environ, **environment},
                check=True,
                timeout=120,
            )
        with server_process(
            command, base_url, log, timeout_s=900, env=environment
        ):
            yield base_url, log
    finally:
        _stop_storage_server(runtime / "uniserve" / "storage.sock")
        storage_log = runtime / "uniserve" / "storage-server.log"
        if storage_log.is_file():
            shutil.copy(storage_log, tmp_path / "storage-server.log")
        shutil.rmtree(runtime, ignore_errors=True)


def _body(name: str, stream: bool) -> dict:
    body = {
        "model": name,
        "messages": [{"role": "user", "content": PROMPT}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_completion_tokens": MAX_TOKENS,
        "stream": stream,
        "logprobs": True,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _chunk_tokens(line: str) -> tuple[list[str], dict | None, bool]:
    """The tokens of a stream line, its usage, and whether it ends the
    stream.
    """  # noqa: D205
    if line == "data: [DONE]":
        return [], None, True
    if not line.startswith("data: "):
        return [], None, False
    chunk = json.loads(line[6:])
    tokens = [
        entry["token"]
        for choice in chunk.get("choices", ())
        for entry in ((choice.get("logprobs") or {}).get("content") or ())
    ]
    return tokens, chunk.get("usage"), False


def _ask(base_url: str, name: str) -> tuple[list[str], dict]:
    """The count's tokens and usage for the request alone."""
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json=_body(name, stream=False),
        timeout=1800,
    )
    response.raise_for_status()
    answer = response.json()
    tokens = [
        entry["token"] for entry in answer["choices"][0]["logprobs"]["content"]
    ]
    return tokens, answer["usage"]


def _chat(base_url: str, name: str, prompt: str, tokens: int = 32) -> str:
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": name,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_completion_tokens": tokens,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


class Streaming:
    """The count streaming on a thread of its own: the tokens it read, its
    usage, and how many tokens it read once `mark` was called.
    """  # noqa: D205

    def __init__(self, base_url: str, name: str, *, body: dict | None = None):
        self.tokens: list[str] = []
        self.usage: dict | None = None
        self.marked: int | None = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._read,
            args=(
                base_url,
                body if body is not None else _body(name, stream=True),
            ),
            daemon=True,
        )
        self._thread.start()

    def _read(self, base_url: str, body: dict) -> None:
        try:
            with httpx.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=body,
                timeout=1800,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    tokens, usage, done = _chunk_tokens(line)
                    if done:
                        break
                    self.tokens.extend(tokens)
                    self.usage = usage or self.usage
        except BaseException as error:  # noqa: BLE001 - the test reports it.
            self.error = error

    def wait_for(self, count: int) -> None:
        _wait_until(
            lambda: len(self.tokens) >= count or not self._thread.is_alive(),
            600,
            "the count did not stream",
        )
        assert len(self.tokens) >= count, "the stream ended early"

    def mark(self) -> None:
        self.marked = len(self.tokens)

    @property
    def finished(self) -> bool:
        return not self._thread.is_alive()

    def join(self) -> None:
        self._thread.join(timeout=scaled(1800))
        assert not self._thread.is_alive(), "the count did not end"
        assert self.error is None, f"the stream failed: {self.error!r}"


def _config(base_url: str) -> dict:
    return httpx.get(f"{base_url}/admin/config", timeout=10).json()


def _apply(
    base_url: str, *assignments: str, force: bool = False
) -> httpx.Response:
    return httpx.put(
        f"{base_url}/admin/config",
        json={"set": list(assignments), "force": force},
        timeout=scaled(1800),
    )


def _admin(
    base_url: str, path: str, body: dict | None = None
) -> httpx.Response:
    return httpx.post(
        f"{base_url}/admin/{path}", json=body or {}, timeout=scaled(1800)
    )


def _world_change(base_url: str, *assignments: str) -> dict:
    """Applies `assignments` as a world change; returns its report."""
    response = _apply(base_url, *assignments)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["change"] == "world", result
    return result["world"]


def _change_under_traffic(
    base_url: str, name: str, streaming: Streaming, assignments: tuple[str, ...]
) -> tuple[dict, list[Streaming]]:
    """Keep one request running throughout preparation, retaining every result.

    A finite request can finish while a successor warms. Start the next with
    the same prompt and token limit so the test observes migration of a live
    request regardless of the number of requests preparation spans.
    """
    streams = [streaming]
    with ThreadPoolExecutor(max_workers=1) as executor:
        change = executor.submit(_world_change, base_url, *assignments)
        while not change.done():
            if streams[-1].finished:
                streams[-1].join()
                streams.append(Streaming(base_url, name))
            time.sleep(0.01)
        report = change.result()
    streams[-1].mark()
    return report, streams


def _planned_storage(base_url: str, *assignments: str) -> dict:
    """The storage plan of a dry run of `assignments`, which changes nothing:
    for every GPU, the serving world's KV cache limit beside a successor
    from the GPU's ledger, and every host's admission of host storage.
    """  # noqa: D205
    response = httpx.put(
        f"{base_url}/admin/config",
        json={"set": list(assignments), "dry_run": True},
        timeout=1800,
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["change"] == "planned", result
    return result["plan"]["storage"]


def _ledgers(status: dict) -> list[dict]:
    """Every GPU's storage ledger ``GET /admin/config`` reports, each of
    whose entries' bytes sum to no more than its quota.
    """  # noqa: D205
    nodes = status["ledgers"]
    assert isinstance(nodes, list) and nodes, status
    ledgers = [ledger for node in nodes for ledger in node["ledgers"]]
    assert ledgers, status
    for ledger in ledgers:
        used = sum(entry["bytes"] for entry in ledger["entries"])
        assert ledger["quota"] is not None and used <= ledger["quota"], ledger
    return ledgers


def _worlds_of(ledgers: list[dict], item: str) -> set[tuple[int, ...]]:
    """The sets of worlds holding the ledgers' entries of `item`."""
    return {
        tuple(entry["worlds"])
        for ledger in ledgers
        for entry in ledger["entries"]
        if entry["item"] == item
    }


def test_independent_replicas_roll_on_disjoint_devices(tmp_path: Path):
    """One replica trials at a time while the shared service keeps serving."""
    name = "qwen3-replicas"
    devices = os.environ["UNISERVE_WORLD_DEVICES"]
    assert len(devices.split(",")) >= 3
    with _serve(tmp_path, name, devices, ("--data-parallel-size", "2")) as (
        base_url,
        _log,
    ):
        before = _config(base_url)["replicas"]
        assert len(before) == 2
        worlds = [replica["slots"][0]["world"] for replica in before]
        assert len(set(worlds)) == 2
        expected = _chat(base_url, name, CACHED_PROMPT)
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        report, streams = _change_under_traffic(
            base_url, name, streaming, (CHANGE,)
        )
        assert report["previous_world"] == worlds[0]
        status = _config(base_url)
        assert status["pending_replicas"] == [1]
        assert status["replicas"][1]["slots"][0]["world"] == worlds[1]
        for ledger in _ledgers(status):
            resident = {
                world
                for entry in ledger["entries"]
                if entry["item"] == "process"
                for world in entry["worlds"]
            }
            assert not {worlds[0], report["world"]} <= resident
        assert _chat(base_url, name, CACHED_PROMPT) == expected
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        status = _config(base_url)
        assert status["replicas"][0]["trial"] is None
        assert status["replicas"][1]["trial"]["previous_world"] == worlds[1]
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        status = _config(base_url)
        assert status["pending_replicas"] == []
        assert all(replica["trial"] is None for replica in status["replicas"])
        for stream in streams:
            stream.join()
        assert _chat(base_url, name, CACHED_PROMPT) == expected


def test_replica_rollback_and_daemon_restart_preserve_serving(tmp_path: Path):
    """A selected replica rolls back; a hot restart adopts both owners."""
    name = "qwen3-replica-recovery"
    devices = os.environ["UNISERVE_WORLD_DEVICES"]
    assert len(devices.split(",")) >= 3
    with _serve(tmp_path, name, devices, ("--data-parallel-size", "2")) as (
        base_url,
        log,
    ):
        successor = None
        try:
            before = _config(base_url)["replicas"]
            worlds = [replica["slots"][0]["world"] for replica in before]
            expected = _chat(base_url, name, CACHED_PROMPT)
            applied = httpx.put(
                f"{base_url}/admin/config",
                json={"set": [CHANGE], "replica": 1},
                timeout=1800,
            )
            assert applied.status_code == 200, applied.text
            status = _config(base_url)
            assert status["replicas"][0]["slots"][0]["world"] == worlds[0]
            assert status["replicas"][1]["trial"]["previous_world"] == worlds[1]
            streaming = Streaming(base_url, name)
            streaming.wait_for(EARLY_TOKENS)
            rolled = _admin(
                base_url, "config/rollback", {"reason": "operator rollback"}
            )
            assert rolled.status_code == 200, rolled.text
            restored = _config(base_url)
            assert [
                replica["slots"][0]["world"] for replica in restored["replicas"]
            ] == worlds
            assert restored["pending_replicas"] == []
            assert _chat(base_url, name, CACHED_PROMPT) == expected
            engines = _pids(log, "Engine")
            restarted = _admin(base_url, "restart")
            assert restarted.status_code == 200, restarted.text
            successor = restarted.json()["result"]["pid"]
            adopted = _config(base_url)
            assert [
                replica["slots"][0]["world"] for replica in adopted["replicas"]
            ] == worlds
            assert _pids(log, "Engine") == engines
            assert _chat(base_url, name, CACHED_PROMPT) == expected
            streaming.join()
        finally:
            if successor is not None and _runs(successor):
                os.kill(successor, signal.SIGTERM)
                _wait_until(
                    lambda: not _runs(successor),
                    120,
                    "successor daemon did not stop",
                )


def test_splitting_an_expert_group_rolls_back_as_one_transaction(
    tmp_path: Path,
):
    """A split trial restores its coupled predecessor as one transaction."""
    name = "qwen3-group-rollback"
    devices = os.environ["UNISERVE_WORLD_DEVICES"]
    options = (
        "--data-parallel-size",
        "2",
        "--set",
        "moe_config.expert_parallel=true",
    )
    with _serve(tmp_path, name, devices, options) as (base_url, _log):
        before = _config(base_url)
        previous = before["replicas"][0]["slots"][0]["world"]
        expected = _chat(base_url, name, CACHED_PROMPT)
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        report, streams = _change_under_traffic(
            base_url, name, streaming, ("moe_config.expert_parallel=false",)
        )
        assert report["previous_worlds"] == [previous]
        assert len(report["worlds"]) == 2
        status = _config(base_url)
        assert status["trial"]["worlds"] == report["worlds"]
        assert len(status["replicas"]) == 2
        # Each Replica computes every expert of its own tokens, where the
        # group dispatched them among its ranks, beside the counts still
        # streaming: its greedy answer may round otherwise, so it is checked
        # for an answer only. The group the rollback restores answers as
        # before once the streams end.
        assert _chat(base_url, name, CACHED_PROMPT)
        rolled = _admin(
            base_url, "config/rollback", {"reason": "operator rollback"}
        )
        assert rolled.status_code == 200, rolled.text
        assert rolled.json()["result"]["rolled_back"] == report["worlds"]
        restored = _config(base_url)
        assert restored["trial"] is None
        assert len(restored["replicas"]) == 1
        assert restored["replicas"][0]["slots"][0]["world"] == previous
        for stream in streams:
            stream.join()
        assert _chat(base_url, name, CACHED_PROMPT) == expected


@pytest.mark.parametrize("topology", ["single", "experts", "dedicated"])
def test_a_world_change_continues_a_streaming_request_and_commits(
    tmp_path: Path, topology: str
):
    devices, options = _topology(topology, tmp_path)
    name = "qwen3-world-change"
    with _serve(tmp_path, name, devices, options) as (base_url, log):
        before = _config(base_url)
        (serving,) = [
            slot for slot in before["slots"] if slot["state"] != "empty"
        ]
        assert serving["state"] == "executing", before
        engines_before = _pids(log, "Engine")
        # Every GPU's ledger books the serving world alone, within its
        # quota: its weights, KV cache segments, graph pool and process.
        ledgers = _ledgers(before)
        for item in ("weights", "cache_segment", "graph_pool", "process"):
            assert _worlds_of(ledgers, item) == {(serving["world"],)}, ledgers

        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        # A dry run plans the change on every GPU's ledger and changes
        # nothing: the successor fits beside the serving world.
        _wait_until(
            lambda: _prefix_counter(base_url, "queries") > 0,
            30,
            "the original request's cache query was not reported",
        )
        queries_before = _prefix_counter(base_url, "queries")
        hits_before = _prefix_hits(base_url)
        storage = _planned_storage(base_url, CHANGE)
        assert storage["feasible"] is True, storage
        assert storage["gpus"] and storage["hosts"], storage
        for gpu in storage["gpus"]:
            # A GPU holding a KV cache keeps at least its first segment; a
            # dedicated expert GPU holds none, so its floor is zero.
            if gpu["segments_held"]:
                assert gpu["kv_limit"] >= gpu["kv_floor"] > 0, gpu
            else:
                assert gpu["kv_floor"] == 0 and gpu["kv_limit"] >= 0, gpu
            assert gpu["successor"]["graph_pool"] > 0, gpu
            print(
                f"GPU {gpu['gpu']}: quota {gpu['quota']}, serving KV cache "
                f"{gpu['serving_kv_cache']}, limit beside the successor "
                f"{gpu['kv_limit']}, {gpu['segments_kept']} of "
                f"{gpu['segments_held']} segments kept"
            )
        report = _world_change(base_url, CHANGE)
        streaming.mark()
        world = report["world"]
        assert report["previous_world"] == serving["world"], report
        assert report["storage"]["feasible"] is True, report
        print(
            f"promotion pause: {report['pause_seconds']:.3f} s "
            f"(park {report['park_seconds']:.3f} s, "
            f"preparation {report['prepare_seconds']:.1f} s)"
        )

        # The successor serves on trial beside the parked previous world, and
        # the trial refuses another change. Both worlds share each GPU's
        # quota: the weights and the segments the successor maps are booked
        # once, held by both, beside each world's graph pool and process.
        status = _config(base_url)
        assert status["trial"]["world"] == world, status
        assert status["slots"] == [
            {"world": world, "state": "executing"},
            {"world": serving["world"], "state": "parked"},
        ], status
        ledgers = _ledgers(status)
        both = (serving["world"], world)
        assert _worlds_of(ledgers, "weights") == {both}, ledgers
        assert both in _worlds_of(ledgers, "cache_segment"), ledgers
        assert _worlds_of(ledgers, "process") == {(serving["world"],), (world,)}
        refused = _apply(base_url, "max_running_requests=4")
        assert refused.status_code == 409, refused.text

        # The operator commits while the count streams on the successor,
        # which writes new pages only to its own segments until then.
        streaming.wait_for(streaming.marked + 64)
        _wait_until(
            lambda: _prefix_counter(base_url, "queries") > queries_before,
            30,
            "the resumed request's cache query was not reported",
        )
        # All computed tokens, including the final partial page, are
        # reusable. Admission leaves only the last emitted token to run.
        reused = _prefix_hits(base_url) - hits_before
        queried = _prefix_counter(base_url, "queries") - queries_before
        assert reused == queried > 0, (reused, queried)
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        assert committed.json()["result"]["world"] == world
        status = _config(base_url)
        assert status["trial"] is None, status
        assert status["slots"] == [
            {"world": world, "state": "executing"},
            {"world": None, "state": "empty"},
        ], status
        graphs = tomllib.loads(status["config"])["cuda_graph_config"]
        assert graphs["decode_batch_sizes"] == [1, 2, 4, 8, 16], graphs
        # The commit released everything only the previous world held: the
        # segments the successor mapped are its own on the ledger now. The
        # previous world's workers end their bookings as their processes
        # exit, a moment after its engine's.
        items = ("weights", "cache_segment", "graph_pool", "process")
        _wait_until(
            lambda: all(
                _worlds_of(_ledgers(_config(base_url)), item) == {(world,)}
                for item in items
            ),
            30,
            "the ledgers did not book the committed world alone",
        )
        streaming.join()
        reference, reference_usage = _ask(base_url, name)

    lines = _log_lines(log)
    for line in lines:
        if "a migrated request generates" in line:
            print(f"migrated request: {line.split(': ', 1)[-1]}")
    assert any("a request migrated" in line for line in lines), (
        "the streaming request did not migrate to the successor"
    )
    assert any(
        re.search(
            r"imported the predecessor's prefix cache index.*pages=[1-9]", line
        )
        for line in lines
    ), "the successor imported no page of the previous world's index"
    assert len(streaming.tokens) > streaming.marked, (
        "the stream ended before the successor served"
    )
    # The previous world's engine exited at the commit.
    (previous_engine,) = engines_before
    assert not _runs(previous_engine)
    assert streaming.tokens == reference
    assert streaming.usage == reference_usage


# Distinct prompts whose pages together outgrow the device KV cache (32768
# tokens), pushing out the pages of a prompt asked before them, which the
# host KV cache keeps.
EVICTING_FILLS = 6


def _host_hits_of(
    base_url: str, name: str, prompt: str, first: int
) -> tuple[float, float]:
    """The prompt tokens a query of `prompt` finds cached once fill prompts
    numbered from `first` pushed its pages out of the device KV cache, and
    those of them loaded from the host KV cache.
    """  # noqa: D205
    for index in range(first, first + EVICTING_FILLS):
        _chat(base_url, name, f"{index}: {FILL_PROMPT}", tokens=1)
    host = _prefix_counter(base_url, "host_hits")
    _, hits = _hits_of(base_url, name, prompt)
    return hits, _prefix_counter(base_url, "host_hits") - host


def test_a_resource_change_continues_requests_with_a_smaller_cache(
    tmp_path: Path,
):
    """Quotas, queue depth and cache capacity change through a world."""
    name = "qwen3-resource-change"
    assignments = (
        "mem_fraction_static=0.8",
        "placement_config.queue_depth=3",
        "kv_cache_config.max_tokens=32768",
        "kv_cache_config.host_gib=8",
    )
    # The cache also holds graph scratch units. Both capacities must fit
    # the full count beside those units throughout successor preparation.
    with _serve(
        tmp_path,
        name,
        options=("--set", "kv_cache_config.max_tokens=65536"),
    ) as (base_url, _):
        before = _config(base_url)
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        report, streams = _change_under_traffic(
            base_url, name, streaming, assignments
        )
        assert report["previous_world"] == before["slots"][0]["world"]
        assert report["storage"]["feasible"] is True, report
        assert report["numerics"] == "unchanged", report
        kv_pages = report["derived_state"]["kv_pages"]
        assert kv_pages["treatment"] == "moved", report
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        configured = tomllib.loads(_config(base_url)["config"])
        assert configured["mem_fraction_static"] == 0.8
        assert configured["placement_config"]["queue_depth"] == 3
        assert configured["kv_cache_config"]["max_tokens"] == 32768
        assert configured["kv_cache_config"]["host_gib"] == 8
        for stream in streams:
            stream.join()
        reference, usage = _ask(base_url, name)
        for stream in streams:
            assert stream.tokens == reference
            assert stream.usage == usage


def test_a_change_of_the_executable_alone_invalidates_nothing(tmp_path: Path):
    name = "qwen3-same-keys"
    with _serve(tmp_path, name) as (base_url, _):
        # The prompt's pages are computed once and then found cached, on the
        # device and, once pushed out of it, in the host KV cache.
        answer = _chat(base_url, name, CACHED_PROMPT)
        cached_answer, hits = _hits_of(base_url, name, CACHED_PROMPT)
        assert cached_answer == answer and hits > 0, (cached_answer, hits)
        host_before = _host_hits_of(base_url, name, CACHED_PROMPT, 0)
        assert host_before[1] > 0, host_before

        # Graph capture changes none of these page keys.
        report = _world_change(base_url, CHANGE)
        weights = report["derived_state"]["weights"]["treatment"]
        assert (report["numerics"], weights) == ("unchanged", "mapped"), report
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text

        # The committed world finds as many of the prompt's tokens cached,
        # on the device and in the host KV cache, and answers it as before.
        assert _hits_of(base_url, name, CACHED_PROMPT) == (answer, hits)
        host_after = _host_hits_of(
            base_url, name, CACHED_PROMPT, EVICTING_FILLS
        )
        assert host_after == host_before, (host_after, host_before)

    # No writer of the deployment's storage was ended to pass its access on.
    storage_log = (tmp_path / "storage-server.log").read_text()
    assert "shooting down a previous writer" not in storage_log


@pytest.mark.parametrize("previous_world", ["parked", "weights"])
def test_an_operator_rollback_resumes_the_previous_world(
    tmp_path: Path, previous_world: str
):
    name = "qwen3-world-rollback"
    with _serve(tmp_path, name) as (base_url, log):
        cached = _chat(base_url, name, CACHED_PROMPT)
        (serving,) = [
            slot
            for slot in _config(base_url)["slots"]
            if slot["state"] != "empty"
        ]
        (engine_before,) = _pids(log, "Engine")

        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        report = _world_change(
            base_url, CHANGE, f"update_config.previous_world={previous_world}"
        )
        world = report["world"]
        status = _config(base_url)
        assert status["slots"][1] == {
            "world": serving["world"],
            "state": previous_world,
        }, status
        # The world on trial publishes the pages of a prompt only it
        # computes.
        trial_answer = _chat(base_url, name, TRIAL_PROMPT)
        if previous_world == "weights":
            _wait_until(
                lambda: not _runs(engine_before),
                120,
                "the previous world's engine did not exit",
            )

            # The previous world keeps its weights alone: each GPU's ledger
            # books nothing else of it once its workers exited, the
            # segments the successor maps being the successor's.
            def kept() -> set[str]:
                return {
                    entry["item"]
                    for ledger in _ledgers(_config(base_url))
                    for entry in ledger["entries"]
                    if serving["world"] in entry["worlds"]
                }

            _wait_until(
                lambda: kept() == {"weights"},
                60,
                "the previous world kept more than its weights",
            )

        rolled = _admin(
            base_url, "config/rollback", {"reason": "the operator's check"}
        )
        assert rolled.status_code == 200, rolled.text
        result = rolled.json()["result"]
        assert result["rolled_back"] == world, result
        assert result["world"] == serving["world"], result
        streaming.mark()
        print(
            f"{previous_world}: promotion pause "
            f"{report['pause_seconds']:.3f} s, "
            f"rollback pause {result['pause_seconds']:.3f} s"
        )
        streaming.join()

        status = _config(base_url)
        assert status["trial"] is None, status
        assert status["slots"][0]["world"] == serving["world"], status
        (unavailable,) = status["unavailable"]
        assert unavailable["world"] == world, unavailable
        assert unavailable["reason"] == "the operator's check", unavailable
        reference, reference_usage = _ask(base_url, name)
        assert streaming.tokens == reference
        assert streaming.usage == reference_usage
        # The previous world reads the pages it published before the change,
        # from its device cache or the host KV cache, and none the discarded
        # world published.
        answer, hits = _hits_of(base_url, name, CACHED_PROMPT)
        assert answer == cached
        assert hits > 0
        answer, hits = _hits_of(base_url, name, TRIAL_PROMPT)
        assert answer == trial_answer
        assert hits == 0

        # The configuration is unavailable until forced.
        again = _apply(
            base_url, CHANGE, f"update_config.previous_world={previous_world}"
        )
        assert again.status_code == 400, again.text
        assert "rolled back" in again.text, again.text
        forced = _apply(
            base_url,
            CHANGE,
            f"update_config.previous_world={previous_world}",
            force=True,
        )
        assert forced.status_code == 200, forced.text
        assert forced.json()["result"]["change"] == "world"
        assert _admin(base_url, "config/commit").status_code == 200
        assert _chat(base_url, name, CACHED_PROMPT) == cached

    lines = _log_lines(log)
    if previous_world == "parked":
        # The parked engine resumed: the same process served before and after.
        assert any("the parked scheduler resumes" in line for line in lines)
    else:
        # The previous world started again on the weights it kept.
        assert any(
            re.search(r"mapped the retained weights .*missing=0", line)
            for line in lines
        ), "the restarted previous world did not map its weights"
    assert (
        sum(
            "uniserve.loading.rank: loaded numerical model" in line
            for line in lines
        )
        == 1
    ), "a world read the checkpoint again rather than the retained weights"


def _forwards(base_url: str, mode: str) -> float:
    """The forwards every world ran in ``mode`` so far (``eager`` or
    ``graph_replay``; ``uniserve:worker_forward_mode_counts``), once the
    reports of what came before arrived.
    """  # noqa: D205
    time.sleep(REPORT_SETTLE)
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("uniserve:worker_forward_mode_counts")
        and f'mode="{mode}"' in line
    )


@pytest.mark.parametrize("topology", ["single", "experts", "dedicated"])
def test_a_serving_world_gives_its_graph_storage_to_a_preparing_successor(
    tmp_path: Path, topology: str
):
    devices, options = _topology(topology, tmp_path)
    name = "qwen3-world-eager"
    with _serve(tmp_path, name, devices, options) as (base_url, _log):
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        eager = _forwards(base_url, "eager")

        # The serving world decodes the count eagerly while the successor
        # prepares in its graph storage, once its eager steps made the
        # baseline its successor's transfers yield to.
        report = _world_change(
            base_url, CHANGE, "update_config.eager_prepare=true"
        )
        assert _forwards(base_url, "eager") > eager
        assert report["eager_baseline_seconds"] is not None, report
        print(
            f"{topology}: eager baseline "
            f"{report['eager_baseline_seconds']:.1f} s, "
            f"preparation {report['prepare_seconds']:.1f} s, "
            f"transfer halvings {report['transfer_halvings']}, "
            f"restorations {report['transfer_restorations']}"
        )

        # Resumed, the previous world replays its graphs again.
        rolled = _admin(
            base_url, "config/rollback", {"reason": "the operator's check"}
        )
        assert rolled.status_code == 200, rolled.text
        replays = _forwards(base_url, "graph_replay")
        eager = _forwards(base_url, "eager")
        streaming.wait_for(len(streaming.tokens) + 64)
        assert _forwards(base_url, "graph_replay") > replays
        assert _forwards(base_url, "eager") == eager
        streaming.join()


# How long the counters an engine reports every second take to include
# what a request it answered counted: two report intervals.
REPORT_SETTLE = 2.0


def _prefix_counter(base_url: str, name: str) -> float:
    """A prefix cache counter every world counted so far, in tokens
    (``uniserve:prefix_cache_<name>``).
    """  # noqa: D205
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith(f"uniserve:prefix_cache_{name}")
    )


def _prefix_hits(base_url: str) -> float:
    """The prefix cache hits every world counted so far."""
    return _prefix_counter(base_url, "hits")


def _hits_of(base_url: str, name: str, prompt: str) -> tuple[str, float]:
    """The answer to `prompt` and the prompt tokens it found cached.

    Its query is counted at admission and the pages it loads from the host
    KV cache once they arrive, before it computes, so the hits are read once
    the report carrying the query arrived and another followed.
    """
    queries = _prefix_counter(base_url, "queries")
    hits = _prefix_hits(base_url)
    answer = _chat(base_url, name, prompt)
    _wait_until(
        lambda: _prefix_counter(base_url, "queries") > queries,
        30,
        "the request's prefix cache query was not reported",
    )
    time.sleep(REPORT_SETTLE)
    return answer, _prefix_hits(base_url) - hits


def _updated_checkpoint() -> str:
    """Another checkpoint of the model, of the same files but for other
    weights (``UNISERVE_QWEN3_UPDATED_MODEL``).
    """  # noqa: D205
    checkpoint = os.environ.get("UNISERVE_QWEN3_UPDATED_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail(
            "UNISERVE_QWEN3_UPDATED_MODEL must name another checkpoint of "
            "the model, with other weights"
        )
    return checkpoint


# The share of the GPU a deployment whose successor loads every weight
# again serves with: both worlds' weights, the successor's graph share and
# both KV caches fit in it.
WEIGHTS_SHARE = ("--set", "mem_fraction_static=0.9")


def _weights_of(base_url: str) -> set[tuple[int, ...]]:
    """The sets of worlds holding weights on the deployment's GPUs."""
    return _worlds_of(_ledgers(_config(base_url)), "weights")


@pytest.mark.parametrize(
    ("previous_world", "end"),
    [
        # The successor serves without the predecessor's weights of the
        # changed files: they are loaded, not mapped.
        pytest.param("parked", "commit", marks=pytest.mark.missing_state),
        ("parked", "rollback"),
        ("weights", "rollback"),
        ("host", "rollback"),
    ],
)
def test_a_change_of_model_files_loads_the_successors_weights(
    tmp_path: Path, previous_world: str, end: str
):
    """A world change to other model files of the same model loads the
    successor's changed weights from them beside the serving world's and
    maps the tensors both checkpoints hold alike; the count streaming
    meanwhile continues on the state it reached. The successor answers with
    its weights. A commit leaves its weights alone on the GPU; a rollback
    answers as before on the previous world: parked, started again on the
    weights it kept on the GPU, or on the weights it kept in host storage,
    which return to the GPU first.
    """  # noqa: D205
    name = "qwen3-world-weights"
    updated = _updated_checkpoint()
    change = (
        f"model={updated}",
        f"update_config.previous_world={previous_world}",
    )
    with _serve(tmp_path, name, options=WEIGHTS_SHARE) as (base_url, _log):
        original = _chat(base_url, name, CACHED_PROMPT)
        (serving,) = [
            slot
            for slot in _config(base_url)["slots"]
            if slot["state"] != "empty"
        ]
        previous = serving["world"]
        assert _weights_of(base_url) == {(previous,)}
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)

        # The plan counts the successor's weights beside the serving
        # world's, and fits.
        storage = _planned_storage(base_url, *change)
        assert storage["feasible"] is True, storage
        for gpu in storage["gpus"]:
            assert gpu["successor"]["weights"] == gpu["serving"]["weights"], gpu
        if previous_world == "host":
            # The host admits the previous world's weights in host storage.
            for host in storage["hosts"]:
                assert host["previous_world_weights"] > 0, host
                assert host["feasible"] is True, host
        report = _world_change(base_url, *change)
        streaming.mark()
        world = report["world"]
        # The successor reads its weights from the other files, and the
        # requests in flight continue on the pages of the serving world's
        # numerics, which it maps.
        state = report["derived_state"]
        treatments = (
            state["weights"]["treatment"],
            state["kv_pages"]["treatment"],
        )
        assert treatments == ("recomputed", "continued"), report
        assert report["numerics"] == "changed", report
        assert report["previous_world_kept"] == previous_world, report
        print(
            f"{previous_world}: promotion pause "
            f"{report['pause_seconds']:.3f} s, "
            f"preparation {report['prepare_seconds']:.1f} s"
        )

        # The successor answers with its own weights. Both worlds hold the
        # tensors the checkpoints share; the previous world keeps its other
        # weights on the GPU, or in host storage, which leaves no weights of
        # the previous world alone in the GPU's ledger.
        changed = _chat(base_url, name, CACHED_PROMPT)
        assert changed != original, "the checkpoints answer alike"
        shared = {(previous, world)}
        if previous_world == "host":
            assert report["moved_to_host_bytes"] > 0, report
            assert _weights_of(base_url) - shared == {(world,)}
            (kept,) = [
                slot
                for slot in _config(base_url)["slots"]
                if slot["world"] == previous
            ]
            assert kept["state"] == "host", kept
        else:
            assert _weights_of(base_url) - shared == {(previous,), (world,)}
        streaming.wait_for(streaming.marked + 32)

        if end == "commit":
            committed = _admin(base_url, "config/commit")
            assert committed.status_code == 200, committed.text
            _wait_until(
                lambda: _weights_of(base_url) == {(world,)},
                60,
                "the previous world's weights outlived the commit",
            )
            streaming.join()
            # The committed world answers with the successor's weights. Its
            # answer during the trial shared its batch with the count, whose
            # greedy decoding may part from an answer computed alone.
            assert _chat(base_url, name, CACHED_PROMPT) != original
            return
        rolled = _admin(base_url, "config/rollback", {"reason": "the check"})
        assert rolled.status_code == 200, rolled.text
        result = rolled.json()["result"]
        assert result["world"] == previous, result
        print(
            f"{previous_world}: rollback pause {result['pause_seconds']:.3f} s"
        )
        streaming.join()
        assert _chat(base_url, name, CACHED_PROMPT) == original
        assert _weights_of(base_url) == {(previous,)}


# A prompt longer than one segment of the KV cache (an eighth of its 32768
# tokens), which leaves the next prompt's pages in later segments.
FILL_PROMPT = "Name one colour of the rainbow. " * 900
# Another such prompt, which shares no page with the first.
SECOND_FILL_PROMPT = "Name one planet of the solar system. " * 900

# The KV storage behind the host KV cache (`kv_cache_config.mooncake`): the
# Mooncake master this host runs (`host:port`) and the library its clients
# load (`libmooncake_store.so`).
MOONCAKE_MASTER = os.environ.get("UNISERVE_MOONCAKE_MASTER")
MOONCAKE_LIBRARY = os.environ.get("MOONCAKE_STORE_LIBRARY")


# A prompt of 17 pages of 64 tokens, which the fill prompts cached after it
# push out of a small host KV cache, and which a rollback prefetches from
# the KV storage whole.
LONG_CACHED_PROMPT = "Describe the sea in one sentence. " * 160


def _mooncake_options() -> tuple[str, ...]:
    """Mooncake behind the host KV cache. The deployment's Storage Server
    gives Mooncake a segment of 4 GiB, which holds every page.
    """  # noqa: D205
    return (
        "--set",
        f'kv_cache_config.mooncake.master="{MOONCAKE_MASTER}"',
        "--set",
        "kv_cache_config.mooncake.global_segment_gib=4",
        "--set",
        f'kv_cache_config.mooncake.library="{MOONCAKE_LIBRARY}"',
    )


def _no_pool_storage() -> tuple[str, ...]:
    """The options of a Storage Server whose host admits no page pool, its
    whole storage reserved, with Mooncake behind its pools: it stages the
    pages of each rank through a pool of `STAGING_POOL_BYTES` (256 MiB).
    """  # noqa: D205
    return (
        "--host-reserve-gib",
        str(1 << 20),
        "--mooncake-master",
        str(MOONCAKE_MASTER),
        "--mooncake-segment-bytes",
        str(4 << 30),
        "--mooncake-library",
        str(MOONCAKE_LIBRARY),
    )


# Without a Mooncake master and its library, the cases with a KV storage
# do not run.
WITHOUT_MOONCAKE = pytest.mark.skipif(
    not (MOONCAKE_MASTER and MOONCAKE_LIBRARY),
    reason="no Mooncake master (UNISERVE_MOONCAKE_MASTER) and library "
    "(MOONCAKE_STORE_LIBRARY)",
)


def _segments_of(base_url: str, world: int) -> int:
    """The KV cache segments `world` holds on the deployment's GPUs."""
    return sum(
        1
        for ledger in _ledgers(_config(base_url))
        for entry in ledger["entries"]
        if entry["item"] == "cache_segment" and world in entry["worlds"]
    )


@pytest.mark.parametrize(
    "host_cache",
    [
        "host",
        pytest.param("kv_storage", marks=WITHOUT_MOONCAKE),
        pytest.param("no_host_pool", marks=WITHOUT_MOONCAKE),
    ],
)
def test_a_parked_world_evicts_the_cache_its_successor_does_not_map(
    tmp_path: Path, host_cache: str
):
    """A change to other model files whose requests in flight compute their
    state again maps none of the serving world's KV cache. The previous
    world, parked, evicts every segment past its first once the idle pages
    there reached the host KV cache, and the successor's cache grows into
    their storage; the dry run states what that frees. A rollback creates
    the segments again and reads the evicted pages back from the host KV
    cache, answering as before.

    With `kv_storage`, the host KV cache of 1 GiB holds about 10900 tokens
    of Qwen3-30B-A3B at 96 KiB a token, fewer than the pages the change
    evicts, and Mooncake keeps what it commits: every evicted page passes
    through it into the KV storage, from which the rollback reads back what
    the host KV cache no longer holds. With `no_host_pool`, the host admits
    no page pool at all, and the pages pass through a staging pool into the
    KV storage.
    """  # noqa: D205
    name = "qwen3-world-evict"
    change = [
        f"model={_updated_checkpoint()}",
        "update_config.previous_world=parked",
    ]
    options, storage, cached = WEIGHTS_SHARE, None, CACHED_PROMPT
    if host_cache == "kv_storage":
        options += ("--set", "kv_cache_config.host_gib=1", *_mooncake_options())
        cached = LONG_CACHED_PROMPT
    if host_cache == "no_host_pool":
        options += _mooncake_options()
        storage, cached = _no_pool_storage(), LONG_CACHED_PROMPT
    with _serve(tmp_path, name, options=options, storage=storage) as (
        base_url,
        _log,
    ):
        # The prompt's pages lie past the first segment.
        _chat(base_url, name, FILL_PROMPT)
        original = _chat(base_url, name, cached)
        if host_cache != "host":
            _chat(base_url, name, SECOND_FILL_PROMPT)
        (serving,) = [
            slot
            for slot in _config(base_url)["slots"]
            if slot["state"] != "empty"
        ]
        previous = serving["world"]
        held = _segments_of(base_url, previous)
        assert held > 1

        planned = httpx.put(
            f"{base_url}/admin/config",
            json={"set": change, "in_flight": "recompute", "dry_run": True},
            timeout=1800,
        )
        assert planned.status_code == 200, planned.text
        plan = planned.json()["result"]["plan"]
        # Before the successor starts the plan states both outcomes: the
        # requests continue from warm copies under the same numerics, and
        # compute their state again under other numerics.
        assert plan["derived_state"]["kv_pages"] == {
            "treatment": "moved",
            "when_numerics_change": "recomputed",
        }, plan
        for gpu in plan["storage"]["gpus"]:
            steps = gpu["steps"]
            assert steps["trial"]["unmapped_kv_segments"] > 0, gpu
            assert steps["promotion_stall_ms"] is not None, gpu

        response = httpx.put(
            f"{base_url}/admin/config",
            json={"set": change, "in_flight": "recompute"},
            timeout=1800,
        )
        assert response.status_code == 200, response.text
        report = response.json()["result"]["world"]
        kv_pages = report["derived_state"]["kv_pages"]
        assert kv_pages["treatment"] == "recomputed", report
        # The preparation gave back the tail past what the successor's
        # weights left; the trial evicts what stayed past the first segment.
        kept = min(gpu["segments_kept"] for gpu in report["storage"]["gpus"])
        evicted = report["evicted"]
        assert evicted["evicted_segments"] == kept - 1, report
        assert evicted["evicted_bytes"] > 0, report
        world = report["world"]
        # The previous world keeps its first segment; the successor's cache
        # grew past its head segment into the storage freed.
        assert _segments_of(base_url, previous) == 1
        assert _segments_of(base_url, world) > 1
        assert _chat(base_url, name, cached) != original

        rolled = _admin(base_url, "config/rollback", {"reason": "the check"})
        assert rolled.status_code == 200, rolled.text
        assert rolled.json()["result"]["world"] == previous
        assert _segments_of(base_url, previous) == held
        answer, hits = _hits_of(base_url, name, cached)
        assert answer == original
        assert hits > 0, "the evicted pages were not read back"
