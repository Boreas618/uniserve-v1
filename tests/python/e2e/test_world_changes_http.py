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
- An operator's rollback resumes the previous world, parked or started
  again on the weights it kept (``update_config.previous_world``); the
  stream continues on it, and the configuration is marked unavailable:
  applying it again is refused unless forced.
- A fault of the world on trial rolls it back by itself: a worker process
  that exits, a job that never ends, an illegal memory access, non-finite
  logits, and the exit of its engine process. A plugin of the test's
  (``FAULT_PLUGIN``) arms each fault in the workers of the world the admin
  API reaches, the one on trial, so the production code has no test path.
  No request is lost or repeated: the stream continues on the previous
  world, which reads its KV cache as it left it and answers a cached prompt
  as before.

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
import tempfile
import threading
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    dedicated_experts,
    find_free_port,
    require_uniserve_binary,
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

# The plugin that arms faults in the workers of the world the admin API
# reaches. Before the KV cache is sized, and so before any graph is
# captured, it places a gate after the model's backbone: a kernel that
# spins while `spin` holds a nonzero word, then scales the hidden states by
# the word `offset` elements past `scale`. Unarmed it scales by one, which
# changes no value. Each hook arms one fault and waits for the device, so
# the next forward, eager or a graph's replay, meets it.
FAULT_PLUGIN = """
import os

import torch
import triton
import triton.language as tl

CONTROL = {}


@triton.jit
def _gate(spin, offset, scale, out):
    while tl.load(spin, volatile=True) != 0:
        pass
    tl.store(out, tl.load(scale + tl.load(offset)))


def register(registry):
    registry.buffer("faults", arm)
    registry.hook("fault_worker_exit", worker_exit)
    registry.hook("fault_stall", stall)
    registry.hook("fault_illegal_access", illegal_access)
    registry.hook("fault_non_finite", non_finite)


def arm(models, device):
    cuda = torch.device("cuda", device)
    CONTROL.update(
        spin=torch.zeros(1, dtype=torch.int32, device=cuda),
        offset=torch.zeros(1, dtype=torch.int64, device=cuda),
        scale=torch.ones(1, dtype=torch.float32, device=cuda),
        out=torch.ones(1, dtype=torch.float32, device=cuda),
    )
    for model in models:
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            backbone.register_forward_hook(_gated)
    return dict(CONTROL)


def _gated(module, args, output):
    if not (isinstance(output, torch.Tensor) and output.is_floating_point()):
        return output
    words = (CONTROL[key] for key in ("spin", "offset", "scale", "out"))
    _gate[(1,)](*words)
    return output * CONTROL["out"].to(output.dtype)


def _set(name, value):
    CONTROL[name].fill_(value)
    torch.cuda.synchronize()
    return {"armed": name, "pid": os.getpid()}


def worker_exit(payload, models):
    os._exit(17)


def stall(payload, models):
    return _set("spin", 1)


def illegal_access(payload, models):
    # An offset that moves the read to the first page of the address
    # space, which nothing maps.
    return _set("offset", -(CONTROL["scale"].data_ptr() // 4))


def non_finite(payload, models):
    return _set("scale", float("nan"))
"""


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
) -> Iterator[tuple[str, Path]]:
    """A deployment serving `name` on `devices`, its URL and its log."""
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

    def __init__(self, base_url: str, name: str):
        self.tokens: list[str] = []
        self.usage: dict | None = None
        self.marked: int | None = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._read, args=(base_url, name), daemon=True
        )
        self._thread.start()

    def _read(self, base_url: str, name: str) -> None:
        try:
            with httpx.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=_body(name, stream=True),
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

    def join(self) -> None:
        self._thread.join(timeout=1800)
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
        timeout=1800,
    )


def _admin(
    base_url: str, path: str, body: dict | None = None
) -> httpx.Response:
    return httpx.post(f"{base_url}/admin/{path}", json=body or {}, timeout=1800)


def _world_change(base_url: str, *assignments: str) -> dict:
    """Applies `assignments` as a world change; returns its report."""
    response = _apply(base_url, *assignments)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["change"] == "world", result
    return result["world"]


@pytest.mark.parametrize("topology", ["single", "experts", "dedicated"])
def test_a_world_change_continues_a_streaming_request_and_commits(
    tmp_path: Path, topology: str
):
    devices, options = _topology(topology, tmp_path)
    name = "qwen3-world-change"
    with _serve(tmp_path, name, devices, options) as (base_url, log):
        before = _config(base_url)
        (serving,) = [slot for slot in before["slots"] if slot]
        assert serving["state"] == "executing", before
        engines_before = _pids(log, "Engine")

        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        report = _world_change(base_url, CHANGE)
        streaming.mark()
        world = report["world"]
        assert report["previous_world"] == serving["world"], report
        print(
            f"promotion pause: {report['pause_seconds']:.3f} s "
            f"(park {report['park_seconds']:.3f} s, "
            f"preparation {report['prepare_seconds']:.1f} s)"
        )

        # The successor serves on trial beside the parked previous world, and
        # the trial refuses another change.
        status = _config(base_url)
        assert status["trial"]["world"] == world, status
        assert status["slots"] == [
            {"world": world, "state": "executing"},
            {"world": serving["world"], "state": "parked"},
        ], status
        refused = _apply(base_url, "max_running_requests=4")
        assert refused.status_code == 409, refused.text

        # The operator commits while the count streams on the successor,
        # which writes new pages only to its own segments until then.
        streaming.wait_for(streaming.marked + 64)
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        assert committed.json()["result"]["world"] == world
        status = _config(base_url)
        assert status["trial"] is None, status
        assert status["slots"] == [
            {"world": world, "state": "executing"},
            None,
        ], status
        graphs = tomllib.loads(status["config"])["cuda_graph_config"]
        assert graphs["decode_batch_sizes"] == [1, 2, 4, 8, 16], graphs
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


@pytest.mark.parametrize("previous_world", ["parked", "weights"])
def test_an_operator_rollback_resumes_the_previous_world(
    tmp_path: Path, previous_world: str
):
    name = "qwen3-world-rollback"
    with _serve(tmp_path, name) as (base_url, log):
        cached = _chat(base_url, name, CACHED_PROMPT)
        (serving,) = [slot for slot in _config(base_url)["slots"] if slot]
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


# The plugin of a release that revised what the normalization computes: the
# next world declares another revision of it, while the arithmetic stays as
# it was, so the requests a world change moves compute the same tokens
# however they continue.
REVISION_PLUGIN = """
from uniserve.nn.norm import RMSNorm


def register(registry):
    RMSNorm.function = "nn.norm.RMSNorm/2"
"""


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


@pytest.mark.parametrize(
    ("in_flight", "previous_world"),
    [("keep", "parked"), ("keep", "weights"), ("recompute", "parked")],
)
def test_a_world_of_other_numerics_continues_the_requests_in_flight(
    tmp_path: Path, in_flight: str, previous_world: str
):
    """A request kept continues on the pages of the state it reached, the
    previous world kept parked or as its weights. A request recomputed
    computes them again in the successor's own segments, which hold its
    head alone until the commit gives it the previous world's: a change
    whose requests in flight need more is refused, and the serving world
    serves on.
    """  # noqa: D205
    name = "qwen3-world-numerics"
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    environment = {
        "PYTHONPATH": os.pathsep.join(
            [str(plugins), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
    }
    with _serve(tmp_path, name, env=environment) as (base_url, log):
        cached = _chat(base_url, name, CACHED_PROMPT)
        (serving,) = [slot for slot in _config(base_url)["slots"] if slot]
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)

        # The next world's processes import the release; the serving
        # world's do not.
        package = plugins / "uniserve_revision_plugin"
        package.mkdir()
        (package / "__init__.py").write_text(REVISION_PLUGIN)
        metadata = plugins / "uniserve_revision_plugin-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: uniserve-revision-plugin\n"
            "Version: 1.0\n"
        )
        (metadata / "entry_points.txt").write_text(
            "[uniserve.plugins]\nrevision = uniserve_revision_plugin:register\n"
        )
        hits = _prefix_hits(base_url)
        response = httpx.put(
            f"{base_url}/admin/config",
            json={
                "set": [
                    CHANGE,
                    f"update_config.previous_world={previous_world}",
                ],
                "in_flight": in_flight,
            },
            timeout=1800,
        )
        if (in_flight, previous_world) == ("recompute", "parked"):
            assert response.status_code == 500, response.text
            assert "free units" in response.text, response.text
            assert _config(base_url)["slots"][0] == serving
            streaming.join()
            reference, reference_usage = _ask(base_url, name)
            assert streaming.tokens == reference
            assert streaming.usage == reference_usage
            return
        assert response.status_code == 200, response.text
        report = response.json()["result"]["world"]
        assert report["numerics"] == "changed", report
        assert report["in_flight"] == in_flight, report
        streaming.wait_for(len(streaming.tokens) + 64)
        assert _admin(base_url, "config/commit").status_code == 200
        streaming.join()

        migrated = _prefix_hits(base_url) - hits
        print(f"{in_flight}: the migrated request hit {migrated:.0f} tokens")
        if in_flight == "keep":
            assert migrated >= EARLY_TOKENS, migrated
        else:
            assert migrated == 0, migrated

        # A request after the change hits no page of the earlier numerics.
        answer, hits = _hits_of(base_url, name, CACHED_PROMPT)
        assert answer == cached
        assert hits == 0

        reference, reference_usage = _ask(base_url, name)
        assert streaming.tokens == reference
        assert streaming.usage == reference_usage


# The faults the world on trial meets, by the hook that arms each, and the
# options the successor serves with for it: a stalled job is found after
# half the watchdog's timeout.
FAULTS = {
    "worker_exit": (),
    "stall": ('worker_config.watchdog_timeout="20s"',),
    "illegal_access": (),
    "non_finite": (),
    "engine_exit": (),
}


@pytest.mark.parametrize("previous_world", ["parked", "weights"])
@pytest.mark.parametrize("fault", list(FAULTS))
def test_a_fault_of_the_world_on_trial_rolls_it_back(
    tmp_path: Path, fault: str, previous_world: str
):
    name = "qwen3-world-fault"
    plugins = tmp_path / "plugins"
    package = plugins / "uniserve_fault_plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(FAULT_PLUGIN)
    metadata = plugins / "uniserve_fault_plugin-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: uniserve-fault-plugin\nVersion: 1.0\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[uniserve.plugins]\nfaults = uniserve_fault_plugin:register\n"
    )
    environment = {
        "PYTHONPATH": os.pathsep.join(
            [str(plugins), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
    }
    with _serve(tmp_path, name, env=environment) as (base_url, log):
        cached = _chat(base_url, name, CACHED_PROMPT)
        (serving,) = [slot for slot in _config(base_url)["slots"] if slot]
        engines_before = _pids(log, "Engine")

        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        report = _world_change(
            base_url,
            CHANGE,
            f"update_config.previous_world={previous_world}",
            *FAULTS[fault],
        )
        world = report["world"]
        # The successor generates before the fault.
        streaming.wait_for(len(streaming.tokens) + 16)

        if fault == "engine_exit":
            (engine,) = _pids(log, "Engine") - engines_before
            os.kill(engine, signal.SIGKILL)
        else:
            armed = _admin(base_url, f"hooks/fault_{fault}")
            if fault != "worker_exit":
                assert armed.status_code == 200, armed.text
        streaming.mark()

        def rolled_back() -> bool:
            last = _config(base_url)["last"] or {}
            return last.get("rolled_back") == world

        _wait_until(rolled_back, 600, "the world on trial was not rolled back")
        status = _config(base_url)
        print(
            f"{fault} on trial, {previous_world}: promotion pause "
            f"{report['pause_seconds']:.3f} s, rollback pause "
            f"{status['last']['pause_seconds']:.3f} s; "
            f"{status['last']['reason']}"
        )
        assert status["trial"] is None, status
        assert status["slots"][0] == {
            "world": serving["world"],
            "state": "executing",
        }, status
        (unavailable,) = status["unavailable"]
        assert unavailable["world"] == world, unavailable
        # The configuration is marked with the fault's first error.
        cause = {
            "non_finite": "non-finite logits",
            "illegal_access": "illegal memory access",
            "stall": "stalled",
        }.get(fault, f"world {world}")
        assert cause in unavailable["reason"], unavailable

        # No request is lost or repeated, and the previous world reads its
        # KV cache as it left it.
        streaming.join()
        assert len(streaming.tokens) > streaming.marked
        reference, reference_usage = _ask(base_url, name)
        assert streaming.tokens == reference
        assert streaming.usage == reference_usage
        assert _chat(base_url, name, CACHED_PROMPT) == cached

        again = _apply(
            base_url,
            CHANGE,
            f"update_config.previous_world={previous_world}",
            *FAULTS[fault],
        )
        assert again.status_code == 400, again.text
        assert "rolled back" in again.text, again.text

    lines = _log_lines(log)
    assert any("rolling back" in line for line in lines)
    assert any("the previous world serves again" in line for line in lines)
