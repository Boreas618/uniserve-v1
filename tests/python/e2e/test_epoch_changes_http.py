"""An epoch change continues the requests in flight on a successor epoch.

The successor serves on trial until the operator commits it or rolls it
back.

Qwen3-MoE (``UNISERVE_QWEN3_MODEL``) streams a long greedy count. While it
streams, ``PUT /admin/config`` changes the decode batch sizes the CUDA graphs
capture, which only a new epoch executes: a successor epoch starts beside
the serving one on the same GPUs and maps its weights and KV cache segments
read-only; at a step boundary the serving epoch parks, its streaming
request leaving it as migratable after every token it generated; the
successor serves on trial, and the daemon submits the request again with
those tokens. The client reads one uninterrupted stream, equal to the count
the committed epoch decodes for the same request alone, with the same usage.

- A commit releases the previous epoch: its engine's process exits and the
  quotas reserve the committed epoch alone. The trial refuses another change
  meanwhile. The deployment runs on one GPU, as two schedulers sharing their
  experts (``--data-parallel-size 2`` with expert parallelism), or as
  attention and expert workers exchanging through DeepEP.
- A change that keeps the functions and formats invalidates nothing: the
  successor maps every retained weight, and after the commit a cached prompt
  finds as many tokens cached as before the change, on the device and, once
  fill prompts pushed its pages out of it, in the host KV cache.
- A change of the model's files loads the successor's weights beside the
  serving epoch's; a commit leaves them alone on the GPU, and a rollback
  resumes the previous epoch on the weights it kept
  (``update_config.previous_epoch``).
- A parked previous epoch whose cache the successor does not map evicts
  every segment past its first, and a rollback reads the evicted pages back.

What the tests check is what the admin API reports: ``PUT /admin/config``
answers the change it started, ``GET /admin/config`` the epoch that executes
(``executing``), the update with its phase (``update``), the node's main
processes (``processes``) and every GPU's quota (``quotas``), and
``/metrics`` the prefix cache's counters. The tests print the pause of every
promotion. Each runs a Storage Server of its own: ``XDG_RUNTIME_DIR`` points
the daemon at a directory of the test's, and the server the daemon starts
there is stopped at the end.
"""

from __future__ import annotations

import json
import os
import subprocess
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
    ATTENTION_BACKEND,
    QWEN3_UPDATED_CHECKPOINT,
    admin,
    admin_config,
    dedicated_experts,
    engine_processes,
    find_free_port,
    granted_devices,
    host_asset,
    log_tail,
    metric,
    qwen3_checkpoint,
    require_uniserve_binary,
    runs,
    runtime_directory,
    scaled,
    server_process,
    wait_until,
    engine_selection,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

MAX_TOKENS = 16384
# Tokens the client reads before it asks for the change.
EARLY_TOKENS = 64
PROMPT = (
    "Count from 1 to 5000, separated by commas and spaces. "
    "Write only the numbers."
)
# A prompt of several pages, whose pages the serving epoch publishes.
CACHED_PROMPT = "Describe the sea in one sentence. " * 40
# Options every deployment of the tests serves with: few captured graphs,
# so that epochs start while the count streams, and a small KV cache.
OPTIONS = (
    "dtype=bfloat16",
    f"attention_backend={ATTENTION_BACKEND}",
    "kv_cache_config.max_tokens=32768",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
)
# The change every test applies: graphs of other decode batch sizes, which
# a new epoch captures.
CHANGE = "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8, 16]"
# How long a successor takes to reach its trial, and a commit or a rollback
# to end, at most.
UPDATE_TIMEOUT_S = 900


def _topology(topology: str, tmp_path: Path) -> tuple[str, tuple[str, ...]]:
    """The devices and extra options of a deployment.

    ``single``: one GPU. ``experts``: two schedulers sharing their experts
    over the all-to-all. ``dedicated``: attention and expert workers
    exchanging through DeepEP.
    """
    if topology == "single":
        return granted_devices(1), ()
    if topology == "experts":
        return granted_devices(2), (
            "--data-parallel-size",
            "2",
            "--set",
            "moe_config.expert_parallel=true",
        )
    return granted_devices(2), (
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
    storage: tuple[str, ...] | None = None,
) -> Iterator[str]:
    """A deployment serving `name` on `devices`; its URL.

    With `storage`, a Storage Server started by hand with those options
    serves the deployment, which starts none of its own.
    """
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        qwen3_checkpoint(),
        "--served-model-name",
        name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *engine_selection(),
    ]
    for option in OPTIONS:
        command += ["--set", option]
    command += options
    with runtime_directory(tmp_path / "storage-server.log") as runtime:
        environment = {
            "CUDA_VISIBLE_DEVICES": devices or granted_devices(1),
            "XDG_RUNTIME_DIR": str(runtime),
        }
        if storage is not None:
            socket = runtime / "uniserve" / "storage-server.sock"
            socket.parent.mkdir(mode=0o700)
            subprocess.run(
                [
                    str(require_uniserve_binary()),
                    "storage-server",
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
            command,
            base_url,
            tmp_path / "server.log",
            timeout_s=900,
            env=environment,
        ):
            yield base_url


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
            target=self._read,
            args=(base_url, _body(name, stream=True)),
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
        wait_until(
            lambda: len(self.tokens) >= count or not self._thread.is_alive(),
            600,
            "the count did not stream",
            interval_s=0.5,
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


def _apply(base_url: str, *assignments: str, **body: object) -> httpx.Response:
    """``PUT /admin/config`` with `assignments` and the other fields of
    `body` (``force``, ``in_flight``, ``dry_run``).
    """  # noqa: D205
    return httpx.put(
        f"{base_url}/admin/config",
        json={"set": list(assignments), **body},
        timeout=scaled(1800),
    )


def _executing(base_url: str) -> int:
    """The epoch that executes, which ``GET /admin/config`` reports."""
    executing = admin_config(base_url)["executing"]
    assert executing["state"] == "executing", executing
    return executing["epoch"]


def _update(base_url: str) -> dict:
    """The update in flight, or the one that ended last."""
    return admin_config(base_url).get("update") or {}


def _phase(base_url: str, phase: str, what: str) -> dict:
    """Waits until the update reaches `phase`; returns it."""
    wait_until(
        lambda: _update(base_url).get("phase") == phase,
        UPDATE_TIMEOUT_S,
        what,
        context=lambda: f"\nupdate: {_update(base_url)}",
    )
    return _update(base_url)


def _epoch_change(
    base_url: str, *assignments: str, **body: object
) -> tuple[dict, int]:
    """Applies `assignments` as an epoch change and waits for its trial;
    returns the update and the epoch on trial.
    """  # noqa: D205
    response = _apply(base_url, *assignments, **body)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["change"] == "epoch", result
    update = _phase(base_url, "probation", "the update did not reach its trial")
    print(f"promotion pause: {update['pause_ms']} ms")
    return update, _executing(base_url)


def _commit(base_url: str) -> None:
    """Commits the trial and waits until it ended the previous epoch."""
    committed = admin(base_url, "config/commit")
    assert committed.status_code == 200, committed.text
    _phase(base_url, "purged", "the commit did not end the previous epoch")


def _rollback(base_url: str, reason: str) -> dict:
    """Rolls the trial back and waits until it ended; returns the update."""
    rolled = admin(base_url, "config/rollback", {"reason": reason})
    assert rolled.status_code == 200, rolled.text
    update = _phase(base_url, "abandoned", "the rollback did not end")
    assert update["failure"]["reason"] == reason, update
    return update


def _change_under_traffic(
    base_url: str, name: str, streaming: Streaming, assignments: tuple[str, ...]
) -> tuple[int, list[Streaming]]:
    """Keep one request running throughout preparation, retaining every
    result; returns the epoch on trial and the streams.

    A finite request can finish while a successor warms. Start the next with
    the same prompt and token limit so the test observes migration of a live
    request regardless of the number of requests preparation spans.
    """  # noqa: D205
    streams = [streaming]
    with ThreadPoolExecutor(max_workers=1) as executor:
        change = executor.submit(_epoch_change, base_url, *assignments)
        while not change.done():
            if streams[-1].finished:
                streams[-1].join()
                streams.append(Streaming(base_url, name))
            time.sleep(0.01)
        _, epoch = change.result()
    streams[-1].mark()
    return epoch, streams


def _planned(base_url: str, *assignments: str, **body: object) -> dict:
    """The plan of a dry run of `assignments`, which changes nothing: the
    epoch it would switch to, its executable, generation and nodes.
    """  # noqa: D205
    serving = _executing(base_url)
    response = _apply(base_url, *assignments, dry_run=True, **body)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["change"] == "planned", result
    plan = result["plan"]
    assert plan["change"] == "epoch", plan
    assert plan["epoch"] != serving, plan
    assert _executing(base_url) == serving
    return plan


def _quotas(status: dict) -> list[dict]:
    """Every GPU's quota ``GET /admin/config`` reports, each of
    whose entries' bytes sum to no more than its quota.
    """  # noqa: D205
    nodes = status["quotas"]
    assert isinstance(nodes, list) and nodes, status
    quotas = [quota for node in nodes for quota in node["quotas"]]
    assert quotas, status
    for quota in quotas:
        used = sum(entry["bytes"] for entry in quota["reservations"])
        assert quota["limit"] is not None and used <= quota["limit"], quota
    return quotas


def _epochs_of(quotas: list[dict], item: str) -> set[tuple[int, ...]]:
    """The sets of epochs holding the quotas' entries of `item`."""
    return {
        tuple(entry["epochs"])
        for quota in quotas
        for entry in quota["reservations"]
        if entry["item"] == item
    }


def _engine_pids(base_url: str) -> set[int]:
    """The process ids of the deployment's main processes."""
    return {process["pid"] for process in engine_processes(base_url)}


@pytest.mark.parametrize(
    "topology",
    [
        "single",
        pytest.param("experts", marks=pytest.mark.gpus(2)),
        pytest.param("dedicated", marks=pytest.mark.gpus(2)),
    ],
)
def test_a_epoch_change_continues_a_streaming_request_and_commits(
    tmp_path: Path, topology: str
):
    devices, options = _topology(topology, tmp_path)
    name = "qwen3-epoch-change"
    with _serve(tmp_path, name, devices, options) as base_url:
        serving = _executing(base_url)
        engines_before = _engine_pids(base_url)
        # Every GPU's quota reserves the serving epoch alone, within its
        # quota: its weights, KV cache segments, graph pool and process.
        quotas = _quotas(admin_config(base_url))
        for item in ("weights", "cache_segment", "graph_pool", "process"):
            assert _epochs_of(quotas, item) == {(serving,)}, quotas

        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        wait_until(
            lambda: _prefix_counter(base_url, "queries") > 0,
            30,
            "the original request's cache query was not reported",
        )
        queries_before = _prefix_counter(base_url, "queries")
        hits_before = _prefix_hits(base_url)
        # A dry run plans the change and changes nothing.
        plan = _planned(base_url, CHANGE)
        print(f"planned: {json.dumps(plan)}")
        _, epoch = _epoch_change(base_url, CHANGE)
        streaming.mark()
        assert epoch != serving

        # The successor serves on trial beside the parked previous epoch,
        # whose main process still runs, and the trial refuses another
        # change. Both epochs share each GPU's quota: the weights and the
        # segments the successor maps are reserved once, held by both, beside
        # each epoch's graph pool and process.
        assert all(runs(pid) for pid in engines_before), engines_before
        quotas = _quotas(admin_config(base_url))
        both = (serving, epoch)
        assert _epochs_of(quotas, "weights") == {both}, quotas
        assert both in _epochs_of(quotas, "cache_segment"), quotas
        assert _epochs_of(quotas, "process") == {(serving,), (epoch,)}
        refused = _apply(base_url, "max_running_requests=4")
        assert refused.status_code == 409, refused.text

        # The operator commits while the count streams on the successor,
        # which writes new pages only to its own segments until then.
        streaming.wait_for(streaming.marked + 64)
        wait_until(
            lambda: _prefix_counter(base_url, "queries") > queries_before,
            30,
            "the resumed request's cache query was not reported",
        )
        # The successor found every token the previous epoch computed of the
        # request cached, including the final partial page. Admission leaves
        # only the last emitted token to run.
        reused = _prefix_hits(base_url) - hits_before
        queried = _prefix_counter(base_url, "queries") - queries_before
        assert reused == queried > 0, (reused, queried)
        _commit(base_url)
        status = admin_config(base_url)
        assert status["executing"]["epoch"] == epoch, status["executing"]
        graphs = tomllib.loads(status["config"])["cuda_graph_config"]
        assert graphs["decode_batch_sizes"] == [1, 2, 4, 8, 16], graphs
        # The commit released everything only the previous epoch held: the
        # segments the successor mapped are its own on the quota now. The
        # previous epoch's workers end their reservations as their processes
        # exit, a moment after its engine's.
        items = ("weights", "cache_segment", "graph_pool", "process")
        wait_until(
            lambda: all(
                _epochs_of(_quotas(admin_config(base_url)), item) == {(epoch,)}
                for item in items
            ),
            30,
            "the quotas did not reserve the committed epoch alone",
        )
        # The previous epoch's engine exited at the commit.
        for pid in engines_before:
            wait_until(
                lambda pid=pid: not runs(pid),
                120,
                f"engine {pid} of the previous epoch outlived the commit",
            )
        streaming.join()
        reference, reference_usage = _ask(base_url, name)

    assert len(streaming.tokens) > streaming.marked, (
        "the stream ended before the successor served"
    )
    assert streaming.tokens == reference
    assert streaming.usage == reference_usage


def _forwards(base_url: str, mode: str) -> float:
    """The forwards every epoch ran in ``mode`` so far (``eager`` or
    ``graph_replay``; ``uniserve:worker_forward_mode_counts``), once the
    reports of what came before arrived.
    """  # noqa: D205
    time.sleep(scaled(REPORT_SETTLE))
    return metric(
        base_url, "uniserve:worker_forward_mode_counts", f'mode="{mode}"'
    )


@pytest.mark.parametrize(
    "topology",
    [
        "single",
        pytest.param("experts", marks=pytest.mark.gpus(2)),
        pytest.param("dedicated", marks=pytest.mark.gpus(2)),
    ],
)
def test_a_serving_epoch_gives_its_graph_storage_to_a_preparing_successor(
    tmp_path: Path, topology: str
):
    devices, options = _topology(topology, tmp_path)
    name = "qwen3-epoch-eager"
    with _serve(tmp_path, name, devices, options) as base_url:
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        eager = _forwards(base_url, "eager")

        # The serving epoch decodes the count eagerly while the successor
        # prepares in its graph storage.
        _epoch_change(base_url, CHANGE, "update_config.eager_prepare=true")
        assert _forwards(base_url, "eager") > eager

        # Resumed, the previous epoch replays its graphs again.
        _rollback(base_url, "the operator's check")
        replays = _forwards(base_url, "graph_replay")
        eager = _forwards(base_url, "eager")
        streaming.wait_for(len(streaming.tokens) + 64)
        assert _forwards(base_url, "graph_replay") > replays
        assert _forwards(base_url, "eager") == eager
        streaming.join()


def test_a_successor_continues_a_request_from_its_predecessors_device_pages(
    tmp_path: Path,
):
    """The successor of an epoch change reads the streaming request's pages
    where the serving epoch left them: with no host KV cache to warm through,
    every token the request computed is a device prefix cache hit on its
    first forward on the successor, and the stream is the count alone.
    """  # noqa: D205
    name = "qwen3-device-warming"
    options = ("--set", "kv_cache_config.host_tier=false")
    with _serve(tmp_path, name, options=options) as base_url:
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        wait_until(
            lambda: _prefix_counter(base_url, "queries") > 0,
            30,
            "the original request's cache query was not reported",
        )
        queries_before = _prefix_counter(base_url, "queries")
        hits_before = _prefix_hits(base_url)
        host_before = _prefix_counter(base_url, "host_hits")

        _epoch_change(base_url, CHANGE)
        streaming.mark()
        streaming.wait_for(streaming.marked + 64)
        wait_until(
            lambda: _prefix_counter(base_url, "queries") > queries_before,
            30,
            "the resumed request's cache query was not reported",
        )
        reused = _prefix_hits(base_url) - hits_before
        queried = _prefix_counter(base_url, "queries") - queries_before
        assert reused == queried > 0, (reused, queried)
        assert _prefix_counter(base_url, "host_hits") == host_before
        streaming.join()
        reference, reference_usage = _ask(base_url, name)

    assert len(streaming.tokens) > streaming.marked, (
        "the stream ended before the successor served"
    )
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
    """Quotas, queue depth and cache capacity change through an epoch."""
    name = "qwen3-resource-change"
    assignments = (
        "mem_fraction_static=0.8",
        "device_config.queue_depth=3",
        "kv_cache_config.max_tokens=32768",
        "kv_cache_config.host_gib=8",
    )
    # The cache also holds graph scratch units. Both capacities must fit
    # the full count beside those units throughout successor preparation.
    with _serve(
        tmp_path,
        name,
        options=("--set", "kv_cache_config.max_tokens=65536"),
    ) as base_url:
        before = _executing(base_url)
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        epoch, streams = _change_under_traffic(
            base_url, name, streaming, assignments
        )
        assert epoch != before
        _commit(base_url)
        configured = tomllib.loads(admin_config(base_url)["config"])
        assert configured["mem_fraction_static"] == 0.8
        assert configured["device_config"]["queue_depth"] == 3
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
    with _serve(tmp_path, name) as base_url:
        serving = _executing(base_url)
        # The prompt's pages are computed once and then found cached, on the
        # device and, once pushed out of it, in the host KV cache.
        answer = _chat(base_url, name, CACHED_PROMPT)
        cached_answer, hits = _hits_of(base_url, name, CACHED_PROMPT)
        assert cached_answer == answer and hits > 0, (cached_answer, hits)
        host_before = _host_hits_of(base_url, name, CACHED_PROMPT, 0)
        assert host_before[1] > 0, host_before

        # Graph capture changes none of the weights or page keys: the
        # successor maps every weight the serving epoch holds.
        _, epoch = _epoch_change(base_url, CHANGE)
        weights = _epochs_of(_quotas(admin_config(base_url)), "weights")
        assert weights == {(serving, epoch)}, weights
        _commit(base_url)

        # The committed epoch finds as many of the prompt's tokens cached,
        # on the device and in the host KV cache, and answers it as before.
        assert _hits_of(base_url, name, CACHED_PROMPT) == (answer, hits)
        host_after = _host_hits_of(
            base_url, name, CACHED_PROMPT, EVICTING_FILLS
        )
        assert host_after == host_before, (host_after, host_before)


# How long the counters an engine reports every second take to include
# what a request it answered counted: two report intervals.
REPORT_SETTLE = 2.0


def _prefix_counter(base_url: str, name: str) -> float:
    """A prefix cache counter every epoch counted so far, in tokens
    (``uniserve:prefix_cache_<name>``).
    """  # noqa: D205
    return metric(base_url, f"uniserve:prefix_cache_{name}")


def _prefix_hits(base_url: str) -> float:
    """The prefix cache hits every epoch counted so far."""
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
    wait_until(
        lambda: _prefix_counter(base_url, "queries") > queries,
        30,
        "the request's prefix cache query was not reported",
    )
    time.sleep(scaled(REPORT_SETTLE))
    return answer, _prefix_hits(base_url) - hits


def _updated_checkpoint() -> str:
    """Another checkpoint of the model, of the same files but for other
    weights (``UNISERVE_QWEN3_UPDATED_MODEL``).
    """  # noqa: D205
    return str(
        host_asset("UNISERVE_QWEN3_UPDATED_MODEL", QWEN3_UPDATED_CHECKPOINT)
    )


# The share of the GPU a deployment whose successor loads every weight
# again serves with: both epochs' weights, the successor's graph share and
# both KV caches fit in it.
WEIGHTS_SHARE = ("--set", "mem_fraction_static=0.9")


def _weights_of(base_url: str) -> set[tuple[int, ...]]:
    """The sets of epochs holding weights on the deployment's GPUs."""
    return _epochs_of(_quotas(admin_config(base_url)), "weights")


@pytest.mark.parametrize(
    ("previous_epoch", "end"),
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
    tmp_path: Path, previous_epoch: str, end: str
):
    """An epoch change to other model files of the same model loads the
    successor's changed weights from them beside the serving epoch's and
    maps the tensors both checkpoints hold alike; the count streaming
    meanwhile continues on the state it reached. The successor answers with
    its weights. A commit leaves its weights alone on the GPU; a rollback
    answers as before on the previous epoch: parked, started again on the
    weights it kept on the GPU, or on the weights it kept in host storage,
    which return to the GPU first.
    """  # noqa: D205
    name = "qwen3-epoch-weights"
    updated = _updated_checkpoint()
    change = (
        f"model={updated}",
        f"update_config.previous_epoch={previous_epoch}",
    )
    with _serve(tmp_path, name, options=WEIGHTS_SHARE) as base_url:
        original = _chat(base_url, name, CACHED_PROMPT)
        previous = _executing(base_url)
        assert _weights_of(base_url) == {(previous,)}
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)

        update, epoch = _epoch_change(base_url, *change)
        streaming.mark()
        if previous_epoch == "host":
            # The previous epoch's weights the successor does not map leave
            # its GPU for host storage while the successor serves.
            wait_until(
                lambda: (
                    _update(base_url)["moved_to_host_bytes"] > 0
                    or _update(base_url)["host_refused"]
                ),
                300,
                "the previous epoch's weights did not move",
            )
            update = _update(base_url)
            assert not update["host_refused"], update
        assert update["previous_epoch_kept"] == previous_epoch, update

        # The successor answers with its own weights. Both epochs hold the
        # tensors the checkpoints share; the previous epoch keeps its other
        # weights on the GPU, or in host storage, which leaves no weights of
        # the previous epoch alone in the GPU's quota.
        changed = _chat(base_url, name, CACHED_PROMPT)
        assert changed != original, "the checkpoints answer alike"
        shared = {(previous, epoch)}
        if previous_epoch == "host":
            assert _weights_of(base_url) - shared == {(epoch,)}
        else:
            assert _weights_of(base_url) - shared == {(previous,), (epoch,)}
        streaming.wait_for(streaming.marked + 32)

        if end == "commit":
            _commit(base_url)
            wait_until(
                lambda: _weights_of(base_url) == {(epoch,)},
                60,
                "the previous epoch's weights outlived the commit",
            )
            streaming.join()
            # The committed epoch answers with the successor's weights. Its
            # answer during the trial shared its batch with the count, whose
            # greedy decoding may part from an answer computed alone.
            assert _chat(base_url, name, CACHED_PROMPT) != original
            return
        _rollback(base_url, "the check")
        assert _executing(base_url) == previous
        streaming.join()
        assert _chat(base_url, name, CACHED_PROMPT) == original
        wait_until(
            lambda: _weights_of(base_url) == {(previous,)},
            60,
            "the successor's weights outlived the rollback",
        )


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


def _segments_of(base_url: str, epoch: int) -> int:
    """The KV cache segments `epoch` holds on the deployment's GPUs."""
    return sum(
        1
        for quota in _quotas(admin_config(base_url))
        for entry in quota["reservations"]
        if entry["item"] == "cache_segment" and epoch in entry["epochs"]
    )


@pytest.mark.parametrize(
    "host_cache",
    [
        "host",
        pytest.param("kv_storage", marks=WITHOUT_MOONCAKE),
        pytest.param("no_host_pool", marks=WITHOUT_MOONCAKE),
    ],
)
def test_a_parked_epoch_evicts_the_cache_its_successor_does_not_map(
    tmp_path: Path, host_cache: str
):
    """A change to other model files whose requests in flight compute their
    state again maps none of the serving epoch's KV cache. The previous
    epoch, parked, evicts every segment past its first once the idle pages
    there reached the host KV cache, and the successor's cache grows into
    their storage. A rollback creates the segments again and reads the
    evicted pages back from the host KV cache, answering as before.

    With `kv_storage`, the host KV cache of 1 GiB holds about 10900 tokens
    of Qwen3-30B-A3B at 96 KiB a token, fewer than the pages the change
    evicts, and Mooncake keeps what it commits: every evicted page passes
    through it into the KV storage, from which the rollback reads back what
    the host KV cache no longer holds. With `no_host_pool`, the host admits
    no page pool at all, and the pages pass through a staging pool into the
    KV storage.
    """  # noqa: D205
    name = "qwen3-epoch-evict"
    change = (
        f"model={_updated_checkpoint()}",
        "update_config.previous_epoch=parked",
    )
    options, storage, cached = WEIGHTS_SHARE, None, CACHED_PROMPT
    if host_cache == "kv_storage":
        options += ("--set", "kv_cache_config.host_gib=1", *_mooncake_options())
        cached = LONG_CACHED_PROMPT
    if host_cache == "no_host_pool":
        options += _mooncake_options()
        storage, cached = _no_pool_storage(), LONG_CACHED_PROMPT
    with _serve(tmp_path, name, options=options, storage=storage) as base_url:
        # The prompt's pages lie past the first segment.
        _chat(base_url, name, FILL_PROMPT)
        original = _chat(base_url, name, cached)
        if host_cache != "host":
            _chat(base_url, name, SECOND_FILL_PROMPT)
        previous = _executing(base_url)
        held = _segments_of(base_url, previous)
        assert held > 1

        # A dry run plans the change and changes nothing.
        _planned(base_url, *change, in_flight="recompute")
        _, epoch = _epoch_change(base_url, *change, in_flight="recompute")
        # The previous epoch keeps its first segment once the pages past it
        # reached the host KV cache; the successor's cache grew past its
        # head segment into the storage freed.
        wait_until(
            lambda: _segments_of(base_url, previous) == 1,
            300,
            "the parked epoch did not evict its segments past the first",
            context=lambda: log_tail(tmp_path / "server.log"),
        )
        assert _segments_of(base_url, epoch) > 1
        assert _chat(base_url, name, cached) != original

        _rollback(base_url, "the check")
        assert _executing(base_url) == previous
        assert _segments_of(base_url, previous) == held
        answer, hits = _hits_of(base_url, name, cached)
        assert answer == original
        assert hits > 0, "the evicted pages were not read back"
