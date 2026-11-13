"""An epoch change moves the requests in flight to a target epoch.

The target serves on trial until the operator commits it or rolls it back.

Qwen3-MoE (``UNISERVE_QWEN3_MODEL``) streams a long greedy count. While it
streams, ``PUT /admin/config`` changes what only a new epoch executes: the
target's engines start beside the serving ones on the same GPUs, open their
sessions with the host's buffer pool, map the weights they share and park.
The switch then lets both epochs step (``specs/updates.md`` §3.7): the target
is promoted and admits every new request, and the serving engines drain beside
it, stepping their running requests until each ends or the drain's deadline
(``update_config.drain_s``) ends it as migratable, its blocks committed under
their keys, the partial last one too. The daemon submits it again to the
target with the tokens it generated. The client reads one uninterrupted
stream, equal to the count the committed epoch decodes for the same request
alone, with the same usage.

- A change of the engine alone keeps the layout: the two epochs name the
  same regions, so every block the serving epoch committed is a hit of the
  target at once, on the device; the request that migrates pays one admission
  and the clone of its partial last block. A commit releases the previous
  epoch: its engine's process exits and the quotas reserve the committed
  epoch alone. The trial refuses another change meanwhile. The deployment
  runs on one GPU, as two schedulers sharing their experts
  (``--data-parallel-size 2`` with expert parallelism), or as attention and
  expert workers exchanging through DeepEP.
- A change of the graph buckets is another layout (``shrink``): the target
  opens regions of its own, the serving engines give their device memory
  back as their requests end, and the target grows into it, so no segment
  is ever held by both epochs. A cached prompt reaches the target through
  the host tier, where the write policy stored its blocks.
- A change of the model's files keeps the layout and changes every key: the
  target loads its weights beside the serving epoch's and reads none of its
  blocks; a commit leaves its weights alone on the GPU, and a rollback
  resumes the previous epoch on the weights it kept
  (``update_config.previous_epoch``) and finds the blocks the target evicted
  from the shared regions in the host tier or the remote store behind it.

What the tests check is what the admin API reports: ``PUT /admin/config``
answers the change it started, ``GET /admin/config`` the epoch that executes
(``executing``), the update with its phase (``update``), the node's main
processes (``processes``) and every GPU's quota (``quotas``), and
``/metrics`` the prefix cache's counters. The tests print the time of every
switch. Each runs a Storage Server of its own: ``XDG_RUNTIME_DIR`` points the
daemon at a directory of the test's, and the server the daemon starts there
is stopped at the end.
"""

from __future__ import annotations

import json
import os
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
    apply_config,
    dedicated_experts,
    engine_processes,
    engine_revision,
    engine_selection,
    find_free_port,
    granted_devices,
    host_asset,
    prefix_counters,
    qwen3_checkpoint,
    require_uniserve_binary,
    runs,
    runtime_directory,
    scaled,
    server_process,
    tokens_found,
    update_report,
    wait_for_phase,
    wait_until,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

MAX_TOKENS = 16384
# Tokens the client reads before it asks for the change.
EARLY_TOKENS = 64
PROMPT = (
    "Count from 1 to 5000, separated by commas and spaces. "
    "Write only the numbers."
)
# A prompt of several blocks, whose blocks the serving epoch commits.
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
# A change of another layout: graphs of other decode batch sizes.
OTHER_LAYOUT = "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8, 16]"
# How long a target takes to reach its trial, and a commit or a rollback to
# end, at most.
UPDATE_TIMEOUT_S = 900
# The items of a GPU's quota an epoch holds while it serves.
EPOCH_ITEMS = ("weights", "cache_segment", "graph_pool", "process")


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
) -> Iterator[str]:
    """A deployment serving `name` on `devices`; its URL."""
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


def _executing(base_url: str) -> int:
    """The epoch that serves, which ``GET /admin/config`` reports."""
    executing = admin_config(base_url)["executing"]
    assert executing["state"] == "serving", executing
    return executing["epoch"]


def _epoch_change(
    base_url: str, *assignments: str, **body: object
) -> tuple[dict, int]:
    """Applies `assignments` as an epoch change and waits for its trial;
    returns the update and the epoch on trial.
    """  # noqa: D205
    response = apply_config(base_url, *assignments, **body)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["change"] == "epoch", result
    update = wait_for_phase(base_url, "trial", timeout_s=UPDATE_TIMEOUT_S)
    print(f"switch: {update['switch_ms']} ms")
    return update, _executing(base_url)


def _commit(base_url: str) -> None:
    """Commits the trial and waits until it released the previous epoch."""
    committed = admin(base_url, "config/commit")
    assert committed.status_code == 200, committed.text
    wait_for_phase(base_url, "committed", timeout_s=UPDATE_TIMEOUT_S)


def _rollback(base_url: str, reason: str) -> dict:
    """Rolls the trial back and waits until it ended; returns the update."""
    rolled = admin(base_url, "config/rollback", {"reason": reason})
    assert rolled.status_code == 200, rolled.text
    update = wait_for_phase(base_url, "rolled_back", timeout_s=UPDATE_TIMEOUT_S)
    assert update["failure"]["reason"] == reason, update
    return update


def _change_under_traffic(
    base_url: str, name: str, streaming: Streaming, assignments: tuple[str, ...]
) -> tuple[int, list[Streaming]]:
    """Keep one request running throughout preparation, retaining every
    result; returns the epoch on trial and the streams.

    A finite request can finish while a target prepares. Start the next with
    the same prompt and token limit so the test observes the migration of a
    live request regardless of the number of requests preparation spans.
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


def _dry_run(base_url: str, *assignments: str, **body: object) -> dict:
    """The report of a dry run of `assignments`, which changes nothing: the
    epoch it would switch to, its engine package, its engines, and whether
    the layouts differ (``shrink``).
    """  # noqa: D205
    serving = _executing(base_url)
    response = apply_config(base_url, *assignments, dry_run=True, **body)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["change"] == "dry_run", result
    report = result["dry_run"]
    assert report["change"] == "epoch", report
    assert report["epoch"] != serving, report
    assert _executing(base_url) == serving
    return report


def _quotas(status: dict) -> list[dict]:
    """Every GPU's quota ``GET /admin/config`` reports, each of
    whose reservations' bytes sum to no more than its limit.
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
    """The sets of epochs holding the quotas' reservations of `item`."""
    return {
        tuple(entry["epochs"])
        for quota in quotas
        for entry in quota["reservations"]
        if entry["item"] == item
    }


def _segments_of(base_url: str, epoch: int) -> int:
    """The KV cache segments `epoch` holds on the deployment's GPUs."""
    return sum(
        1
        for quota in _quotas(admin_config(base_url))
        for entry in quota["reservations"]
        if entry["item"] == "cache_segment" and epoch in entry["epochs"]
    )


def _engine_pids(base_url: str) -> set[int]:
    """The process ids of the deployment's main processes."""
    return {process["pid"] for process in engine_processes(base_url)}


def _reused_by_the_migrated_request(base_url: str, before: dict) -> None:
    """The target's admission of the migrated request found every token it
    looked up cached: the blocks the serving epoch committed under their
    keys, the partial last one by its tail key, so admission leaves only the
    last emitted token to compute.
    """  # noqa: D205
    wait_until(
        lambda: prefix_counters(base_url)["queries"] > before["queries"],
        30,
        "the migrated request's cache query was not reported",
    )
    time.sleep(scaled(2.0))
    after = prefix_counters(base_url)
    reused = after["hits"] - before["hits"]
    queried = after["queries"] - before["queries"]
    assert reused == queried > 0, (reused, queried)


@pytest.mark.parametrize(
    "topology",
    [
        "single",
        pytest.param("experts", marks=pytest.mark.gpus(2)),
        pytest.param("dedicated", marks=pytest.mark.gpus(2)),
    ],
)
def test_an_epoch_change_continues_a_streaming_request_and_commits(
    tmp_path: Path, topology: str
):
    devices, options = _topology(topology, tmp_path)
    name = "qwen3-epoch-change"
    with _serve(tmp_path, name, devices, options) as base_url:
        serving = _executing(base_url)
        engines_before = _engine_pids(base_url)
        # Every GPU's quota reserves the serving epoch alone, within its
        # limit: its weights, KV cache segments, graph pool and process.
        quotas = _quotas(admin_config(base_url))
        for item in EPOCH_ITEMS:
            assert _epochs_of(quotas, item) == {(serving,)}, quotas

        change = engine_revision()
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        wait_until(
            lambda: prefix_counters(base_url)["queries"] > 0,
            30,
            "the original request's cache query was not reported",
        )
        counters = prefix_counters(base_url)
        # A dry run reports the change and changes nothing; a change of the
        # engine alone keeps the layout.
        report = _dry_run(base_url, change)
        print(f"dry run: {json.dumps(report)}")
        assert report["shrink"] is False, report
        update, epoch = _epoch_change(base_url, change)
        streaming.mark()
        assert epoch != serving
        assert update["shrink"] is False, update

        # The target serves on trial beside the parked previous epoch, whose
        # main process still runs, and the trial refuses another change.
        # Both epochs share each GPU's quota: the weights and the segments of
        # the regions both name are reserved once, held by both, beside each
        # epoch's graph pool and process.
        assert all(runs(pid) for pid in engines_before), engines_before
        quotas = _quotas(admin_config(base_url))
        both = (serving, epoch)
        assert _epochs_of(quotas, "weights") == {both}, quotas
        assert both in _epochs_of(quotas, "cache_segment"), quotas
        assert _epochs_of(quotas, "process") == {(serving,), (epoch,)}
        refused = apply_config(base_url, "max_running_requests=4")
        assert refused.status_code == 409, refused.text

        # The operator commits while the count streams on the target.
        streaming.wait_for(streaming.marked + 64)
        _reused_by_the_migrated_request(base_url, counters)
        _commit(base_url)
        status = admin_config(base_url)
        assert status["executing"]["epoch"] == epoch, status["executing"]
        configured = tomllib.loads(status["config"])["worker_config"]
        revision = configured["engine_revision"]
        assert change == f"worker_config.engine_revision={revision}", status
        # The commit released everything only the previous epoch held: the
        # segments the target mapped are its own on the quota now. The
        # previous epoch's ranks end their reservations as its process
        # exits.
        wait_until(
            lambda: all(
                _epochs_of(_quotas(admin_config(base_url)), item) == {(epoch,)}
                for item in EPOCH_ITEMS
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
        "the stream ended before the target served"
    )
    assert streaming.tokens == reference
    assert streaming.usage == reference_usage


def test_a_target_continues_a_request_from_its_predecessors_device_blocks(
    tmp_path: Path,
):
    """A target of the same layout reads the streaming request's blocks
    where the serving epoch left them: with no host tier, every token the
    request computed is a device hit of its admission on the target, and the
    stream is the count alone.
    """  # noqa: D205
    name = "qwen3-device-blocks"
    options = ("--set", "kv_cache_config.host_tier=false")
    with _serve(tmp_path, name, options=options) as base_url:
        change = engine_revision()
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        wait_until(
            lambda: prefix_counters(base_url)["queries"] > 0,
            30,
            "the original request's cache query was not reported",
        )
        counters = prefix_counters(base_url)

        _epoch_change(base_url, change)
        streaming.mark()
        streaming.wait_for(streaming.marked + 64)
        _reused_by_the_migrated_request(base_url, counters)
        host_hits = prefix_counters(base_url)["host_hits"]
        assert host_hits == counters["host_hits"], (host_hits, counters)
        streaming.join()
        reference, reference_usage = _ask(base_url, name)

    assert len(streaming.tokens) > streaming.marked, (
        "the stream ended before the target served"
    )
    assert streaming.tokens == reference
    assert streaming.usage == reference_usage


# Distinct prompts whose blocks together outgrow the device KV cache (32768
# tokens), pushing out the blocks of a prompt asked before them, which the
# host tier keeps.
EVICTING_FILLS = 6
# A prompt longer than one segment of the KV cache (an eighth of its 32768
# tokens).
FILL_PROMPT = "Name one colour of the rainbow. " * 900


def _fill(base_url: str, name: str, first: int) -> None:
    """Asks `EVICTING_FILLS` fill prompts numbered from `first`, whose blocks
    push older blocks out of the device KV cache.
    """  # noqa: D205
    for index in range(first, first + EVICTING_FILLS):
        _chat(base_url, name, f"{index}: {FILL_PROMPT}", tokens=1)


def _found(base_url: str, name: str, prompt: str) -> tuple[str, dict]:
    """The answer to `prompt` and the tokens it found cached per tier."""
    return tokens_found(base_url, lambda: _chat(base_url, name, prompt))


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
    # The cache also holds the executor's scratch units. Both capacities must
    # fit the full count beside those units throughout the target's
    # preparation.
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


def test_a_change_of_the_engine_alone_invalidates_nothing(tmp_path: Path):
    name = "qwen3-same-keys"
    with _serve(tmp_path, name) as base_url:
        serving = _executing(base_url)
        # The prompt's blocks are computed once and then found cached, on the
        # device and, once pushed out of it, in the host tier.
        answer = _chat(base_url, name, CACHED_PROMPT)
        cached_answer, device_before = _found(base_url, name, CACHED_PROMPT)
        assert cached_answer == answer and device_before["hits"] > 0, (
            cached_answer,
            device_before,
        )
        _fill(base_url, name, 0)
        _, host_before = _found(base_url, name, CACHED_PROMPT)
        assert host_before["host_hits"] > 0, host_before

        # Another engine revision changes none of the weights or keys: the
        # target maps every weight the serving epoch holds.
        _, epoch = _epoch_change(base_url, engine_revision())
        weights = _epochs_of(_quotas(admin_config(base_url)), "weights")
        assert weights == {(serving, epoch)}, weights
        _commit(base_url)

        # The committed epoch finds as many of the prompt's tokens cached,
        # on the device and in the host tier, and answers it as before.
        cached_answer, device_after = _found(base_url, name, CACHED_PROMPT)
        assert cached_answer == answer
        assert device_after["hits"] == device_before["hits"], device_after
        _fill(base_url, name, EVICTING_FILLS)
        _, host_after = _found(base_url, name, CACHED_PROMPT)
        assert (host_after["hits"], host_after["host_hits"]) == (
            host_before["hits"],
            host_before["host_hits"],
        ), (host_after, host_before)


def test_a_change_of_layout_moves_device_memory_and_reaches_the_host_tier(
    tmp_path: Path,
):
    """A change of the graph buckets is another layout. The target opens
    regions of its own; the serving engines give their device memory back as
    their requests end or migrate, and the target grows into it, segment by
    segment, so no segment is held by the two epochs at once (R10). The
    streaming request migrates at the drain's deadline and computes on the
    target what it reaches through the host tier. A prompt the serving epoch
    cached is found as deep as before, every block of it loaded from the host
    tier, where the write policy stored it.
    """  # noqa: D205
    name = "qwen3-layout-change"
    with _serve(tmp_path, name) as base_url:
        serving = _executing(base_url)
        engines_before = _engine_pids(base_url)
        held = _segments_of(base_url, serving)
        answer = _chat(base_url, name, CACHED_PROMPT)
        _, cached = _found(base_url, name, CACHED_PROMPT)
        assert cached["hits"] > 0, cached

        report = _dry_run(base_url, OTHER_LAYOUT)
        assert report["shrink"] is True, report
        streaming = Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        update, epoch = _epoch_change(base_url, OTHER_LAYOUT)
        streaming.mark()
        assert update["shrink"] is True, update

        # The two layouts' device memory is never in two regions: no segment
        # is reserved for both epochs, the drained epoch holds fewer than it
        # did, and the target holds the rest.
        quotas = _quotas(admin_config(base_url))
        assert (serving, epoch) not in _epochs_of(quotas, "cache_segment"), (
            quotas
        )
        assert _segments_of(base_url, serving) < held, quotas
        assert _segments_of(base_url, epoch) > 0, quotas

        # The migrated request streams on the target, so its admission is
        # counted before the cached prompt's is read.
        streaming.wait_for(streaming.marked + 64)
        time.sleep(scaled(2.0))
        cached_answer, found = _found(base_url, name, CACHED_PROMPT)
        assert cached_answer == answer
        assert found["hits"] == cached["hits"], (found, cached)
        assert found["host_hits"] == found["hits"], found

        _commit(base_url)
        status = admin_config(base_url)
        graphs = tomllib.loads(status["config"])["cuda_graph_config"]
        assert graphs["decode_batch_sizes"] == [1, 2, 4, 8, 16], graphs
        wait_until(
            lambda: all(
                _epochs_of(_quotas(admin_config(base_url)), item) == {(epoch,)}
                for item in EPOCH_ITEMS
            ),
            30,
            "the quotas did not reserve the committed epoch alone",
        )
        for pid in engines_before:
            wait_until(
                lambda pid=pid: not runs(pid),
                120,
                f"engine {pid} of the previous epoch outlived the commit",
            )
        streaming.join()
        reference, reference_usage = _ask(base_url, name)

    assert len(streaming.tokens) > streaming.marked, (
        "the stream ended before the target served"
    )
    assert streaming.tokens == reference
    assert streaming.usage == reference_usage


def _updated_checkpoint() -> str:
    """Another checkpoint of the model, of the same files but for other
    weights (``UNISERVE_QWEN3_UPDATED_MODEL``).
    """  # noqa: D205
    return str(
        host_asset("UNISERVE_QWEN3_UPDATED_MODEL", QWEN3_UPDATED_CHECKPOINT)
    )


# The share of the GPU a deployment whose target loads every weight again
# serves with: both epochs' weights, the target's graph pool and the KV
# regions fit in it.
WEIGHTS_SHARE = ("--set", "mem_fraction_static=0.9")


def _weights_of(base_url: str) -> set[tuple[int, ...]]:
    """The sets of epochs holding weights on the deployment's GPUs."""
    return _epochs_of(_quotas(admin_config(base_url)), "weights")


@pytest.mark.parametrize(
    ("previous_epoch", "end"),
    [
        # The target serves without the predecessor's weights of the
        # changed files: they are loaded, not mapped.
        pytest.param("parked", "commit", marks=pytest.mark.missing_state),
        ("parked", "rollback"),
        ("weights", "rollback"),
        ("host", "rollback"),
    ],
)
def test_a_change_of_model_files_loads_the_targets_weights(
    tmp_path: Path, previous_epoch: str, end: str
):
    """An epoch change to other model files of the same model loads the
    target's changed weights from them beside the serving epoch's and maps
    the tensors both checkpoints hold alike; the count streaming meanwhile
    migrates at the drain's deadline and computes its state again on the
    target, whose keys reach none of the serving epoch's blocks. The target
    answers with its weights. A commit leaves its weights alone on the GPU; a
    rollback answers as before on the previous epoch: parked, started again
    on the weights it kept on the GPU, or on the weights it kept in host
    storage, which return to the GPU first.
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
        assert update["shrink"] is False, update
        if previous_epoch == "host":
            # The previous epoch's weights the target does not map leave
            # its GPU for host storage while the target serves.
            wait_until(
                lambda: (
                    update_report(base_url)["moved_to_host_bytes"] > 0
                    or update_report(base_url)["host_refused"]
                ),
                300,
                "the previous epoch's weights did not move",
            )
            update = update_report(base_url)
            assert not update["host_refused"], update
        assert update["previous_epoch_kept"] == previous_epoch, update

        # The target answers with its own weights. Both epochs hold the
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
                context=lambda: f": weights held by {_weights_of(base_url)}",
            )
            streaming.join()
            # The committed epoch answers with the target's weights. Its
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
            "the target's weights outlived the rollback",
            context=lambda: f": weights held by {_weights_of(base_url)}",
        )


# The remote store behind the host tier (`kv_cache_config.mooncake`): the
# Mooncake master this host runs (`host:port`) and the library its clients
# load (`libmooncake_store.so`).
MOONCAKE_MASTER = os.environ.get("UNISERVE_MOONCAKE_MASTER")
MOONCAKE_LIBRARY = os.environ.get("MOONCAKE_STORE_LIBRARY")

# A prompt of 17 blocks of 64 tokens, which the fill prompts cached after it
# push out of a small host tier, and which a rollback prefetches from the
# remote store whole.
LONG_CACHED_PROMPT = "Describe the sea in one sentence. " * 160


def _mooncake_options() -> tuple[str, ...]:
    """Mooncake behind the host tier. The deployment's Storage Server gives
    Mooncake a segment of 4 GiB, which holds every block.
    """  # noqa: D205
    return (
        "--set",
        f'kv_cache_config.mooncake.master="{MOONCAKE_MASTER}"',
        "--set",
        "kv_cache_config.mooncake.global_segment_gib=4",
        "--set",
        f'kv_cache_config.mooncake.library="{MOONCAKE_LIBRARY}"',
    )


# Without a Mooncake master and its library, the case with a remote store
# does not run.
WITHOUT_MOONCAKE = pytest.mark.skipif(
    not (MOONCAKE_MASTER and MOONCAKE_LIBRARY),
    reason="no Mooncake master (UNISERVE_MOONCAKE_MASTER) and library "
    "(MOONCAKE_STORE_LIBRARY)",
)


@pytest.mark.parametrize(
    "tiers",
    ["host", pytest.param("remote_store", marks=WITHOUT_MOONCAKE)],
)
def test_a_rollback_finds_the_blocks_its_target_evicted_in_the_lower_tiers(
    tmp_path: Path, tiers: str
):
    """A change to other model files keeps the layout, so the target shares
    the serving epoch's regions, and changes every key, so it reads none of
    their blocks. Filling the regions with its own blocks, the target evicts
    the previous epoch's unpinned blocks, which it learns of from the buffer
    pool's listing. A rollback promotes the previous epoch again, which finds
    its prompt's blocks in the host tier, where its write policy stored
    them, and answers as before.

    With `remote_store`, the host tier of 1 GiB holds about 10900 tokens of
    Qwen3-30B-A3B at 96 KiB a token, fewer than the target's fills store, and
    every block stored on the host is stored on to Mooncake: the rollback's
    admission prefetches what the host tier no longer holds from the remote
    store, and loads it from the host.
    """  # noqa: D205
    name = "qwen3-epoch-evict"
    change = (
        f"model={_updated_checkpoint()}",
        "update_config.previous_epoch=parked",
    )
    options, cached = WEIGHTS_SHARE, CACHED_PROMPT
    if tiers == "remote_store":
        options += ("--set", "kv_cache_config.host_gib=1", *_mooncake_options())
        cached = LONG_CACHED_PROMPT
    with _serve(tmp_path, name, options=options) as base_url:
        original = _chat(base_url, name, cached)
        previous = _executing(base_url)

        report = _dry_run(base_url, *change, in_flight="recompute")
        assert report["shrink"] is False, report
        _epoch_change(base_url, *change, in_flight="recompute")
        assert _chat(base_url, name, cached) != original
        _fill(base_url, name, 0)

        _rollback(base_url, "the check")
        assert _executing(base_url) == previous
        answer, found = _found(base_url, name, cached)
        assert answer == original
        assert found["host_hits"] > 0, (
            f"the evicted blocks were not read back: {found}"
        )
