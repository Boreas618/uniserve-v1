"""An epoch change moves the requests in flight to a target epoch.

The target serves on trial until the operator commits it or rolls it back.

Qwen3-MoE streams a long greedy count. While it streams, ``PUT
/admin/config`` changes what only a new epoch executes: the target's engines
start beside the serving ones on the same GPUs, open their sessions with the
host's buffer pool, map the weights they share and park. The switch then lets
both epochs step (``specs/updates.md`` §3.7): the target is promoted and
admits every new request, and the serving engines drain beside it, stepping
their running requests until each ends or the drain's deadline
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
answers the change it started, once its target is on trial, ``GET
/admin/config`` the epoch that executes (``executing``), the update with its
phase (``update``), the node's main processes (``processes``) and every GPU's
quota (``quotas``), ``GET /admin/events`` the phases an update enters, and
``/metrics`` the prefix cache's counters. The tests print the time of every
switch. Each test's deployment runs a Storage Server of its own.
"""

from __future__ import annotations

import json
import threading
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import httpx
import pytest
from uniserve_testing import admin, builds, chat, client
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.builds import BuildServer
from uniserve_testing.deployments import HOST_CACHE_GIB, Deployments
from uniserve_testing.servers import Server

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

NAME = "qwen3-epoch-change"
# Tokens the client reads before it asks for the change.
EARLY_TOKENS = 64

#: The count, greedy and with each token's logprob, whose tokens a stream
#: and a later answer compare.
COUNT = chat.body(
    NAME,
    "Count from 1 to 5000, separated by commas and spaces. "
    "Write only the numbers.",
    16384,
    temperature=0,
    logprobs=True,
    **chat.NO_THINKING,
)
# A prompt of several blocks, whose blocks the serving epoch commits.
CACHED_PROMPT = "Describe the sea in one sentence. " * 40
# Options every deployment of the tests serves with: few captured graphs,
# so that epochs start while the count streams, and a small KV cache.
OPTIONS = (
    "--served-model-name",
    NAME,
    "--set",
    "kv_cache_config.max_tokens=32768",
    "--set",
    'chat_config.reasoning_parser="none"',
    "--set",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "--set",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
)
# A change of another layout: graphs of other decode batch sizes.
OTHER_LAYOUT = "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8, 16]"
# The items of a GPU's quota an epoch holds while it serves.
EPOCH_ITEMS = ("weights", "cache_segment", "graph_pool", "process")
# The states the daemon reports of a main process that no longer runs.
ENDED_STATES = ("stopped", "failed")

#: How long a stream may wait for its next line, in seconds (scaled): across
#: a switch, until the target serves the migrated request.
STREAM_GAP_S = 1800.0

#: How long the statistics a scheduler reports once a second take to include
#: what came before, in seconds (scaled): two report intervals. A hit is
#: counted when its admission's answers arrive, before the request computes,
#: so the counters are read once the report holding the query arrived and
#: another followed.
REPORT_SETTLE_S = 2.0

#: How long a request's prefix cache query takes to reach ``/metrics``, in
#: seconds (scaled): the report that carries it follows within a second.
QUERY_REPORTED_S = 30.0

#: How long a commit or a rollback takes to release what only one epoch held,
#: in seconds (scaled): the released epoch's processes finish their last
#: batch, exit, and their reservations end with them.
RELEASE_S = 120.0

#: How long the previous epoch's weights take to leave its GPU for host
#: storage once its target serves, in seconds (scaled): a copy of the
#: weights the target does not map.
MOVE_TO_HOST_S = 300.0


def _serve(
    deployments: Deployments, *options: str, host_gib: int = HOST_CACHE_GIB
) -> AbstractContextManager[Server]:
    """A deployment of Qwen3 with :data:`OPTIONS` and then `options`, its
    host tier `host_gib` GiB.
    """  # noqa: D205
    return deployments.serve(
        *OPTIONS,
        *options,
        model=CHECKPOINTS["qwen3"].model,
        host_gib=host_gib,
    )


def _topology(topology: str, directory: Path) -> tuple[str, ...]:
    """The options of a deployment on the case's GPUs.

    ``single``: one GPU. ``experts``: two schedulers sharing their experts
    over the all-to-all. ``dedicated``: attention on the first GPU and the
    routed experts on the second, exchanging through DeepEP; its deployment
    file is written into `directory`.
    """
    if topology == "single":
        return ()
    if topology == "experts":
        return (
            "--data-parallel-size",
            "2",
            "--set",
            "moe_config.expert_parallel=true",
        )
    return (
        *client.dedicated_experts(directory),
        "--set",
        "moe_config.expert_parallel=true",
    )


def _engine_revision(build_server: BuildServer, repository: Path) -> str:
    """An assignment that selects a new revision of the tree's engine.

    The revision is installed and built now at the run's Build Server. It
    runs the same code as the serving one under another revision, which is a
    change of the epoch's engine and of nothing else, and an update to it
    builds nothing while it prepares: a request started after this still
    streams when the switch comes.
    """
    installed = build_server.install(repository / builds.ENGINE_PACKAGE)
    build_server.resolve(installed.op, installed.backend, installed.revision)
    return f"worker_config.engine_revision={installed.revision}"


def _tokens(answer: dict[str, Any]) -> list[str]:
    """The tokens of an answer's first choice, from its logprobs."""
    return [
        entry["token"] for entry in answer["choices"][0]["logprobs"]["content"]
    ]


def _ask(server: Server) -> tuple[list[str], dict[str, Any]]:
    """The count's tokens and usage for the request alone."""
    answer = chat.complete(
        server.base_url, COUNT, server.scaled(chat.COMPLETION_S)
    )
    return _tokens(answer), answer["usage"]


def _chat(server: Server, prompt: str, tokens: int = 32) -> str:
    """The greedy answer to `prompt`, of at most `tokens` tokens."""
    request = chat.body(NAME, prompt, tokens, temperature=0, **chat.NO_THINKING)
    return chat.content(chat.complete(server.base_url, request))


class Streaming:
    """The count streaming on a thread of its own: the tokens it read, its
    usage, and how many tokens it had read when `mark` was called.

    ``done`` is the stream's future, which ends with the stream and raises
    what the stream raised.
    """  # noqa: D205

    def __init__(self, server: Server) -> None:
        """Start streaming the count from `server`."""
        self.tokens: list[str] = []
        self.usage: dict[str, Any] | None = None
        self.marked: int | None = None
        self._ended = False
        # Notified with each line the stream reads and when it ends.
        self._changed = threading.Condition()
        executor = ThreadPoolExecutor(max_workers=1)
        self.done: Future[None] = executor.submit(self._read, server)
        executor.shutdown(wait=False)

    def _read(self, server: Server) -> None:
        try:
            with httpx.stream(
                "POST",
                f"{server.base_url}/v1/chat/completions",
                json=chat.stream_options(COUNT),
                timeout=server.scaled(STREAM_GAP_S),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    tokens, usage, done = chat.stream_tokens(line)
                    if done:
                        break
                    with self._changed:
                        self.tokens.extend(tokens)
                        self.usage = usage or self.usage
                        self._changed.notify_all()
        finally:
            with self._changed:
                self._ended = True
                self._changed.notify_all()

    def wait_for(self, count: int) -> None:
        """Return once the stream read `count` tokens.

        Raises:
            AssertionError: When the stream ended first, or what the stream
                raised.
        """
        with self._changed:
            self._changed.wait_for(
                lambda: len(self.tokens) >= count or self._ended
            )
        if len(self.tokens) < count:
            self.done.result()
            raise AssertionError("the stream ended early")

    def mark(self) -> None:
        """Note how many tokens the stream read until now (``marked``)."""
        with self._changed:
            self.marked = len(self.tokens)

    def join(self) -> None:
        """Return once the stream ended, raising what it raised."""
        self.done.result()


def _executing(server: Server) -> int:
    """The epoch that serves, which ``GET /admin/config`` reports."""
    executing = admin.config(server)["executing"]
    assert executing["state"] == "serving", executing
    return executing["epoch"]


def _epoch_change(
    server: Server, *assignments: str, **body: object
) -> tuple[dict[str, Any], int]:
    """Apply `assignments` as an epoch change, answered once its target is on
    trial; the update and the epoch on trial.
    """  # noqa: D205
    response = admin.apply(server, *assignments, **body)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["change"] == "epoch", result
    update = result["update"]
    assert update["phase"] == "trial", update
    print(f"switch: {update['switch_ms']} ms")
    return update, _executing(server)


def _commit(server: Server) -> None:
    """Commit the trial; return once the update is committed."""
    after = admin.next_event(server)
    committed = admin.operation(server, "config/commit")
    assert committed.status_code == 200, committed.text
    admin.wait_for_phase(server, "committed", after=after)


def _rollback(server: Server, reason: str) -> dict[str, Any]:
    """Roll the trial back; the update once it rolled back."""
    after = admin.next_event(server)
    rolled = admin.operation(server, "config/rollback", {"reason": reason})
    assert rolled.status_code == 200, rolled.text
    update = admin.wait_for_phase(server, "rolled_back", after=after)
    assert update["failure"]["reason"] == reason, update
    return update


def _change_under_traffic(
    server: Server, streaming: Streaming, assignments: tuple[str, ...]
) -> tuple[int, list[Streaming]]:
    """Keep one request running throughout preparation, retaining every
    result; returns the epoch on trial and the streams.

    A finite request can finish while a target prepares. Start the next with
    the same prompt and token limit so the test observes the migration of a
    live request regardless of the number of requests preparation spans.
    """  # noqa: D205
    streams = [streaming]
    with ThreadPoolExecutor(max_workers=1) as executor:
        change = executor.submit(_epoch_change, server, *assignments)
        while True:
            wait((change, streams[-1].done), return_when=FIRST_COMPLETED)
            if change.done():
                break
            streams[-1].join()
            streams.append(Streaming(server))
        _, epoch = change.result()
    streams[-1].mark()
    return epoch, streams


def _dry_run(server: Server, *assignments: str, **body: object) -> dict:
    """The report of a dry run of `assignments`, which changes nothing: the
    epoch it would switch to, its engine package, its engines, and whether
    the layouts differ (``shrink``).
    """  # noqa: D205
    serving = _executing(server)
    response = admin.apply(server, *assignments, dry_run=True, **body)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["change"] == "dry_run", result
    report = result["dry_run"]
    assert report["change"] == "epoch", report
    assert report["epoch"] != serving, report
    assert _executing(server) == serving
    return report


def _quotas(server: Server) -> list[dict[str, Any]]:
    """Every GPU's quota of the deployment, each of whose reservations' bytes
    sum to no more than its limit.
    """  # noqa: D205
    quotas = admin.quotas(server)
    assert quotas, "the deployment reports no quota"
    for quota in quotas:
        used = sum(entry["bytes"] for entry in quota["reservations"])
        assert quota["limit"] is not None and used <= quota["limit"], quota
    return quotas


def _epochs_of(quotas: list[dict[str, Any]], item: str) -> set[tuple[int, ...]]:
    """The sets of epochs holding the quotas' reservations of `item`."""
    return {
        tuple(entry["epochs"])
        for quota in quotas
        for entry in quota["reservations"]
        if entry["item"] == item
    }


def _segments_of(server: Server, epoch: int) -> int:
    """The KV cache segments `epoch` holds on the deployment's GPUs."""
    return sum(
        1
        for entry in admin.reservations(server)
        if entry["item"] == "cache_segment" and epoch in entry["epochs"]
    )


def _running_engines(server: Server) -> set[int]:
    """The process ids of the deployment's main processes that run, as the
    daemon reports them (``processes``).
    """  # noqa: D205
    return {
        process["pid"]
        for process in admin.engine_processes(server)
        if process["pid"] is not None and process["state"] not in ENDED_STATES
    }


def _released(server: Server, epoch: int, previous: set[int]) -> None:
    """Return once the quotas reserve the committed `epoch` alone and the
    previous epoch's main processes `previous` ended.

    The commit's phase is an event, but the releases that follow it are not:
    the processes' exits and the reservations' ends are read every second.
    """  # noqa: D205
    server.wait_until(
        lambda: all(
            _epochs_of(_quotas(server), item) == {(epoch,)}
            for item in EPOCH_ITEMS
        ),
        RELEASE_S,
        "the quotas did not reserve the committed epoch alone",
    )
    server.wait_until(
        lambda: not previous & _running_engines(server),
        RELEASE_S,
        "an engine of the previous epoch outlived the commit",
    )


def _counters(server: Server) -> dict[str, float]:
    """The prefix cache's counters of every epoch so far, in tokens: the
    tokens admissions looked up (``queries``, a request's prompt less its
    last token), the tokens of their hits (``hits``) and those of the hits
    loaded from the host tier (``host_hits``).
    """  # noqa: D205
    return {
        name: client.metric(server.base_url, f"uniserve:prefix_cache_{name}")
        for name in ("queries", "hits", "host_hits")
    }


def _query_reported(server: Server, queries: float, what: str) -> None:
    """Return once ``/metrics`` counts more prefix cache queries than
    `queries`.

    The counters reach ``/metrics`` with the schedulers' reports, which no
    event of the node's log announces, so they are read every second.
    """  # noqa: D205
    server.wait_until(
        lambda: _counters(server)["queries"] > queries, QUERY_REPORTED_S, what
    )


def _settled_counters(server: Server) -> dict[str, float]:
    """The counters once the report after the latest one arrived.

    No report states a sequence number or the requests it counts, so the
    reports of what came before are waited for by their interval.
    """
    time.sleep(server.scaled(REPORT_SETTLE_S))
    return _counters(server)


def _found(
    server: Server, ask: Callable[[], str]
) -> tuple[str, dict[str, float]]:
    """What `ask` answers, and how much each prefix cache counter grew over
    it: the tokens its admission found cached, per tier.

    The counters before `ask` are read after the reports of the admissions
    before it, so the growth counts `ask`'s admission alone. `ask` must look
    its prompt up: a deployment without ``prefix_caching`` counts no query.
    """  # noqa: D205
    before = _settled_counters(server)
    answer = ask()
    _query_reported(
        server,
        before["queries"],
        "the request's prefix cache query was not reported",
    )
    after = _settled_counters(server)
    return answer, {name: after[name] - before[name] for name in after}


def _reused_by_the_migrated_request(
    server: Server, before: dict[str, float]
) -> None:
    """The target's admission of the migrated request found every token it
    looked up cached: the blocks the serving epoch committed under their
    keys, the partial last one by its tail key, so admission leaves only the
    last emitted token to compute.
    """  # noqa: D205
    _query_reported(
        server,
        before["queries"],
        "the migrated request's cache query was not reported",
    )
    after = _settled_counters(server)
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
    deployments: Deployments,
    build_server: BuildServer,
    repository: Path,
    tmp_path: Path,
    topology: str,
):
    with _serve(deployments, *_topology(topology, tmp_path)) as server:
        serving = _executing(server)
        engines_before = _running_engines(server)
        # Every GPU's quota reserves the serving epoch alone, within its
        # limit: its weights, KV cache segments, graph pool and process.
        quotas = _quotas(server)
        for item in EPOCH_ITEMS:
            assert _epochs_of(quotas, item) == {(serving,)}, quotas

        change = _engine_revision(build_server, repository)
        streaming = Streaming(server)
        streaming.wait_for(EARLY_TOKENS)
        _query_reported(
            server, 0, "the original request's cache query was not reported"
        )
        counters = _counters(server)
        # A dry run reports the change and changes nothing; a change of the
        # engine alone keeps the layout.
        report = _dry_run(server, change)
        print(f"dry run: {json.dumps(report)}")
        assert report["shrink"] is False, report
        update, epoch = _epoch_change(server, change)
        streaming.mark()
        assert epoch != serving
        assert update["shrink"] is False, update

        # The target serves on trial beside the parked previous epoch, whose
        # main process still runs, and the trial refuses another change.
        # Both epochs share each GPU's quota: the weights and the segments of
        # the regions both name are reserved once, held by both, beside each
        # epoch's graph pool and process.
        assert engines_before <= _running_engines(server), engines_before
        quotas = _quotas(server)
        both = (serving, epoch)
        assert _epochs_of(quotas, "weights") == {both}, quotas
        assert both in _epochs_of(quotas, "cache_segment"), quotas
        assert _epochs_of(quotas, "process") == {(serving,), (epoch,)}
        refused = admin.apply(server, "max_running_requests=4")
        assert refused.status_code == 409, refused.text

        # The operator commits while the count streams on the target.
        streaming.wait_for(streaming.marked + 64)
        _reused_by_the_migrated_request(server, counters)
        _commit(server)
        status = admin.config(server)
        assert status["executing"]["epoch"] == epoch, status["executing"]
        configured = tomllib.loads(status["config"])["worker_config"]
        revision = configured["engine_revision"]
        assert change == f"worker_config.engine_revision={revision}", status
        # The commit released everything only the previous epoch held: the
        # segments the target mapped are its own on the quota now, and the
        # previous epoch's engine exited, its reservations ending with it.
        _released(server, epoch, engines_before)
        streaming.join()
        reference, reference_usage = _ask(server)

    assert len(streaming.tokens) > streaming.marked, (
        "the stream ended before the target served"
    )
    assert streaming.tokens == reference
    assert streaming.usage == reference_usage


def test_a_target_continues_a_request_from_its_predecessors_device_blocks(
    deployments: Deployments, build_server: BuildServer, repository: Path
):
    """A target of the same layout reads the streaming request's blocks
    where the serving epoch left them: with no host tier, every token the
    request computed is a device hit of its admission on the target, and the
    stream is the count alone.
    """  # noqa: D205
    with _serve(
        deployments, "--set", "kv_cache_config.host_tier=false"
    ) as server:
        change = _engine_revision(build_server, repository)
        streaming = Streaming(server)
        streaming.wait_for(EARLY_TOKENS)
        _query_reported(
            server, 0, "the original request's cache query was not reported"
        )
        counters = _counters(server)

        _epoch_change(server, change)
        streaming.mark()
        streaming.wait_for(streaming.marked + 64)
        _reused_by_the_migrated_request(server, counters)
        host_hits = _counters(server)["host_hits"]
        assert host_hits == counters["host_hits"], (host_hits, counters)
        streaming.join()
        reference, reference_usage = _ask(server)

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


def _fill(server: Server, first: int) -> None:
    """Asks `EVICTING_FILLS` fill prompts numbered from `first`, whose blocks
    push older blocks out of the device KV cache.
    """  # noqa: D205
    for index in range(first, first + EVICTING_FILLS):
        _chat(server, f"{index}: {FILL_PROMPT}", tokens=1)


def test_a_resource_change_continues_requests_with_a_smaller_cache(
    deployments: Deployments,
):
    """Quotas, queue depth and cache capacity change through an epoch."""
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
        deployments, "--set", "kv_cache_config.max_tokens=65536"
    ) as server:
        before = _executing(server)
        streaming = Streaming(server)
        streaming.wait_for(EARLY_TOKENS)
        epoch, streams = _change_under_traffic(server, streaming, assignments)
        assert epoch != before
        _commit(server)
        configured = tomllib.loads(admin.config(server)["config"])
        assert configured["mem_fraction_static"] == 0.8
        assert configured["device_config"]["queue_depth"] == 3
        assert configured["kv_cache_config"]["max_tokens"] == 32768
        assert configured["kv_cache_config"]["host_gib"] == 8
        for stream in streams:
            stream.join()
        reference, usage = _ask(server)
        for stream in streams:
            assert stream.tokens == reference
            assert stream.usage == usage


def test_a_change_of_the_engine_alone_invalidates_nothing(
    deployments: Deployments, build_server: BuildServer, repository: Path
):
    with _serve(deployments) as server:
        serving = _executing(server)
        # The prompt's blocks are computed once and then found cached, on the
        # device and, once pushed out of it, in the host tier.
        answer = _chat(server, CACHED_PROMPT)
        cached_answer, device_before = _found(
            server, lambda: _chat(server, CACHED_PROMPT)
        )
        assert cached_answer == answer and device_before["hits"] > 0, (
            cached_answer,
            device_before,
        )
        _fill(server, 0)
        _, host_before = _found(server, lambda: _chat(server, CACHED_PROMPT))
        assert host_before["host_hits"] > 0, host_before

        # Another engine revision changes none of the weights or keys: the
        # target maps every weight the serving epoch holds.
        _, epoch = _epoch_change(
            server, _engine_revision(build_server, repository)
        )
        weights = _epochs_of(_quotas(server), "weights")
        assert weights == {(serving, epoch)}, weights
        _commit(server)

        # The committed epoch finds as many of the prompt's tokens cached,
        # on the device and in the host tier, and answers it as before.
        cached_answer, device_after = _found(
            server, lambda: _chat(server, CACHED_PROMPT)
        )
        assert cached_answer == answer
        assert device_after["hits"] == device_before["hits"], device_after
        _fill(server, EVICTING_FILLS)
        _, host_after = _found(server, lambda: _chat(server, CACHED_PROMPT))
        assert (host_after["hits"], host_after["host_hits"]) == (
            host_before["hits"],
            host_before["host_hits"],
        ), (host_after, host_before)


def test_a_change_of_layout_moves_device_memory_and_reaches_the_host_tier(
    deployments: Deployments,
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
    with _serve(deployments) as server:
        serving = _executing(server)
        engines_before = _running_engines(server)
        held = _segments_of(server, serving)
        answer = _chat(server, CACHED_PROMPT)
        _, cached = _found(server, lambda: _chat(server, CACHED_PROMPT))
        assert cached["hits"] > 0, cached

        report = _dry_run(server, OTHER_LAYOUT)
        assert report["shrink"] is True, report
        streaming = Streaming(server)
        streaming.wait_for(EARLY_TOKENS)
        update, epoch = _epoch_change(server, OTHER_LAYOUT)
        streaming.mark()
        assert update["shrink"] is True, update

        # The two layouts' device memory is never in two regions: no segment
        # is reserved for both epochs, the drained epoch holds fewer than it
        # did, and the target holds the rest.
        quotas = _quotas(server)
        assert (serving, epoch) not in _epochs_of(quotas, "cache_segment"), (
            quotas
        )
        assert _segments_of(server, serving) < held, quotas
        assert _segments_of(server, epoch) > 0, quotas

        # The migrated request streams on the target, so its admission is
        # counted before the cached prompt's is read.
        streaming.wait_for(streaming.marked + 64)
        cached_answer, found = _found(
            server, lambda: _chat(server, CACHED_PROMPT)
        )
        assert cached_answer == answer
        assert found["hits"] == cached["hits"], (found, cached)
        assert found["host_hits"] == found["hits"], found

        _commit(server)
        status = admin.config(server)
        graphs = tomllib.loads(status["config"])["cuda_graph_config"]
        assert graphs["decode_batch_sizes"] == [1, 2, 4, 8, 16], graphs
        _released(server, epoch, engines_before)
        streaming.join()
        reference, reference_usage = _ask(server)

    assert len(streaming.tokens) > streaming.marked, (
        "the stream ended before the target served"
    )
    assert streaming.tokens == reference
    assert streaming.usage == reference_usage


# The share of the GPU a deployment whose target loads every weight again
# serves with: both epochs' weights, the target's graph pool and the KV
# regions fit in it.
WEIGHTS_SHARE = ("--set", "mem_fraction_static=0.9")


def _weights_of(server: Server) -> set[tuple[int, ...]]:
    """The sets of epochs holding weights on the deployment's GPUs."""
    return _epochs_of(_quotas(server), "weights")


def _model_change(previous_epoch: str) -> tuple[str, str]:
    """The assignments of an epoch change to another checkpoint of the
    model, of the same files but for other weights, which keeps the previous
    epoch as `previous_epoch` (``update_config.previous_epoch``).

    Skips the calling case when no repository publishes that checkpoint, so
    a case asks for it before its deployment starts.
    """  # noqa: D205
    return (
        f'model="{CHECKPOINTS["qwen3-other-weights"].model}"',
        f'update_config.previous_epoch="{previous_epoch}"',
    )


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
    deployments: Deployments, previous_epoch: str, end: str
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
    change = _model_change(previous_epoch)
    with _serve(deployments, *WEIGHTS_SHARE) as server:
        original = _chat(server, CACHED_PROMPT)
        previous = _executing(server)
        assert _weights_of(server) == {(previous,)}
        streaming = Streaming(server)
        streaming.wait_for(EARLY_TOKENS)

        update, epoch = _epoch_change(server, *change)
        streaming.mark()
        assert update["shrink"] is False, update
        if previous_epoch == "host":
            # The previous epoch's weights the target does not map leave
            # its GPU for host storage while the target serves; the update
            # reports the move, which no event announces.
            def moved() -> bool:
                report = admin.config(server)["update"]
                return (
                    report["moved_to_host_bytes"] > 0 or report["host_refused"]
                )

            server.wait_until(
                moved,
                MOVE_TO_HOST_S,
                "the previous epoch's weights did not move",
            )
            update = admin.config(server)["update"]
            assert not update["host_refused"], update
        assert update["previous_epoch_kept"] == previous_epoch, update

        # The target answers with its own weights. Both epochs hold the
        # tensors the checkpoints share; the previous epoch keeps its other
        # weights on the GPU, or in host storage, which leaves no weights of
        # the previous epoch alone in the GPU's quota.
        changed = _chat(server, CACHED_PROMPT)
        assert changed != original, "the checkpoints answer alike"
        shared = {(previous, epoch)}
        if previous_epoch == "host":
            assert _weights_of(server) - shared == {(epoch,)}
        else:
            assert _weights_of(server) - shared == {(previous,), (epoch,)}
        streaming.wait_for(streaming.marked + 32)

        if end == "commit":
            _commit(server)
            # The weights' reservations end once the released epoch's
            # processes exited, which no event announces.
            server.wait_until(
                lambda: _weights_of(server) == {(epoch,)},
                RELEASE_S,
                "the previous epoch's weights outlived the commit",
            )
            streaming.join()
            # The committed epoch answers with the target's weights. Its
            # answer during the trial shared its batch with the count, whose
            # greedy decoding may part from an answer computed alone.
            assert _chat(server, CACHED_PROMPT) != original
            return
        _rollback(server, "the check")
        assert _executing(server) == previous
        streaming.join()
        assert _chat(server, CACHED_PROMPT) == original
        server.wait_until(
            lambda: _weights_of(server) == {(previous,)},
            RELEASE_S,
            "the target's weights outlived the rollback",
        )


# A prompt of 17 blocks of 64 tokens, which the fill prompts cached after it
# push out of a small host tier, and which a rollback prefetches from the
# remote store whole.
LONG_CACHED_PROMPT = "Describe the sea in one sentence. " * 160

#: The host tier of the remote store's case, in GiB: smaller than the blocks
#: the target's fills store, so the rollback finds the prompt's blocks in
#: the remote store rather than on the host.
SMALL_HOST_GIB = 1

#: The host storage the deployment's Storage Server gives Mooncake, in GiB:
#: room for every block the case stores, below the share at which Mooncake's
#: master evicts the oldest objects.
MOONCAKE_SEGMENT_GIB = 8


def _rollback_reads_back_evicted_blocks(
    server: Server, prompt: str, change: tuple[str, ...]
) -> None:
    """A change to other model files (`change`) keeps the layout, so the
    target shares the serving epoch's regions, and changes every key, so it
    reads none of their blocks. Filling the regions with its own blocks, the
    target evicts the previous epoch's unpinned blocks, which it learns of
    from the buffer pool's listing. A rollback promotes the previous epoch
    again, which finds its prompt's blocks in the lower tiers, where its
    write policy stored them, and answers as before.
    """  # noqa: D205
    original = _chat(server, prompt)
    previous = _executing(server)

    report = _dry_run(server, *change, in_flight="recompute")
    assert report["shrink"] is False, report
    _epoch_change(server, *change, in_flight="recompute")
    assert _chat(server, prompt) != original
    _fill(server, 0)

    _rollback(server, "the check")
    assert _executing(server) == previous
    answer, found = _found(server, lambda: _chat(server, prompt))
    assert answer == original
    assert found["host_hits"] > 0, (
        f"the evicted blocks were not read back: {found}"
    )


def test_a_rollback_finds_the_blocks_its_target_evicted_in_the_host_tier(
    deployments: Deployments,
):
    change = _model_change("parked")
    with _serve(deployments, *WEIGHTS_SHARE) as server:
        _rollback_reads_back_evicted_blocks(server, CACHED_PROMPT, change)


def test_a_rollback_finds_the_blocks_its_target_evicted_in_the_remote_store(
    deployments: Deployments, mooncake_master: str
):
    """Every block stored on the host is stored on to Mooncake, and the host
    tier holds fewer blocks than the target's fills store: the rollback's
    admission prefetches what the host tier no longer holds from the remote
    store, and loads it from the host.
    """  # noqa: D205
    change = _model_change("parked")
    with _serve(
        deployments,
        *WEIGHTS_SHARE,
        "--set",
        f'kv_cache_config.mooncake.master="{mooncake_master}"',
        "--set",
        f"kv_cache_config.mooncake.segment_gib={MOONCAKE_SEGMENT_GIB}",
        host_gib=SMALL_HOST_GIB,
    ) as server:
        _rollback_reads_back_evicted_blocks(server, LONG_CACHED_PROMPT, change)
