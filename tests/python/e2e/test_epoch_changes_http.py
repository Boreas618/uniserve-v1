"""An epoch change moves the requests in flight to a target epoch.

The target serves on trial until its trial lasted ``update_config.trial``,
when it commits by itself, or until a fault on trial rolls it back.

Qwen3-MoE streams a long greedy count. While it streams, ``PUT
/admin/policy`` changes what only a new epoch executes: the target's engines
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
  and the clone of its partial last block. The commit releases the previous
  epoch: its engine's process exits and the quotas reserve the committed
  epoch alone. The deployment runs on one GPU, as two schedulers sharing
  their experts (``data_parallel_size = 2`` with expert parallelism), or as
  attention and expert workers exchanging through DeepEP.
- A change of the request slots (``max_running_requests``) is another
  layout (``shrink``): the target opens regions of its own, the serving
  engines give their device memory back as their requests end, and the
  target grows into it, so no segment is ever held by both epochs. A cached
  prompt reaches the target through the host tier, where the write policy
  stored its blocks.
- A change of the model's files keeps the layout and changes every key: the
  target loads its weights beside the serving epoch's and reads none of its
  blocks; the commit leaves its weights alone on the GPU, and a rollback,
  which the loss of the target's main process on trial causes, resumes the
  previous epoch on the weights it kept (``update_config.previous_epoch``)
  and finds the blocks the target evicted from the shared regions in the
  host tier or the remote store behind it.

A change is applied on a thread of its own
(:func:`uniserve_testing.admin.changing`), whose answer arrives once it
committed or rolled back, while the case reads the target on trial. A case
that reads the trial serves with a trial of :data:`READ_TRIAL`, one whose
target a fault rolls back with :data:`FAULT_TRIAL`, and one that reads
nothing of the trial with :data:`SHORT_TRIAL`. What the tests check is what
the admin API reports: ``PUT /admin/policy`` answers the revision the change
converged to and ``GET /admin/policy`` states the configuration that serves;
the observation store (``/admin/query``) holds the update as it entered each
phase (``log`` events of source ``update``), the main processes' states
(``process`` events), the epoch replica 0 executes
(``uniserve_replica_epoch``), every GPU's quota and its reservations
(``uniserve_gpu_quota_bytes``, ``uniserve_gpu_reserved_bytes``) and the
prefix cache's counters. The tests print the time of every switch. Each
test's deployment runs a Storage Server of its own.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from uniserve_testing import admin, chat, client, updates
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import HOST_CACHE_GIB, Deployments
from uniserve_testing.packages import Config, merged
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
OPTIONS: Config = {
    "model": CHECKPOINTS["qwen3"].model,
    "served_model_name": NAME,
    "kv_cache_config": {"max_tokens": 32768},
    "chat_config": {"reasoning_parser": "none"},
    "cuda_graph_config": {
        "prefill_token_sizes": [512],
        "decode_batch_sizes": [1, 2, 4, 8],
    },
}
# A change of another layout: fewer request slots than the default 128, which
# lay the epoch's device memory out anew (`proto::config::Layout`).
OTHER_LAYOUT: Config = {"max_running_requests": 64}
# The items of a GPU's quota an epoch holds while it serves.
EPOCH_ITEMS = ("weights", "cache_segment", "graph_pool", "process")

#: The trial of a case that reads the target on trial: longer than what the
#: case reads takes (a few reports of the schedulers, a few dozen tokens of
#: the stream), after which the change commits by itself.
READ_TRIAL: Config = {"update_config": {"trial": "1m"}}

#: The trial of a case whose target a fault rolls back: longer than the case
#: takes to read the trial and fault, so that the fault alone ends it.
FAULT_TRIAL: Config = {"update_config": {"trial": "30m"}}

#: The trial of a case that reads nothing of it: the change commits at once.
SHORT_TRIAL: Config = {"update_config": {"trial": "1s"}}

#: The failure kinds a trial that lost its target's main process states: the
#: daemon's observation of the process's exit, or the engine's own report.
PROCESS_EXIT = ("process_exit", "process_exit_fault")

#: How long a stream may wait for its next line, in seconds (scaled): across
#: a switch, until the target serves the migrated request.
STREAM_GAP_S = 1800.0

#: How long the statistics a scheduler reports once a second take to include
#: what came before, in seconds (scaled): two report intervals. A hit is
#: counted when its admission's answers arrive, before the request computes,
#: so the counters are read once the report holding the query arrived and
#: another followed.
REPORT_SETTLE_S = 2.0

#: How long a request's prefix cache query takes to reach the observation
#: store, in seconds (scaled): the report that carries it follows within a
#: second.
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
    deployments: Deployments, *configs: Config, host_gib: int = HOST_CACHE_GIB
) -> AbstractContextManager[Server]:
    """A deployment of Qwen3 with :data:`OPTIONS` and `configs` over them,
    its host tier `host_gib` GiB.
    """  # noqa: D205
    return deployments.serve(merged(OPTIONS, *configs), host_gib=host_gib)


def _topology(topology: str) -> Config:
    """The options of a deployment on the case's GPUs.

    ``single``: one GPU. ``experts``: two schedulers sharing their experts
    over the all-to-all. ``dedicated``: attention on the first GPU and the
    routed experts on the second, exchanging through DeepEP.
    """
    if topology == "single":
        return {}
    if topology == "experts":
        return {
            "data_parallel_size": 2,
            "moe_config": {"expert_parallel": True},
        }
    return merged(
        client.dedicated_experts(), {"moe_config": {"expert_parallel": True}}
    )


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


def _executing(server: Server, after: int | None = None) -> int:
    """The epoch that serves, as the leader states replica 0's: as it last
    sampled it, or in a second after the moment `after` when stated.
    """  # noqa: D205
    serving = admin.replica(server, after=after)
    assert serving.state == "serving", serving
    assert serving.epoch is not None, serving
    return serving.epoch


@dataclass
class Trial:
    """An epoch change on trial: the update as its trial began, the epoch on
    trial, the answer the change's ``PUT`` will give, the daemon's clock
    before the change, after which its phases are read, and the moment its
    trial began, after which the gauges state the trial.
    """  # noqa: D205

    update: dict[str, Any]
    epoch: int
    answer: Future[httpx.Response]
    after: int
    began: int


@contextmanager
def _on_trial(server: Server, changes: Config) -> Iterator[Trial]:
    """Apply `changes` over the deployment's package as an epoch change, on
    a thread of its own; the change once its target is on trial, while the
    block runs. The block's end waits for the change's answer.
    """  # noqa: D205
    after = admin.now(server)
    with admin.changing(server, changes) as answer:
        entered = admin.wait_for_log(
            server,
            "update",
            after=after,
            phase=("trial", "abandoned", "rolled_back"),
        )
        update = entered.fields
        assert update["phase"] == "trial", update
        assert update["change"] == "epoch", update
        print(f"switch: {update['switch_ms']} ms")
        # The switch promoted the target before the trial began, so replica 0
        # executes it in every second the leader sampled since.
        executing = _executing(server, entered.time)
        assert executing == update["epoch"], (executing, update)
        yield Trial(update, executing, answer, after, began=entered.time)


def _committed(trial: Trial) -> dict[str, Any]:
    """The revision the change on trial converged to, once its trial ended
    and it committed by itself.
    """  # noqa: D205
    record = admin.revision(trial.answer.result())
    assert record["effect"] == "epoch", record
    return record


def _fault(server: Server, trial: Trial) -> dict[str, Any]:
    """Kill the target's main process on trial, which rolls the trial back;
    the update once it rolled back, which the change's answer states as its
    reason (``409``).
    """  # noqa: D205
    target = admin.serving_engine(server)["pid"]
    os.kill(target, signal.SIGKILL)
    update = admin.wait_for_log(
        server, "update", after=trial.after, phase="rolled_back"
    ).fields
    assert update["failure"]["kind"] in PROCESS_EXIT, update
    answer = trial.answer.result()
    assert answer.status_code == 409, answer.text
    return update


def _change_under_traffic(
    server: Server, streaming: Streaming, changes: Config
) -> tuple[int, list[Streaming]]:
    """Keep one request running throughout preparation, retaining every
    result, and apply `changes` as an epoch change, answered once it
    committed; returns the epoch it committed and the streams.

    A finite request can finish while a target prepares. Start the next with
    the same prompt and token limit so the test observes the migration of a
    live request regardless of the number of requests preparation spans.
    """  # noqa: D205
    streams = [streaming]
    after = admin.now(server)
    with (
        admin.changing(server, changes) as answer,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        on_trial = executor.submit(
            admin.wait_for_log,
            server,
            "update",
            after=after,
            phase=("trial", "abandoned", "rolled_back"),
        )
        while True:
            wait((on_trial, streams[-1].done), return_when=FIRST_COMPLETED)
            if on_trial.done():
                break
            streams[-1].join()
            streams.append(Streaming(server))
        update = on_trial.result().fields
        assert update["phase"] == "trial", update
        streams[-1].mark()
        epoch = update["epoch"]
        record = admin.revision(answer.result())
    assert record["effect"] == "epoch", record
    return epoch, streams


def _epochs(label: str) -> tuple[int, ...]:
    """The epochs a reservation's ``epochs`` label names, comma-joined."""
    return tuple(int(epoch) for epoch in label.split(",") if epoch)


def _quotas(server: Server, after: int | None = None) -> list[dict[str, Any]]:
    """Every GPU's quota of the deployment, each of whose reservations' bytes
    sum to no more than its limit, as the leader stated them in a second
    after the moment `after` (the daemon's clock now by default): each
    quota's ``limit`` and its ``reservations``, each with its ``item``,
    ``epochs`` and ``bytes``.
    """  # noqa: D205
    if after is None:
        after = admin.now(server)
    quotas: dict[tuple[str, str], dict[str, Any]] = {}

    def quota(labels: dict[str, str]) -> dict[str, Any]:
        gpu = (labels["host"], labels["gpu"])
        return quotas.setdefault(gpu, {"limit": None, "reservations": []})

    for sample in admin.gauge(server, "uniserve_gpu_quota_bytes", after=after):
        quota(sample.labels)["limit"] = int(sample.value)
    for sample in admin.gauge(
        server, "uniserve_gpu_reserved_bytes", after=after
    ):
        quota(sample.labels)["reservations"].append(
            {
                "item": sample.labels["item"],
                "epochs": _epochs(sample.labels["epochs"]),
                "bytes": int(sample.value),
            }
        )
    assert quotas, "the deployment reports no quota"
    for stated in quotas.values():
        used = sum(entry["bytes"] for entry in stated["reservations"])
        assert stated["limit"] is not None and used <= stated["limit"], stated
    return list(quotas.values())


def _epochs_of(quotas: list[dict[str, Any]], item: str) -> set[tuple[int, ...]]:
    """The sets of epochs holding the quotas' reservations of `item`."""
    return {
        tuple(entry["epochs"])
        for quota in quotas
        for entry in quota["reservations"]
        if entry["item"] == item
    }


def _segments_of(server: Server, epoch: int, after: int | None = None) -> int:
    """The KV cache segments `epoch` holds on the deployment's GPUs, in a
    second after the moment `after` (:func:`_quotas`).
    """  # noqa: D205
    return sum(
        1
        for quota in _quotas(server, after)
        for entry in quota["reservations"]
        if entry["item"] == "cache_segment" and epoch in entry["epochs"]
    )


def _running_engines(server: Server) -> set[int]:
    """The process ids of the deployment's main processes that run, as their
    latest ``process`` events state them.
    """  # noqa: D205
    return {
        process["pid"]
        for process in admin.processes(server, group="engine")
        if process["pid"] is not None and process["state"] not in admin.ENDED
    }


def _released(server: Server, epoch: int, previous: set[int]) -> None:
    """Return once the quotas reserve the committed `epoch` alone and the
    previous epoch's main processes `previous` ended.

    The releases follow the commit's phase: the processes' exits are their
    ``process`` events and the reservations' ends the quotas' gauges, which
    are read every second.
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
        name: admin.counter(server, f"uniserve_prefix_cache_{counter}_total")
        for name, counter in (
            ("queries", "queried_tokens"),
            ("hits", "hit_tokens"),
            ("host_hits", "host_hit_tokens"),
        )
    }


def _query_reported(server: Server, queries: float, what: str) -> None:
    """Return once the observation store counts more prefix cache queries
    than `queries`.

    The counters reach the store with the schedulers' reports, which no
    event announces, so they are read every second.
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
    repository: Path,
    topology: str,
):
    with _serve(deployments, _topology(topology), READ_TRIAL) as server:
        serving = _executing(server)
        engines_before = _running_engines(server)
        # Every GPU's quota reserves the serving epoch alone, within its
        # limit: its weights, KV cache segments, graph pool and process.
        quotas = _quotas(server)
        for item in EPOCH_ITEMS:
            assert _epochs_of(quotas, item) == {(serving,)}, quotas

        change = updates.engine_revision(server.build_server, repository)
        streaming = Streaming(server)
        streaming.wait_for(EARLY_TOKENS)
        _query_reported(
            server, 0, "the original request's cache query was not reported"
        )
        counters = _counters(server)
        # A change of the engine alone keeps the layout.
        with _on_trial(server, change) as trial:
            streaming.mark()
            assert trial.epoch != serving
            assert trial.update["shrink"] is False, trial.update

            # The target serves on trial beside the parked previous epoch,
            # whose main process still runs. Both epochs share each GPU's
            # quota: the weights and the segments of the regions both name
            # are reserved once, held by both, beside each epoch's graph
            # pool and process.
            assert engines_before <= _running_engines(server), engines_before
            quotas = _quotas(server, trial.began)
            both = (serving, trial.epoch)
            assert _epochs_of(quotas, "weights") == {both}, quotas
            assert both in _epochs_of(quotas, "cache_segment"), quotas
            assert _epochs_of(quotas, "process") == {
                (serving,),
                (trial.epoch,),
            }

            # The trial ends while the count streams on the target.
            streaming.wait_for(streaming.marked + 64)
            _reused_by_the_migrated_request(server, counters)
            record = _committed(trial)
        print(f"committed: {json.dumps(record)}")
        assert _executing(server) == trial.epoch
        configured = admin.policy(server).config["worker_config"]
        assert (
            configured["engine_revision"]
            == change["worker_config"]["engine_revision"]
        ), configured
        # The commit released everything only the previous epoch held: the
        # segments the target mapped are its own on the quota now, and the
        # previous epoch's engine exited, its reservations ending with it.
        _released(server, trial.epoch, engines_before)
        streaming.join()
        reference, reference_usage = _ask(server)

    assert len(streaming.tokens) > streaming.marked, (
        "the stream ended before the target served"
    )
    assert streaming.tokens == reference
    assert streaming.usage == reference_usage


def test_a_target_continues_a_request_from_its_predecessors_device_blocks(
    deployments: Deployments, repository: Path
):
    """A target of the same layout reads the streaming request's blocks
    where the serving epoch left them: with no host tier, every token the
    request computed is a device hit of its admission on the target, and the
    stream is the count alone.
    """  # noqa: D205
    with _serve(
        deployments, {"kv_cache_config": {"host_tier": False}}, SHORT_TRIAL
    ) as server:
        change = updates.engine_revision(server.build_server, repository)
        streaming = Streaming(server)
        streaming.wait_for(EARLY_TOKENS)
        _query_reported(
            server, 0, "the original request's cache query was not reported"
        )
        counters = _counters(server)

        with _on_trial(server, change) as trial:
            streaming.mark()
            _committed(trial)
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
    changes = {
        "mem_fraction_static": 0.8,
        "device_config": {"queue_depth": 3},
        "kv_cache_config": {"max_tokens": 32768, "host_gib": 8},
    }
    # The cache also holds the executor's scratch units. Both capacities must
    # fit the full count beside those units throughout the target's
    # preparation.
    with _serve(
        deployments, {"kv_cache_config": {"max_tokens": 65536}}, SHORT_TRIAL
    ) as server:
        before = _executing(server)
        streaming = Streaming(server)
        streaming.wait_for(EARLY_TOKENS)
        epoch, streams = _change_under_traffic(server, streaming, changes)
        assert epoch != before
        configured = admin.policy(server).config
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
    deployments: Deployments, repository: Path
):
    with _serve(deployments, READ_TRIAL) as server:
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
        change = updates.engine_revision(server.build_server, repository)
        with _on_trial(server, change) as trial:
            weights = _epochs_of(_quotas(server, trial.began), "weights")
            assert weights == {(serving, trial.epoch)}, weights
            _committed(trial)

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
    """A change of the request slots is another layout. The target opens
    regions of its own; the serving engines give their device memory back as
    their requests end or migrate, and the target grows into it, segment by
    segment, so no segment is held by the two epochs at once (R10). The
    streaming request migrates at the drain's deadline and computes on the
    target what it reaches through the host tier. A prompt the serving epoch
    cached is found as deep as before, every block of it loaded from the host
    tier, where the write policy stored it.
    """  # noqa: D205
    with _serve(deployments, READ_TRIAL) as server:
        serving = _executing(server)
        engines_before = _running_engines(server)
        held = _segments_of(server, serving)
        answer = _chat(server, CACHED_PROMPT)
        _, cached = _found(server, lambda: _chat(server, CACHED_PROMPT))
        assert cached["hits"] > 0, cached

        streaming = Streaming(server)
        streaming.wait_for(EARLY_TOKENS)
        with _on_trial(server, OTHER_LAYOUT) as trial:
            streaming.mark()
            assert trial.update["shrink"] is True, trial.update

            # The two layouts' device memory is never in two regions: no
            # segment is reserved for both epochs, the drained epoch holds
            # fewer than it did, and the target holds the rest.
            quotas = _quotas(server, trial.began)
            assert (serving, trial.epoch) not in _epochs_of(
                quotas, "cache_segment"
            ), quotas
            assert _segments_of(server, serving, trial.began) < held, quotas
            assert _segments_of(server, trial.epoch, trial.began) > 0, quotas

            # The migrated request streams on the target, so its admission
            # is counted before the cached prompt's is read.
            streaming.wait_for(streaming.marked + 64)
            cached_answer, found = _found(
                server, lambda: _chat(server, CACHED_PROMPT)
            )
            assert cached_answer == answer
            assert found["hits"] == cached["hits"], (found, cached)
            assert found["host_hits"] == found["hits"], found
            _committed(trial)

        configured = admin.policy(server).config
        assert configured["max_running_requests"] == 64, configured
        _released(server, trial.epoch, engines_before)
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
WEIGHTS_SHARE: Config = {"mem_fraction_static": 0.9}


def _model_change() -> Config:
    """The options of an epoch change to another checkpoint of the model,
    of the same files but for other weights.

    Skips the calling case when no repository publishes that checkpoint, so
    a case asks for it before its deployment starts.
    """  # noqa: D205
    return {"model": CHECKPOINTS["qwen3-other-weights"].model}


def _kept(previous_epoch: str) -> Config:
    """The options of a deployment whose updates keep the previous epoch as
    `previous_epoch` (``update_config.previous_epoch``) through their trial.
    """  # noqa: D205
    return {"update_config": {"previous_epoch": previous_epoch}}


def _weights_of(
    server: Server, after: int | None = None
) -> set[tuple[int, ...]]:
    """The sets of epochs holding weights on the deployment's GPUs, in a
    second after the moment `after` (:func:`_quotas`).
    """  # noqa: D205
    return _epochs_of(_quotas(server, after), "weights")


def _last_update(server: Server) -> dict[str, Any]:
    """The update as it entered its latest phase: the fields of the latest
    ``log`` event of source ``update``.
    """  # noqa: D205
    entered = admin.logs(server, "update")
    assert entered, "the deployment recorded no update"
    return entered[-1].fields


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
    answers with its weights. The commit leaves its weights alone on the
    GPU; a rollback, which the loss of the target's main process causes,
    answers as before on the previous epoch: parked, started again on the
    weights it kept on the GPU, or on the weights it kept in host storage,
    which return to the GPU first.
    """  # noqa: D205
    change = _model_change()
    trial_length = READ_TRIAL if end == "commit" else FAULT_TRIAL
    with _serve(
        deployments, WEIGHTS_SHARE, trial_length, _kept(previous_epoch)
    ) as server:
        original = _chat(server, CACHED_PROMPT)
        previous = _executing(server)
        assert _weights_of(server) == {(previous,)}
        streaming = Streaming(server)
        streaming.wait_for(EARLY_TOKENS)

        rolled: dict[str, Any] | None = None
        with _on_trial(server, change) as trial:
            streaming.mark()
            epoch = trial.epoch
            update = trial.update
            assert update["shrink"] is False, update
            if previous_epoch == "host":
                # The previous epoch's weights the target does not map leave
                # its GPU for host storage while the target serves, which no
                # event announces: the GPU's quota then reserves none of them
                # for the previous epoch alone. The update states the bytes
                # it moved in the phase it enters next.
                server.wait_until(
                    lambda: (previous,) not in _weights_of(server),
                    MOVE_TO_HOST_S,
                    "the previous epoch's weights did not move",
                )

            # The target answers with its own weights. Both epochs hold the
            # tensors the checkpoints share; the previous epoch keeps its
            # other weights on the GPU, or in host storage, which leaves no
            # weights of the previous epoch alone in the GPU's quota.
            changed = _chat(server, CACHED_PROMPT)
            assert changed != original, "the checkpoints answer alike"
            shared = {(previous, epoch)}
            if previous_epoch == "host":
                assert _weights_of(server) - shared == {(epoch,)}
            else:
                assert _weights_of(server) - shared == {(previous,), (epoch,)}
            streaming.wait_for(streaming.marked + 32)

            if end == "commit":
                _committed(trial)
            else:
                rolled = _fault(server, trial)
                print(f"rolled back: {json.dumps(rolled)}")

        # The update states what the previous epoch kept through the trial,
        # and, kept in host storage, that the node moved its weights there.
        ended = rolled if rolled is not None else _last_update(server)
        assert ended["previous_epoch_kept"] == previous_epoch, ended
        if previous_epoch == "host":
            assert not ended["host_refused"], ended
            assert ended["moved_to_host_bytes"] > 0, ended

        if end == "commit":
            # The weights' reservations end once the released epoch's
            # processes exited, which the quota states.
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


#: How the updates of a deployment whose rollback reads back evicted blocks
#: run: a trial a fault rolls back, and the requests a switch migrates
#: computing their state again, since the target reads none of the serving
#: epoch's keys.
RECOMPUTED: Config = merged(
    FAULT_TRIAL, {"update_config": {"in_flight": "recompute"}}
)


def _rollback_reads_back_evicted_blocks(server: Server, prompt: str) -> None:
    """A change to other model files keeps the layout, so the target shares
    the serving epoch's regions, and changes every key, so it reads none of
    their blocks. Filling the regions with its own blocks, the target evicts
    the previous epoch's unpinned blocks, which it learns of from the buffer
    pool's listing. A rollback, which the loss of the target's main process
    on trial causes, promotes the previous epoch again, which finds its
    prompt's blocks in the lower tiers, where its write policy stored them,
    and answers as before.
    """  # noqa: D205
    original = _chat(server, prompt)
    previous = _executing(server)

    with _on_trial(server, _model_change()) as trial:
        assert trial.update["shrink"] is False, trial.update
        assert _chat(server, prompt) != original
        _fill(server, 0)
        _fault(server, trial)

    assert _executing(server) == previous
    answer, found = _found(server, lambda: _chat(server, prompt))
    assert answer == original
    assert found["host_hits"] > 0, (
        f"the evicted blocks were not read back: {found}"
    )


def test_a_rollback_finds_the_blocks_its_target_evicted_in_the_host_tier(
    deployments: Deployments,
):
    _model_change()
    with _serve(deployments, WEIGHTS_SHARE, RECOMPUTED) as server:
        _rollback_reads_back_evicted_blocks(server, CACHED_PROMPT)


def test_a_rollback_finds_the_blocks_its_target_evicted_in_the_remote_store(
    deployments: Deployments, mooncake_master: str
):
    """Every block stored on the host is stored on to Mooncake, and the host
    tier holds fewer blocks than the target's fills store: the rollback's
    admission prefetches what the host tier no longer holds from the remote
    store, and loads it from the host.
    """  # noqa: D205
    _model_change()
    remote = {
        "kv_cache_config": {
            "mooncake": {
                "master": mooncake_master,
                "segment_gib": MOONCAKE_SEGMENT_GIB,
            }
        }
    }
    with _serve(
        deployments,
        WEIGHTS_SHARE,
        RECOMPUTED,
        remote,
        host_gib=SMALL_HOST_GIB,
    ) as server:
        _rollback_reads_back_evicted_blocks(server, LONG_CACHED_PROMPT)
