"""A deployment's lifecycle and live updates through the admin API.

Qwen3 serves on the GPUs each case leases, as the deployment both update
modules serve (:data:`uniserve_testing.updates.PROFILE`), with a Storage
Server of the case's own: a case that reads or restarts it names the runtime
directory it serves under (the ``runtime`` fixture).

- A deployment's storage belongs to it: once a deployment whose package
  states ``retire = true`` comes down (its daemon ends on SIGTERM), its
  Storage Server has stopped, releasing what it held of it, and a
  deployment whose host regions the host's available storage cannot hold
  does not come up, with the sizes that decided it.
- A serving deployment answers its admin API, on the daemon's admin socket:
  new requests wait while the package states ``paused`` and are served once
  it no longer does; a change of ``flush`` evicts the device copies of the
  cached blocks; scheduling limits change at the next step while a request
  streams, and a change of the request slots is an epoch change under the
  same stream, which commits once its trial (``update_config.trial``)
  passed; the manager applies a package to the node.
- A hot restart, which a change of the daemon's own options starts, hands
  the deployment and its responses in flight to a successor daemon, and a
  Storage Server restart keeps it serving.
- A target that finds no storage, a lost main process, a fault on trial and
  a regression on trial leave the requests in flight whole.

Every change is a policy package ``PUT /admin/policy`` applies
(:mod:`uniserve_testing.admin`): the options a case changes over the
deployment's current package. What the deployment executes and owns a case
reads from its observation store (``/admin/query``): the leader's gauges of
each replica and GPU quota, and the ``log`` events of each update phase.
The options of how an update proceeds (``update_config``) are the daemon's
own, so a case states them in the package it starts with.
- A sleeping deployment gives back its KV cache, holds new requests and
  recomputes the ones it ran once it wakes, and answers as before.
- The host tier keeps a prompt's blocks across flushes, devices, cache
  pressure and weight updates.
"""

from __future__ import annotations

import json
import signal
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import httpx
import pytest
from uniserve_testing import admin, chat, packages, updates
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.builds import BuildServer
from uniserve_testing.caching import cached_tokens
from uniserve_testing.deployments import (
    HOST_CACHE_GIB,
    Deployments,
    free_port,
    runtime_directory,
    storage_socket,
    storage_status,
)
from uniserve_testing.devices import Lease
from uniserve_testing.packages import Config, merged
from uniserve_testing.plugin import environment_program
from uniserve_testing.servers import Server, environment
from uniserve_testing.updates import Stream, numbers

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

#: How long a trial lasts in a case that does not act in it: the update
#: commits soon after its switch, so a case that changes the deployment
#: several times spends no GPU time waiting.
TRIAL = "5s"

#: A trial no case outlasts, for a case that ends it by a fault or a
#: threshold of its own and reads what the trial keeps meanwhile.
HELD_TRIAL = "10m"

#: A trial a case causes a fault in once it began, short enough that the
#: other updates of the same deployment commit soon after it.
FAULT_TRIAL = "60s"

#: The options every deployment of the module serves with.
OPTIONS = merged(updates.PROFILE, {"update_config": {"trial": TRIAL}})


#: A change of the options that only a new epoch executes: fewer request slots
#: than the default 128, which lay the epoch's device memory out anew.
CHANGE = {"max_running_requests": 64}

#: The options of a deployment whose KV cache takes the device storage its
#: share leaves, rather than a fixed token count.
SIZED_OPTIONS = {
    key: value for key, value in OPTIONS.items() if key != "kv_cache_config"
}

#: The count a case streams while it acts on the deployment: long enough to
#: be in flight across what the case does.
PROMPT = (
    "Count from 1 to 3000, separated by commas and spaces. "
    "Write only the numbers."
)

#: The completion tokens a streamed count may take, more than any count of
#: the module needs.
COUNT_TOKENS = 8192

#: A prompt whose blocks a second request finds cached.
SEA = "Describe the sea in one sentence. "

#: The count a recovery case streams across its kill, and asks for when it
#: compares answers: long enough that the kill and a switch land inside the
#: response, short enough to be compared whole.
RECOVERY_PROMPT = (
    "Count from 1 to 300, separated by commas and spaces. "
    "Write only the numbers."
)

#: The completion tokens a recovery case compares: the same prompt every
#: time, so the same numbers are the same answer to the same question.
RECOVERY_TOKENS = 256

#: The span a trial's step time and throughput are averaged over, in
#: seconds.
TRIAL_WINDOW_S = 5

#: How long a request a case sends to an idle deployment takes to be held or
#: admitted and to appear in the deployment's gauges, in seconds: the daemon
#: takes it within a step, the leader samples its gauges every second and
#: its schedulers report once a second. No event states a single request
#: before it ends.
ADMITTED_S = 60

#: How long a hot restart's successor takes to serve the admin socket and
#: record that it serves (a ``node`` log of state ``serving``) once the
#: change that restarted the daemon answered, in seconds.
TAKEOVER_S = 120

#: How long a Storage Server's successor takes to serve the socket after the
#: restart command answered, in seconds: the handover waits until no copy of
#: the deployment is in flight, which a step ends.
HANDOVER_S = 120

#: How long a standby takes to park, in seconds: it starts as an update's
#: target does, and the bound is the one a target's preparation has
#: (:func:`uniserve_testing.admin.wait_for_log`).
STANDBY_S = 900

#: How long the copies of a request's blocks to the host tier take to
#: complete after the request answered, in seconds. Nothing announces their
#: completion and a flush does not wait for them, so a case waits this long
#: before it flushes the device copies.
STORE_S = 2

#: The pause between the small requests that keep a trial measuring after
#: its count ended, in seconds: far fewer tokens per second than the count
#: computed, so the throughput falls past its threshold.
TRICKLE_S = 1

#: How long a trial takes to roll back by its throughput once its count
#: ended, in seconds: it decides on the window it reads at each report.
ROLLBACK_S = 24 * TRIAL_WINDOW_S

#: The bytes of a GiB.
GIB = 1 << 30


@pytest.fixture
def runtime(uniserve: Path, tmp_path: Path) -> Iterator[Path]:
    """The runtime directory of a case that reads or restarts its Storage
    Server, whose logs the case keeps under ``storage-server``.
    """  # noqa: D205
    with runtime_directory(uniserve, tmp_path / "storage-server") as directory:
        yield directory


def _serve(
    deployments: Deployments,
    name: str,
    options: Config = OPTIONS,
    runtime: Path | None = None,
    host_gib: int | None = HOST_CACHE_GIB,
) -> AbstractContextManager[Server]:
    """The deployment `name` of Qwen3 serving `name` with `options`."""
    return deployments.serve(
        merged(
            {
                "model": CHECKPOINTS["qwen3"].model,
                "served_model_name": name,
            },
            options,
        ),
        name=name,
        runtime=runtime,
        host_gib=host_gib,
    )


def _request(name: str, prompt: str, tokens: int) -> dict[str, Any]:
    """A greedy request for `prompt` without a thinking section."""
    return chat.body(name, prompt, tokens, **chat.GREEDY, **chat.NO_THINKING)


def _ask(base_url: str, name: str, prompt: str, tokens: int = 16) -> str:
    """The greedy answer to `prompt`."""
    return chat.content(chat.complete(base_url, _request(name, prompt, tokens)))


def _cached(base_url: str, name: str, prompt: str, tokens: int = 1) -> int:
    """The prompt tokens the admission of a request for `prompt` found
    cached, in any tier (:mod:`uniserve_testing.caching`).
    """  # noqa: D205
    return cached_tokens(
        chat.complete(base_url, _request(name, prompt, tokens))
    )


def _counting_to(last: int) -> str:
    """A prompt that asks for the numbers from 1 to `last`."""
    return (
        f"Count from 1 to {last}, separated by commas and spaces. "
        "Write only the numbers."
    )


def _count(base_url: str, name: str, prompt: str = PROMPT) -> Stream:
    """Stream the count `prompt` asks for."""
    request = _request(name, prompt, COUNT_TOKENS)
    return Stream(base_url, {**request, "stream": True})


def _streaming(server: Server, name: str) -> Stream:
    """Stream the count, once it carried its first deltas."""
    stream = _count(server.base_url, name)
    stream.wait_for(
        lambda streamed: len(streamed.deltas) >= 32, "the count did not stream"
    )
    return stream


def _start_count(server: Server, name: str) -> Stream:
    """Stream the count, once it wrote its first numbers."""
    stream = _count(server.base_url, name)
    stream.wait_for(
        lambda streamed: len(numbers(streamed.text)) >= 8,
        "the count did not stream",
    )
    return stream


def _assert_whole(stream: Stream) -> int:
    """Wait for the count's end; it lost and repeated nothing, and finished
    once. Returns how many numbers it wrote.
    """  # noqa: D205
    stream.join()
    assert stream.error is None, f"the count failed: {stream.error!r}"
    counted = numbers(stream.text)
    wrong = next(
        (index for index, number in enumerate(counted) if number != index + 1),
        None,
    )
    if wrong is not None:
        text = stream.text
        at = text.find(f"{wrong}, ")
        pytest.fail(
            f"the response lost or repeated a count at {wrong}: "
            f"{text[max(at - 60, 0) : at + 80]!r}, deltas "
            f"{stream.deltas[max(3 * wrong - 12, 0) : 3 * wrong + 12]}"
        )
    assert len(stream.reasons) == 1, f"the count ended {stream.reasons}"
    return len(counted)


def _assert_counts_from_one(stream: Stream, count: int) -> None:
    """Wait for the count's end; it failed nothing and wrote its first
    `count` numbers in order.
    """  # noqa: D205
    stream.join()
    assert stream.error is None, f"the count failed: {stream.error!r}"
    written = numbers(stream.text)
    assert written[:count] == list(range(1, count + 1)), stream.text[:400]


def _apply(server: Server, changes: Config) -> dict:
    """Apply `changes` over the deployment's package; the revision's record
    once the deployment converged to it: an update's once its trial passed
    and it committed.
    """  # noqa: D205
    return admin.revision(admin.change(server, changes))


def _change_epoch(server: Server, changes: Config) -> dict:
    """Apply `changes` as an epoch switch, which commits once its trial
    passed; the update as its last phase's ``update`` log states it then.
    """  # noqa: D205
    record = _apply(server, changes)
    assert record["effect"] == "epoch", record
    update = admin.logs(server, "update")[-1].fields
    assert update["phase"] == "committed", update
    return update


def _restarted(response: httpx.Response) -> dict:
    """The record of a change whose daemon options a hot restart applies:
    answered ``202`` once the restart began, its successor converging the
    rest.
    """  # noqa: D205
    assert response.status_code == 202, response.text
    record = response.json()
    assert record["effect"] == "daemon", record
    assert record["result"] == "converging", record
    return record


def _successor(
    server: Server, previous: int, after: int, successors: list[int]
) -> int:
    """The daemon that took the deployment over from daemon `previous` in a
    hot restart a change applied after the moment `after` started, once it
    serves; appended to `successors` as soon as it serves the admin socket.

    The change answers ``202`` as soon as the restart began. The successor
    binds the admin socket over its predecessor's, which names it
    (:func:`uniserve_testing.admin.daemon_pid`), and records that it serves
    the deployment as it begins to serve (a ``node`` log of state
    ``serving``).
    """  # noqa: D205

    def bound() -> int | None:
        try:
            pid = admin.daemon_pid(server)
        except OSError:
            return None
        return None if pid == previous else pid

    server.wait_until(
        lambda: bound() is not None,
        TAKEOVER_S,
        f"no daemon took the admin socket over from {previous}",
    )
    pid = bound()
    assert pid is not None
    successors.append(pid)
    admin.wait_for_log(
        server,
        "node",
        after=after,
        timeout_s=TAKEOVER_S,
        what=f"daemon {pid} did not serve the deployment",
        state="serving",
    )
    return pid


def _restart(server: Server, successors: list[int]) -> tuple[int, dict]:
    """Restart the daemon in place: a change of one of its own options
    (``logging_config.log_requests``, turned over), answered ``202``; the
    successor once it serves (:func:`_successor`) and the change's record.
    """  # noqa: D205
    previous = admin.daemon_pid(server)
    after = admin.now(server)
    logging = admin.policy(server).config.get("logging_config", {})
    stated = logging.get("log_requests", False)
    record = _restarted(
        admin.change(server, {"logging_config": {"log_requests": not stated}})
    )
    return _successor(server, previous, after, successors), record


def _held_requests(server: Server) -> int:
    """The new requests the daemon holds before an engine takes them."""
    return int(
        sum(
            sample.value
            for sample in admin.gauge(server, "uniserve_requests_held")
        )
    )


def _epochs(sample: admin.Sample) -> set[int]:
    """The epochs a reservation's gauge names (its ``epochs`` label, the
    epoch numbers joined by commas).
    """  # noqa: D205
    return {int(epoch) for epoch in sample.labels["epochs"].split(",") if epoch}


def _reservations(
    server: Server, after: int | None = None
) -> list[admin.Sample]:
    """Every reservation of every GPU quota of the deployment, as the leader
    last sampled them (in a second that began after the moment `after`,
    when stated).
    """  # noqa: D205
    return admin.gauge(server, "uniserve_gpu_reserved_bytes", after=after)


def _bytes_of(
    server: Server,
    item: str,
    epoch: int | None = None,
    after: int | None = None,
) -> int:
    """The bytes the deployment's quotas reserve for `item` (of `epoch`'s
    reservations, when stated), sampled after the moment `after` when
    stated.
    """  # noqa: D205
    return int(
        sum(
            sample.value
            for sample in _reservations(server, after)
            if sample.labels["item"] == item
            and (epoch is None or epoch in _epochs(sample))
        )
    )


def _engine_pids(server: Server) -> set[int]:
    """The process ids of the deployment's main processes that run."""
    return {
        process["pid"]
        for process in admin.processes(server, group="engine")
        if process["state"] not in admin.ENDED
    }


def _failed_requests(server: Server) -> float:
    """The requests that ended in an error so far."""
    return admin.counter(server, "uniserve_requests_total", outcome="error")


def _flush(server: Server) -> None:
    """Evict the device copies of every cached block."""
    admin.flush(server)


def _held_of(listing: dict[str, Any], name: str) -> dict[str, list]:
    """What a Storage Server's listing holds of the deployment `name`: its
    claim, its tensors, its quotas and its buffer pool.
    """  # noqa: D205
    return {
        "deployments": [
            entry
            for entry in listing["deployments"]
            if entry["deployment"] == name
        ],
        "tensors": [
            tensor
            for tensor in listing["tensors"]
            if any(scope["deployment"] == name for scope in tensor["scopes"])
        ],
        "quotas": [
            quota for quota in listing["quotas"] if quota["deployment"] == name
        ],
        "buffer_pools": [
            pool
            for pool in listing["buffer_pools"]
            if pool["deployment"] == name
        ],
    }


def test_a_deployment_releases_its_storage_and_one_too_large_does_not_come_up(
    deployments: Deployments, uniserve: Path, runtime: Path
):
    name = "qwen3-live-storage-owner"
    with _serve(deployments, name, runtime=runtime) as server:
        assert _ask(server.base_url, name, "Say hello.")

        # The deployment's Storage Server holds the deployment: its claim and
        # the weights its quota reserves.
        assert storage_status(uniserve, runtime, name) is not None
        held = _held_of(updates.listing(uniserve, runtime), name)
        assert held["deployments"], held
        reserved = {
            entry["item"]
            for quota in held["quotas"]
            for entry in quota["reservations"]
        }
        assert "weights" in reserved, held
        # The host's available storage, as the Storage Server reads it.
        (usage,) = admin.gauge(server, "uniserve_host_available_bytes")
        available = int(usage.value)

        # A deployment that retires releases its storage as it comes down,
        # which its daemon does on SIGTERM: the daemon, whose node serves this
        # deployment alone, stops the deployment's Storage Server, which
        # released its weights and caches with it, and ends.
        retiring = admin.revision(admin.change(server, {"retire": True}))
        assert retiring["effect"] == "step", retiring
        server.process.send_signal(signal.SIGTERM)
        assert server.process.wait() == 0, "the daemon ended in an error"
    assert storage_status(uniserve, runtime, name) is None, (
        "the retired deployment's Storage Server outlived it"
    )

    # A host tier the host's available storage cannot hold, under the same
    # runtime directory: the deployment's own Storage Server reads the
    # host's available storage, the deployment does not come up, and its
    # daemon's log states the sizes that decided it.
    too_large = "qwen3-live-too-large"
    with (
        pytest.raises(RuntimeError, match="ended before it was serving"),
        _serve(
            deployments,
            too_large,
            runtime=runtime,
            host_gib=2 * available // GIB,
        ),
    ):
        pass
    log = (deployments.directory / f"{too_large}.log").read_text()
    refusal = [
        line
        for line in log.splitlines()
        if "available" in line and "reserve" in line
    ]
    print(f"refused: {refusal}")
    assert refusal, log[-4000:]


def test_a_serving_deployment_answers_its_admin_api(deployments: Deployments):
    name = "qwen3-live-admin"
    # The deployment keeps no host tier: what a request finds cached is on the
    # device, so the tokens its answer states as cached say what a flush
    # evicted.
    options = merged(OPTIONS, {"kv_cache_config": {"host_tier": False}})
    with _serve(deployments, name, options=options) as server:
        node = admin.wait_for_log(
            server, "node", after=0, timeout_s=ADMITTED_S, state="serving"
        )
        assert node.fields["deployment"] == name, node
        replica = admin.replica(server)
        assert replica.state == "serving", replica

        # New requests wait while the package states generation paused.
        paused = admin.revision(admin.change(server, {"paused": True}))
        assert paused["effect"] == "step", paused
        answered: list[str] = []
        waiting = threading.Thread(
            target=lambda: answered.append(
                _ask(server.base_url, name, "Say hello.")
            )
        )
        waiting.start()
        server.wait_until(
            lambda: _held_requests(server) >= 1,
            ADMITTED_S,
            "the daemon reported no held request while generation paused",
        )
        assert not answered, "a request was served while generation paused"
        admin.revision(admin.change(server, {"paused": False}))
        waiting.join()
        assert answered and answered[0], answered

        # A flush evicts the device copies of the blocks the same prompt
        # committed: asked again, it finds none of them on the device.
        prompt = SEA * 40
        _ask(server.base_url, name, prompt)
        cached = _cached(server.base_url, name, prompt)
        assert cached > 0, cached
        _flush(server)
        after_flush = _cached(server.base_url, name, prompt)
        assert after_flush == 0, after_flush

        # Scheduling limits change at the next step under a streaming
        # request, and the request slots by an epoch change under the same
        # stream.
        stream = _streaming(server, name)
        limits = _apply(server, {"scheduler_config": {"max_batch": 4}})
        assert limits["effect"] == "step", limits
        before_update = len(stream.deltas)
        report = _change_epoch(server, CHANGE)
        print(f"epoch switch: {report['switch_ms']} ms")
        _assert_counts_from_one(stream, 100)
        assert len(stream.deltas) > before_update, (
            "the stream ended before the update"
        )

        # The manager attaches to the node by its admin socket and applies a
        # package directory to it. It names the node by its address at the
        # head of each line it prints, and prints what the node answers as
        # JSON after it.
        manager = str(environment_program("uniserve-manager"))
        node = server.endpoint("admin")
        current = admin.policy(server)
        package = packages.write(
            deployments.directory / "manager-package",
            merged(current.config, {"scheduler_config": {"max_batch": 8}}),
        )
        applied = subprocess.run(
            [manager, "--node", node, "apply", str(package)],
            env=environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        record = json.loads(applied.removeprefix(f"{node}: "))
        assert record["result"] == "converged", record
        assert record["revision"] == current.revision + 1, record
        served = admin.policy(server)
        assert served.revision == record["revision"], served
        assert served.config["scheduler_config"]["max_batch"] == 8, served
        assert served.config["served_model_name"] == name, served

        # The deployment's revision events record the change the manager
        # applied.
        recorded = [
            event.fields.get("revision")
            for event in admin.events(server, "revision")
        ]
        assert record["revision"] in recorded, recorded
        assert _ask(server.base_url, name, "Say hello.")


#: The streams a hot restart hands over: three counts of different lengths,
#: so their event boundaries fall at different points of the response, and
#: the number each stream must carry with no gap and no repetition.
HANDED_STREAMS = (200, 500, 900)

#: The count a restart hands over with a stop string, and the number whose
#: text stops it: the engine holds a request with stop strings for the
#: daemon's acknowledgements, so the successor must go on acknowledging its
#: tokens for the count to reach the stop string well after the restart.
STOPPED_STREAM = 900
STOP_AT = 800

#: The prompt of the one-body request a restart hands over: long enough that
#: its generation is certainly in flight when the restart pauses it.
JSON_PROMPT = _counting_to(1500)


def test_a_hot_restart_hands_the_deployment_to_a_successor_daemon(
    deployments: Deployments,
):
    name = "qwen3-live-restart"
    with (
        _serve(deployments, name) as server,
        updates.successors() as successors,
    ):
        _restart_while_streaming(server, name, server.build_server, successors)


def test_a_switch_after_a_hot_restart_finds_the_storage_of_the_epoch_before(
    deployments: Deployments, repository: Path
):
    """A successor daemon learns what a target maps from the process that serves.

    The epoch that serves took its storage from the one before it, under that
    epoch's name. The daemon that switched it is gone, and the next target still
    maps that storage.
    """  # noqa: E501
    name = "qwen3-live-restart-switch"
    with (
        _serve(deployments, name) as server,
        updates.successors() as successors,
    ):
        revisions = [
            updates.engine_revision(server.build_server, repository)
            for _ in range(2)
        ]
        answer = _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32)
        _change_epoch(server, revisions[0])
        _restart(server, successors)
        _ask(server.base_url, name, RECOVERY_PROMPT, tokens=8)
        _change_epoch(server, revisions[1])
        assert _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32) == (
            answer
        )


def _restart_while_streaming(
    server: Server,
    name: str,
    build_server: BuildServer | None,
    successors: list[int],
) -> None:
    engines_before = _engine_pids(server)
    replica = admin.replica(server)
    assert replica.state == "serving", replica
    first = admin.now(server)

    # Three streams in flight at the restart, of different lengths so their
    # boundaries fall at different events of different requests.
    streams = [
        _count(server.base_url, name, _counting_to(last))
        for last in HANDED_STREAMS
    ]
    # One more, which a stop string ends after the restart.
    stopped = Stream(
        server.base_url,
        {
            **_request(name, _counting_to(STOPPED_STREAM), COUNT_TOKENS),
            "stream": True,
            "stop": [f"{STOP_AT},"],
        },
    )
    for stream in [*streams, stopped]:
        stream.wait_for(
            lambda streamed: len(streamed.deltas) >= 32,
            "the counts did not stream",
        )
    successor, _ = _restart(server, successors)
    assert successor != server.pid
    before_restart = [len(stream.deltas) for stream in streams]
    stopped_before = numbers("".join(stopped.deltas))

    # The successor serves the same deployment, so it holds the same modules'
    # server: a deployment reaches its scheduling policy only through it, and
    # the successor serves the package the old process served rather than a
    # command line of its own (design 7.5).
    assert build_server is not None
    inherited = admin.policy(server).config
    assert (
        inherited["build_config"]["package_root"]
        == build_server.settings["package_root"]
    ), f"the successor did not inherit the modules' server: {inherited}"

    # New requests are served while the old daemon finishes its streams.
    for _ in range(3):
        assert _ask(server.base_url, name, "Say hello."), (
            "a new request after the restart answered an empty completion"
        )
    for index, (stream, last) in enumerate(
        zip(streams, HANDED_STREAMS, strict=True)
    ):
        stream.join()
        assert stream.error is None, f"stream {index} failed: {stream.error!r}"
        assert len(stream.deltas) > before_restart[index], (
            f"stream {index} ended at the restart ({len(stream.deltas)} "
            f"deltas, {before_restart[index]} before it, last={last})"
        )
        # Every stream carries its count whole: a frame lost, repeated or
        # reordered at the boundary — the one thing the handoff must not do —
        # breaks the sequence, and the boundary lies inside it, not after the
        # first two hundred numbers.
        counted = numbers(stream.text)
        assert counted == list(range(1, len(counted) + 1)), (
            f"stream {index} broke: {stream.text[:400]}"
        )
        # `before_restart` counts deltas, and a delta need not hold a number.
        before = numbers("".join(stream.deltas[: before_restart[index]]))
        assert len(counted) > len(before), (
            f"stream {index} carried no number after the restart"
        )
        assert counted and counted[-1] >= last // 2, (
            f"stream {index} stopped at {counted[-1] if counted else 0}"
        )
        # It finished rather than being closed: the successor's last chunk
        # names why it stopped.
        assert stream.reasons, f"stream {index} ended without a finish reason"
        assert stream.reasons[-1] in ("stop", "length"), stream.reasons

    # The count with a stop string ends at it, under the successor: every
    # number before it, whole, and a finish by the stop string.
    stopped.join()
    assert stopped.error is None, f"the stopped count failed: {stopped.error!r}"
    assert len(stopped_before) < STOP_AT - 1, stopped_before[-1:]
    counted = numbers(stopped.text)
    assert counted == list(range(1, STOP_AT)), (
        f"the stopped count broke: {stopped.text[-400:]}"
    )
    assert stopped.reasons == ["stop"], stopped.reasons

    # A one-body chat response in flight at a further restart is written by
    # its successor too, whole (design 7.5.1: every response in flight moves).
    restarted = _restart_with_a_json_body(server, name, successors)

    # The daemon the case started exits once its requests ended; the engine
    # and its workers are the same processes throughout. The one-body
    # restart replaced `successor` in turn, so the daemon that serves now is
    # the last of them, on the same address.
    server.process.wait()
    assert admin.daemon_pid(server) == successors[-1]
    assert _ask(server.base_url, name, "Say hello.")
    replica = admin.replica(server)
    assert replica.state == "serving", replica
    assert _engine_pids(server) == engines_before
    # Each successor records the deployment it serves and how the change
    # that restarted its predecessor ended: the revision converged.
    serving = admin.logs(server, "node", after=first)
    assert any(event.fields.get("state") == "serving" for event in serving), (
        serving
    )
    admin.wait_for(
        server,
        "revision",
        lambda event: (
            event.fields.get("revision") == restarted["revision"]
            and event.fields.get("result") == "converged"
        ),
        after=first,
        timeout_s=TAKEOVER_S,
        what="the successor did not record the restart's revision converged",
    )

    # The successor restarts in turn, and takes a change at the next step.
    _restart(server, successors)
    updates.wait_for_end(successor)
    assert _ask(server.base_url, name, "Say hello.")
    # A scheduling limit takes effect at a scheduler's next step;
    # `max_running_requests` sizes what the engine is started for and would
    # switch the epoch.
    limits = _apply(server, {"scheduler_config": {"max_batch": 4}})
    assert limits["effect"] == "step", limits

    # A change of the listening address restarts the daemon on it: the new
    # address serves, and the old one closes once the replaced daemon exits.
    previous = successors[-1]
    after = admin.now(server)
    port = free_port()
    moved = _restarted(
        admin.change(
            server, {"admin_config": {"host": "localhost", "port": port}}
        )
    )
    _successor(server, previous, after, successors)
    # The successor binds the new address once it holds the deployment, and
    # records the revision converged.
    moved_url = f"http://localhost:{port}"
    assert _ask(moved_url, name, "Say hello.")
    updates.wait_for_end(previous)
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{server.base_url}/health", timeout=5)
    served = admin.policy(server).config["admin_config"]
    assert served["host"] == "localhost", served
    server.wait_until(
        lambda: admin.policy(server).revision == moved["revision"],
        TAKEOVER_S,
        "the successor did not record the address's revision converged",
    )

    # A change of both the daemon's options and what an epoch executes
    # restarts the daemon on its new options first, and its successor then
    # changes the epoch in an update of its own, which commits once its
    # trial passed.
    previous = successors[-1]
    before = admin.now(server)
    port = free_port()
    both = _restarted(
        admin.change(
            server,
            merged(
                {"admin_config": {"host": "127.0.0.1", "port": port}}, CHANGE
            ),
        )
    )
    _successor(server, previous, before, successors)
    both_url = f"http://127.0.0.1:{port}"
    ended = admin.wait_for_log(
        server,
        "update",
        after=before,
        phase=("committed", "abandoned", "rolled_back"),
    )
    update = ended.fields
    assert update["phase"] == "committed", update
    assert update["change"] == "epoch", update
    # The revision becomes current once the deployment converged to it, and
    # the update the successor started is numbered by it.
    server.wait_until(
        lambda: admin.policy(server).revision == both["revision"],
        TAKEOVER_S,
        "the successor did not record the change's revision converged",
    )
    admin.wait_for(
        server,
        "revision",
        lambda event: (
            event.fields.get("revision") == both["revision"]
            and event.fields.get("result") == "converged"
        ),
        after=before,
        timeout_s=TAKEOVER_S,
        what="the successor recorded no converged revision of the change",
    )
    assert ended.labels["update"] == str(both["revision"]), (ended, both)
    # The configuration is the target's from then on.
    served = admin.policy(server).config
    assert served["max_running_requests"] == 64, served
    assert served["admin_config"]["host"] == "127.0.0.1", served
    assert _ask(both_url, name, "Say hello.")
    updates.wait_for_end(previous)


def _restart_with_a_json_body(
    server: Server, name: str, successors: list[int]
) -> dict:
    """A non-streaming chat body in flight across a restart is written whole;
    the record of the change that restarted the daemon.

    The body is one response built from a collector's state, so a field lost in
    the crossing is silent: the request still answers 200 with a response that
    parses. What catches it is a body that finished, whose text is a contiguous
    count and whose usage adds up.
    """  # noqa: D205
    bodies: list[dict] = []
    failures: list[BaseException] = []

    def collect() -> None:
        try:
            bodies.append(
                chat.complete(
                    server.base_url, _request(name, JSON_PROMPT, COUNT_TOKENS)
                )
            )
        except BaseException as error:  # noqa: BLE001 - a failure is the case's
            failures.append(error)

    request = threading.Thread(target=collect)
    request.start()
    # The restart pauses what the engine serves, so it is issued only once the
    # one-body request is being served: the schedulers' gauge says so, and no
    # event states a single request before it ends.
    server.wait_until(
        lambda: (
            sum(
                sample.value
                for sample in admin.gauge(
                    server, "uniserve_scheduler_running_requests"
                )
            )
            >= 1
        ),
        ADMITTED_S,
        "the one-body request was never served",
    )
    _, restarted = _restart(server, successors)
    request.join()
    assert not failures, failures
    assert bodies, "the one-body request answered nothing"
    body = bodies[0]
    assert body["choices"][0]["finish_reason"] not in (None, "migratable"), body
    text = chat.content(body)
    counted = numbers(text)
    assert counted == list(range(1, len(counted) + 1)), text[:400]
    assert len(counted) > 100, text[:400]
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0, usage
    assert usage["completion_tokens"] >= len(counted), usage
    assert (
        usage["total_tokens"]
        == usage["prompt_tokens"] + usage["completion_tokens"]
    ), usage
    return restarted


def _storage_pid(uniserve: Path, runtime: Path) -> int | None:
    """The process that serves the Storage Server's socket under `runtime`,
    as its readiness record states it; ``None`` while none answers.
    """  # noqa: D205
    record = storage_status(uniserve, runtime)
    return None if record is None else record["pid"]


def _held(listing: dict[str, Any]) -> dict[str, Any]:
    """What a Storage Server holds: its tensors, deployments and the quotas
    of its deployments.

    Its Transfer Server's process, contexts, link measurements and tasks are
    no storage it holds: a restarted server's Transfer Server measures anew
    and reserves its context only while it has work on a GPU. Its buffer
    pools' blocks, groups and copies change with every step of the requests
    the deployment serves.
    """  # noqa: D205
    return {
        "tensors": listing["tensors"],
        "deployments": listing["deployments"],
        "quotas": [
            {
                "gpu": quota["gpu"],
                "deployment": quota["deployment"],
                "limit": quota["limit"],
                "reservations": [
                    entry
                    for entry in quota["reservations"]
                    if entry["item"] != "transfer_context"
                ],
            }
            for quota in listing["quotas"]
            if quota["deployment"] is not None
        ],
    }


def test_a_storage_server_restart_keeps_the_deployment_serving(
    deployments: Deployments, uniserve: Path, runtime: Path
):
    name = "qwen3-live-storage"
    with _serve(deployments, name, runtime=runtime) as server:
        prompt = SEA * 40
        _ask(server.base_url, name, prompt)
        before = _held(updates.listing(uniserve, runtime))
        old_server = _storage_pid(uniserve, runtime)

        stream = _streaming(server, name)
        subprocess.run(
            [
                str(uniserve),
                "storage-server",
                "restart",
                "--socket",
                str(storage_socket(runtime)),
            ],
            env=environment(),
            check=True,
            capture_output=True,
            text=True,
        )
        # The successor takes over once nothing is in flight, and no event
        # announces it: the socket's readiness record names it once it
        # serves there.
        server.wait_until(
            lambda: _storage_pid(uniserve, runtime) not in (None, old_server),
            HANDOVER_S,
            "the successor Storage Server did not take over",
        )
        assert _held(updates.listing(uniserve, runtime)) == before

        # The deployment serves throughout: the count goes on, a new
        # request is answered, and an epoch change maps the weights the
        # successor holds.
        assert _ask(server.base_url, name, prompt)
        _assert_counts_from_one(stream, 200)
        weights = _bytes_of(server, "weights")
        _change_epoch(server, CHANGE)
        committed = admin.now(server)
        assert _ask(server.base_url, name, "Say hello.")
        # The quota reserves a storage once however many epochs map it, so a
        # target that read the checkpoint again would reserve its weights
        # beside these.
        assert _bytes_of(server, "weights", after=committed) == weights


#: The bytes the ballast of a full GPU leaves free: room for a target's CUDA
#: context, so its start reaches its storage, and not for its graph pool.
TARGET_ROOM = 2 * GIB


def test_a_target_that_finds_no_storage_leaves_the_serving_epoch_serving(
    deployments: Deployments, lease: Lease
):
    """A target beside a full GPU ends the update and changes nothing that serves.

    A target prepares beside the serving epoch, which keeps its regions until
    it drains: the target starts with what the GPU has free. With another
    process holding all of it but ``TARGET_ROOM``, room for the target's CUDA
    context and no more, the target finds no room for its graph pool, so its
    start fails, the update is abandoned before any
    switch (the ``PUT`` answers ``409`` with the reason, the update's last
    ``update`` log states it abandoned and the revision stays the current
    one, ``docs/admin.md``), and the stream goes on with the serving epoch's
    regions as they were.
    """  # noqa: E501
    name = "qwen3-live-pressure"
    prompt = "Describe the sea in one sentence."
    with _serve(deployments, name, options=SIZED_OPTIONS) as server:
        before = _ask(server.base_url, name, prompt, 32)
        serving = admin.replica(server).epoch
        current = admin.policy(server).revision
        cache = _bytes_of(server, "cache_segment", serving)
        stream = _streaming(server, name)
        with updates.ballast(lease, leave=TARGET_ROOM) as held:
            print(f"ballast holds {held / GIB:.1f} GiB")
            # The change answers once the update ended: abandoned, with 409
            # and its reason.
            response = admin.change(server, CHANGE)
            assert response.status_code == 409, response.text
            update = admin.logs(server, "update")[-1].fields
            print("epoch switch under pressure:", json.dumps(update))
            # The target's start failed before it joined: a failed
            # preparation, whose reason names the storage it did not find.
            assert update["phase"] == "abandoned", update
            assert update["failure"]["kind"] == "prepare_failed", update
            assert "graph pool" in update["failure"]["reason"], update
        _assert_counts_from_one(stream, 200)
        # The serving epoch's quota reserves at least the KV cache it held,
        # and its revision is still the current one.
        ended = admin.now(server)
        assert admin.replica(server).epoch == serving
        assert admin.policy(server).revision == current
        assert _bytes_of(server, "cache_segment", serving, ended) >= cache, (
            "the abandoned update shrank the serving epoch's KV regions"
        )
        assert _ask(server.base_url, name, prompt, 32) == before


def test_a_sleeping_deployment_gives_back_its_cache_and_graph_storage(
    deployments: Deployments,
):
    name = "qwen3-live-sleep"
    prompt = "Describe the sea in one sentence."
    with _serve(deployments, name, options=SIZED_OPTIONS) as server:
        before = _ask(server.base_url, name, prompt, 32)
        epoch = admin.replica(server).epoch
        awake = _bytes_of(server, "cache_segment", epoch)
        assert awake > 0, _reservations(server)
        stream = _streaming(server, name)

        slept = admin.revision(admin.change(server, {"asleep": True}))
        assert slept["effect"] == "step", slept
        assert admin.policy(server).config["asleep"] is True
        # The workers give back every segment of the KV cache, which the
        # GPU's quota reserves no more.
        asleep_at = admin.now(server)
        asleep = _bytes_of(server, "cache_segment", epoch, asleep_at)
        assert asleep == 0, _reservations(server)
        answered: list[str] = []
        waiting = threading.Thread(
            target=lambda: answered.append(
                _ask(server.base_url, name, prompt, 32)
            )
        )
        waiting.start()
        server.wait_until(
            lambda: _held_requests(server) >= 1,
            ADMITTED_S,
            "the daemon reported no held request while the engine slept",
        )
        assert not answered, "a sleeping deployment served a request"

        admin.revision(admin.change(server, {"asleep": False}))
        woke_at = admin.now(server)
        waiting.join()
        assert answered and answered[0], answered
        _assert_counts_from_one(stream, 200)
        # The wake creates the KV cache again, as large as it was.
        woken = _bytes_of(server, "cache_segment", epoch, woke_at)
        print(f"KV cache: awake {awake} B, asleep {asleep} B, woken {woken} B")
        assert woken == awake, (awake, woken)
        # Alone, as before the sleep, the deployment answers as it did: the
        # request that waited ran beside the recomputed count, whose batch
        # changes the numerics.
        assert _ask(server.base_url, name, prompt, 32) == before
        # The policy read back leaves an option at its default out, so the
        # woken deployment's states neither.
        config = admin.policy(server).config
        assert config.get("asleep", False) is False, config
        assert config.get("paused", False) is False, config


def _prepared_standby(server: Server) -> list[int]:
    """The launch generations of the parked standby, which parks once it
    captured its graphs; empty while there is none.
    """  # noqa: D205
    return admin.replica(server).standby


def _wait_for_standby(
    server: Server, replaced: list[int] | None = None
) -> list[int]:
    """Wait until a standby parked, other than `replaced`; its launch
    generations.

    The leader's gauges state the standby's launch generations once it
    parked (``uniserve_replica_standby_generation``), which are read every
    second.
    """  # noqa: D205
    server.wait_until(
        lambda: _prepared_standby(server) not in ([], replaced),
        STANDBY_S,
        "the deployment reported no prepared standby",
    )
    return _prepared_standby(server)


def _lose_the_serving_engine(
    server: Server, name: str, case: str, standby: bool
) -> dict:
    """Kills the engine that serves under a request in flight.

    The request in flight is a count: it must carry every number it wrote
    before the loss and every number after it, with no gap and no repetition,
    because the response migrates to the engine that takes over. The answer to
    the same question after the recovery must be the answer the deployment gave
    before it — that is what says the surviving engine serves the same model and
    not merely a live one.
    """
    if standby:
        # The standby is what takes over, so the loss waits for it to stand
        # by with its graphs.
        _wait_for_standby(server)
    answer = chat.complete(
        server.base_url, _request(name, RECOVERY_PROMPT, RECOVERY_TOKENS)
    )
    baseline = chat.content(answer)
    baseline_numbers = numbers(baseline)
    # The prompt's own token count is in the answer, and it is what separates a
    # deployment that reads the question wrongly from one that answers wrongly:
    # this sentence is about twenty tokens long, so a `prompt_tokens` of a
    # handful says the request never reached the model as text it could read.
    assert baseline_numbers[:10] == list(range(1, 11)), (
        f"{baseline[:200]!r} usage={answer.get('usage')}"
    )

    stream = _count(server.base_url, name, RECOVERY_PROMPT)
    stream.wait_for(
        lambda streamed: len(numbers(streamed.text)) >= 8,
        "the count did not stream",
    )
    before = len(numbers(stream.text))
    serving = admin.replica(server)
    standby_generations = serving.standby
    killed = admin.serving_engine(server)["pid"]
    started = time.monotonic()
    updates.terminate(killed)
    stream.wait_for(
        lambda streamed: len(numbers(streamed.text)) > before,
        f"the response in flight did not continue after the engine {killed} "
        "ended",
    )
    stream.join()

    # What the client lived through: the longest silence of the stream from the
    # last token before the loss on. The arrival of each token is its own
    # reading, so this is exact to the token.
    last_before = [moment for moment in stream.arrivals if moment < started]
    after = [moment for moment in stream.arrivals if moment >= started]
    spans = (last_before[-1:] or [started]) + after
    recovery_s = max(
        later - earlier for earlier, later in zip(spans, spans[1:])
    )
    # The measurement is taken before the tokens are judged: a run whose served
    # path is broken still states how long the recovery took, which is the one
    # number this case exists to record.
    evidence_time = round(recovery_s, 3)
    assert stream.error is None, f"the count failed: {stream.error!r}"
    counted = numbers(stream.text)
    assert counted == list(range(1, len(counted) + 1)), (
        f"the response lost or repeated a count across the loss: {counted[:40]}"
    )
    assert len(counted) >= 300, f"the count stopped at {len(counted)} numbers"
    assert stream.reasons == ["stop"], f"the count ended {stream.reasons}"

    again = _ask(server.base_url, name, RECOVERY_PROMPT, RECOVERY_TOKENS)
    assert numbers(again) == baseline_numbers, (
        f"the engine that serves after the loss answers differently: "
        f"{again[:200]}"
    )

    # The main process that serves now: the standby's, promoted, or one
    # launched anew.
    generation = admin.replica(server).generation
    promoted = bool(standby_generations) and (
        generation == standby_generations[0]
    )
    relaunched = generation not in (
        None,
        serving.generation,
        *standby_generations,
    )
    assert promoted or relaunched, "no engine took the lost one's place"
    expected = promoted if standby else relaunched
    assert expected, (
        f"a deployment with standby={standby} recovered by "
        f"promoted={promoted} relaunched={relaunched}"
    )

    evidence = {
        "case": case,
        "standby": standby,
        "engine_lost": killed,
        "recovery_s": evidence_time,
        "promoted": promoted,
        "relaunched": relaunched,
        "numbers_across_the_loss": len(counted),
        "answer_matches_the_answer_before": True,
    }
    print(f"{case}: {json.dumps(evidence, sort_keys=True)}")
    return evidence


def test_a_lost_main_process_is_promoted_from_its_standby(
    deployments: Deployments,
):
    """A deployment with a standby loses its serving process and serves on.

    The standby stands beside it for exactly this: it is promoted in place,
    with the graphs it captured at their addresses, so the requests in flight
    and the ones after them are answered by the process that was already
    parked rather than by one that starts now.
    """
    name = "qwen3-live-standby"
    options = merged(OPTIONS, {"worker_config": {"standby": True}})
    with _serve(deployments, name, options=options) as server:
        evidence = _lose_the_serving_engine(
            server, name, "standby-promoted", standby=True
        )
        assert evidence["promoted"]


def test_a_lost_main_process_is_relaunched_without_a_standby(
    deployments: Deployments,
):
    """The same loss with no standby beside the serving process.

    The daemon launches a replacement and hands it the epoch's own modules and
    the requests in flight: the count migrates to the new process, which answers
    the next request as the lost one did.
    """
    name = "qwen3-live-relaunched"
    with _serve(deployments, name) as server:
        evidence = _lose_the_serving_engine(
            server, name, "relaunched", standby=False
        )
        assert evidence["relaunched"]


def test_a_standby_started_after_a_switch_maps_what_its_epoch_serves_from(
    deployments: Deployments, repository: Path
):
    """A standby of a target maps storage another epoch created.

    A target serves from the storage of the epoch before it, under that epoch's
    name. The standby started beside the target finds it under the same name,
    and takes over when the target's process is lost.
    """
    name = "qwen3-live-standby-switch"
    options = merged(OPTIONS, {"worker_config": {"standby": True}})
    with _serve(deployments, name, options=options) as server:
        revision = updates.engine_revision(server.build_server, repository)
        first = _wait_for_standby(server)
        _change_epoch(server, revision)
        _wait_for_standby(server, replaced=first)
        evidence = _lose_the_serving_engine(
            server, name, "standby-after-a-switch", standby=True
        )
        assert evidence["promoted"]


def _fail_the_target(server: Server, after: int) -> dict:
    """End the update on trial by a fault: the target's serving main process
    ends, which rolls a trial back whatever its thresholds; the update once
    it entered ``rolled_back``, which serves its kept side again. `after` is
    the daemon's clock the case read before it applied the change.
    """  # noqa: D205
    trial = admin.wait_for_log(server, "update", after=after, phase="trial")
    # The target serves from its switch on, which the trial follows.
    updates.terminate(admin.serving_engine(server, after=trial.time)["pid"])
    return admin.wait_for_log(
        server, "update", after=after, phase="rolled_back"
    ).fields


def _rolled_back(response: httpx.Response) -> None:
    """The answer of a change its trial rolled back: refused, with the
    reason, and no revision converged.
    """  # noqa: D205
    assert response.status_code == 409, response.text
    print("rolled back:", response.text)


def test_a_relaunched_epoch_hands_its_own_storage_to_the_next_target(
    deployments: Deployments, repository: Path
):
    """A process started again keys the storage it creates by its own epoch.

    The epoch that replaces it maps that storage. So it does after a target's
    process is lost, when the first epoch's storage is gone with it, and after a
    rollback that started the epoch again on the weights it kept: the target
    of the second update is lost on trial, which rolls it back.
    """
    name = "qwen3-live-relaunched-storage"
    options = merged(
        OPTIONS,
        {"update_config": {"previous_epoch": "weights", "trial": FAULT_TRIAL}},
    )
    with _serve(deployments, name, options=options) as server:
        revisions = [
            updates.engine_revision(server.build_server, repository)
            for _ in range(3)
        ]
        answer = _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32)
        _change_epoch(server, revisions[0])
        _lose_the_serving_engine(
            server, name, "relaunched-after-a-switch", standby=False
        )
        before = admin.now(server)
        with admin.changing(server, revisions[1]) as applied:
            _fail_the_target(server, before)
        _rolled_back(applied.result())
        # The rollback released the target, so the next update starts.
        _change_epoch(server, revisions[2])
        assert _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32) == (
            answer
        )


def test_an_epoch_switch_to_a_new_executable_fails_no_request(
    deployments: Deployments, repository: Path
):
    """A deployment switches to another worker executable under a request.

    The executable is a new revision of the engine package the deployment runs,
    so the switch replaces every main process of the deployment with a process
    of a different file. A request in flight crosses the switch with its count
    whole and nothing of it fails: an epoch switch is not a restart of the
    service the client sees.
    """
    name = "qwen3-live-executable"
    with _serve(deployments, name) as server:
        revision = updates.engine_revision(server.build_server, repository)
        before = _engine_pids(server)
        failed = _failed_requests(server)
        baseline = _ask(server.base_url, name, RECOVERY_PROMPT, RECOVERY_TOKENS)

        stream = _count(server.base_url, name, RECOVERY_PROMPT)
        stream.wait_for(
            lambda streamed: len(numbers(streamed.text)) >= 8,
            "the count did not stream",
        )
        report = _change_epoch(server, revision)
        assert report, "the epoch switch reported no epoch"
        # The released processes end.
        for pid in before:
            updates.wait_for_end(pid)

        stream.join()
        assert stream.error is None, f"the count failed: {stream.error!r}"
        counted = numbers(stream.text)
        assert counted == list(range(1, len(counted) + 1)), (
            f"the response lost or repeated a count across the switch: "
            f"{counted[:40]}"
        )
        assert stream.reasons == ["stop"], f"the count ended {stream.reasons}"

        after = _ask(server.base_url, name, RECOVERY_PROMPT, RECOVERY_TOKENS)
        assert numbers(after) == numbers(baseline), (
            "the new executable answers differently from the one it replaced: "
            f"{after[:200]}"
        )
        assert _failed_requests(server) == failed, "a request failed"


def _repeated_attention(
    build_server: BuildServer, repository: Path, directory: Path
) -> dict[str, Any]:
    """Installs a decode attention that steps slower under the same stream.

    It is a package of its own: the tree's trtllm-gen paged attention
    package under the backend name ``trtllm_repeated``, whose entry runs the
    kernel several times per call (``packages/paged_attention.toml``,
    ``repeated``). It computes the same tokens, later, with no fault to
    report; selected alone, it is a kernel generation of the serving epoch.

    Its modules are built here, for the decode keys the deployments of the
    module's Build Server ran: an update's preparation then builds nothing,
    and a stream started after this still runs when the switch comes.

    Returns:
        The options that select it: the ``[kernels]`` table pinning each of
        those decode keys to its revision.
    """
    repeated = build_server.install(
        updates.derived(repository, "paged_attention", "repeated", directory)
    )

    # The decode keys the serving package's modules were built for, each with
    # the device target it was built for, whatever its revision: the store
    # holds a module per key and target a deployment launched.
    decode = sorted(
        {
            (module["target"], module["descriptor"])
            for package in build_server.list()
            if (package["op"], package["backend"])
            == ("paged_attention", "trtllm")
            for module in package["modules"]
            if module["descriptor"].startswith("phase_decode_")
        }
    )
    assert decode, "the deployment's attention built no decode key"
    for target, key in decode:
        build_server.resolve(
            repeated.op,
            repeated.backend,
            repeated.revision,
            key=key,
            target=target,
        )
    pinned = f"{repeated.backend}@{repeated.revision}"
    keys = sorted({key for _, key in decode})
    return {"kernels": {"paged_attention": dict.fromkeys(keys, pinned)}}


def _slower_target(
    build_server: BuildServer, repository: Path, directory: Path
) -> dict[str, Any]:
    """Installs what makes an epoch step slower under the same stream: a new
    revision of the tree's engine whose decode attention is
    :func:`_repeated_attention`. The engine revision is what makes the
    change an epoch.

    Returns:
        The options that select the target.
    """  # noqa: D205
    engine = updates.engine_revision(build_server, repository)
    return merged(
        engine, _repeated_attention(build_server, repository, directory)
    )


@pytest.mark.parametrize("previous_epoch", ["parked", "weights", "host"])
def test_a_fault_on_trial_returns_to_the_previous_epoch_under_a_request(
    deployments: Deployments, repository: Path, previous_epoch: str
):
    """A target lost on trial rolls back with no request lost.

    The previous epoch is kept as ``update_config.previous_epoch`` states: its
    drained processes stay alive through the trial, or they exit with its
    weights kept, and the rollback starts them again on those. The trial
    outlasts the case, which ends it by ending the target's main process.
    """
    name = "qwen3-live-rollback"
    options = merged(
        OPTIONS,
        {
            "update_config": {
                "previous_epoch": previous_epoch,
                "trial": HELD_TRIAL,
            }
        },
    )
    with _serve(deployments, name, options=options) as server:
        revision = updates.engine_revision(server.build_server, repository)
        before = _engine_pids(server)
        previous = admin.replica(server).epoch
        answer = _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32)
        stream = _start_count(server, name)
        applying = admin.now(server)
        with admin.changing(server, revision) as applied:
            trial = admin.wait_for_log(
                server, "update", after=applying, phase="trial"
            ).fields
            assert trial["change"] == "epoch", trial
            if previous_epoch != "parked":
                # The previous epoch's processes end.
                for pid in before:
                    updates.wait_for_end(pid)
            else:
                assert all(updates.running(pid) for pid in before)

            if previous_epoch == "weights":
                # The previous epoch keeps its weights alone: each GPU's
                # quota reserves nothing else of it once its processes
                # exited, the segments the target maps being the target's.
                # The Storage Server releases a process's reservations once
                # it saw the process end, and the leader's gauges state the
                # quotas every second.
                def kept() -> set[str]:
                    return {
                        sample.labels["item"]
                        for sample in _reservations(server)
                        if previous in _epochs(sample)
                    }

                try:
                    server.wait_until(
                        lambda: kept() == {"weights"},
                        ADMITTED_S,
                        "the previous epoch kept more than its weights",
                    )
                except TimeoutError as timeout:
                    held = [
                        sample
                        for sample in _reservations(server)
                        if previous in _epochs(sample)
                    ]
                    raise AssertionError(
                        f"the previous epoch keeps {held}"
                    ) from timeout
            # The previous epoch serves again once its processes stopped for
            # the trial started again; the target serves until it is lost.
            update = _fail_the_target(server, applying)
            print("rolled back:", json.dumps(update))
        _rolled_back(applied.result())
        # A node that cannot move the weights keeps them on its GPUs, and
        # says why: the update states what the previous epoch kept.
        if not update["host_refused"]:
            assert update["previous_epoch_kept"] == previous_epoch, update
        _assert_whole(stream)
        assert _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32) == (
            answer
        )


def test_a_regression_of_the_epoch_on_trial_rolls_it_back(
    deployments: Deployments, repository: Path, tmp_path: Path
):
    """A target that steps slower past the threshold rolls back by itself.

    The target's decode attention runs its kernel several times per call
    (:func:`_slower_target`), so the same stream steps slower on it with no
    fault. The stream crosses the switch and the rollback whole, the update
    ends unavailable with the threshold's reason, its ``update`` log states
    the rates that decided it, and applying it again needs ``force``. A
    deployment without the threshold takes the same slower target through a
    trial of several windows: the threshold, and nothing else, rolled it
    back.
    """
    name = "qwen3-live-regression"
    # The step time past which the trial rolls back, as a fraction of the
    # previous epoch's steps up to the switch: above what the target's loading
    # and capture on the same GPU add to those steps, below what the repeated
    # attention adds to the target's. The trial outlasts the decision.
    regression = 0.2
    thresholds = merged(
        OPTIONS,
        {
            "update_config": {
                "trial": HELD_TRIAL,
                "rollback": {
                    "step_time_regression": regression,
                    "window": f"{TRIAL_WINDOW_S}s",
                },
            }
        },
    )
    with _serve(deployments, name, options=thresholds) as server:
        slower = _slower_target(server.build_server, repository, tmp_path)
        answer = _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32)
        current = admin.policy(server).revision
        # The stream runs through the preparation, the switch, a whole window
        # of the trial and the rollback's drain, which the count outlasts only
        # when it starts with the update. The preparation's own seconds give
        # the baseline its window.
        stream = _start_count(server, name)
        _rolled_back(admin.change(server, slower))
        update = admin.logs(server, "update")[-1].fields
        assert update["phase"] == "rolled_back", update
        assert update["failure"]["kind"] == "step_time_regression", update
        print("rolled back:", json.dumps(update))
        # The update states the rates the decision read.
        (replica,) = update["trial"]["replicas"]
        assert replica["baseline"]["tokens_per_second"] > 0, update
        baseline = replica["baseline"]["step_us"]
        assert replica["rates"]["step_us"] > baseline * (1 + regression), update
        _assert_whole(stream)
        assert _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32) == (
            answer
        )
        assert admin.policy(server).revision == current

        # The rollback marked the target unavailable: applying it again is
        # refused without `force`, as a target a rollback marked unavailable
        # is (``400``).
        again = admin.change(server, slower)
        assert again.status_code == 400, again.text

    # Without the threshold, the same target stands a trial of three windows
    # under a stream and commits.
    control = f"{name}-unthresholded"
    options = merged(
        OPTIONS, {"update_config": {"trial": f"{3 * TRIAL_WINDOW_S}s"}}
    )
    with _serve(deployments, control, options=options) as server:
        stream = _start_count(server, control)
        record = _change_epoch(server, slower)
        assert record["trial_s"] >= 2 * TRIAL_WINDOW_S, record
        _assert_whole(stream)


def test_a_package_update_past_a_threshold_rolls_back_in_its_epoch(
    deployments: Deployments, repository: Path, tmp_path: Path
):
    """A generation update stands the same trial as an epoch switch.

    The update selects another decode attention package
    (:func:`_repeated_attention`), a kernel generation of the serving epoch.
    The serving epoch computes a stream before the update and a trickle
    after it, so its token throughput falls past the threshold. The update
    rolls back in the same main process, with the threshold as its failure.
    """
    name = "qwen3-live-package-trial"
    options = merged(
        OPTIONS,
        {
            "update_config": {
                "trial": HELD_TRIAL,
                "rollback": {
                    "throughput_regression": 0.5,
                    "window": f"{TRIAL_WINDOW_S}s",
                },
            }
        },
    )
    with _serve(deployments, name, options=options) as server:
        # The package is built for the decode keys this deployment ran.
        repeated = _repeated_attention(
            server.build_server, repository, tmp_path
        )
        engines = _engine_pids(server)
        stream = _start_count(server, name)
        # The baseline is the window of steps before the switch: the count's
        # own deltas say when it has streamed for one.
        stream.wait_for(
            lambda streamed: (
                streamed.arrivals[-1] - streamed.arrivals[0] > TRIAL_WINDOW_S
            ),
            "the count did not stream for a window",
        )
        before = admin.now(server)
        with admin.changing(server, repeated) as applied:
            _assert_whole(stream)

            # A trickle of small requests keeps the trial reading steps, one
            # each pause, until the update's ``update`` log states its end.
            deadline = time.monotonic() + server.scaled(ROLLBACK_S)
            update = None
            while update is None:
                assert time.monotonic() < deadline, (
                    "the update was not rolled back"
                )
                _ask(server.base_url, name, RECOVERY_PROMPT, tokens=2)
                ends = [
                    event.fields
                    for event in admin.logs(server, "update", after=before)
                    if event.fields.get("phase") in ("committed", "rolled_back")
                ]
                if ends:
                    update = ends[0]
                else:
                    time.sleep(TRICKLE_S)
        _rolled_back(applied.result())
        print("rolled back:", json.dumps(update))
        assert update["change"] == "generation", update
        assert update["phase"] == "rolled_back", update
        assert update["failure"]["kind"] == "throughput_regression", update
        # The update ran in the serving main processes.
        assert _engine_pids(server) == engines


def test_a_request_that_ends_at_once_leaves_its_blocks_in_the_host_tier(
    deployments: Deployments,
):
    """A prompt's blocks reach the host tier though no batch follows them.

    A request of one output token ends with the step that fills its blocks,
    and nothing else runs, so the copies its stores need have no later batch
    to go with: the scheduler sends its executor an empty batch, which runs
    them on the device's copy stream (``specs/buffer-pool.md`` §10).
    """
    name = "qwen3-live-copies-alone"
    prompt = " ".join(f"entry {index} of the table" for index in range(800))
    with _serve(deployments, name) as server:
        _ask(server.base_url, name, prompt, tokens=1)
        time.sleep(server.scaled(STORE_S))
        _flush(server)
        # The device holds none of the prompt's blocks after the flush, so
        # what the admission finds cached it loads from the host tier.
        found = _cached(server.base_url, name, prompt)
        print("host tokens of a request of one token:", found)
        assert found > 0, found
        # What moved the blocks is on the report beside what computed them.
        copied = admin.counter(
            server, "uniserve_stream_seconds_total", stream="copy"
        )
        print("copy stream:", copied)
        assert copied > 0, copied


def test_host_hits_beyond_the_device_cache_all_finish(
    deployments: Deployments,
):
    """Requests that find more blocks in the host tier than the device holds
    all end.

    A host hit takes the units of its blocks at admission, and a running
    request keeps its units until it ends. Admitted together, requests whose
    tails run past their first chunk would hold every unit with none left for
    those tails, so admission keeps available the units the running requests
    still need to reach their next call and low watermark
    (``specs/scheduler-design.md`` §4.4).
    """  # noqa: D205
    name = "qwen3-live-host-hits"
    prompts = [
        " ".join(f"{turn}-{word}" for word in range(2495)) for turn in range(4)
    ]
    tail = " ".join(f"tail-{word}" for word in range(300))
    with _serve(deployments, name) as server:
        for prompt in prompts:
            _ask(server.base_url, name, prompt, tokens=1)
        time.sleep(server.scaled(STORE_S))
        _flush(server)
        found: list[int] = []
        errors: list[BaseException] = []

        def ask(prompt: str) -> None:
            try:
                found.append(_cached(server.base_url, name, prompt))
            except httpx.HTTPError as error:
                errors.append(error)

        # The four prompts share no block, and the device holds none of
        # theirs after the flush: what each admission finds cached it loads
        # from the host tier.
        asking = [
            threading.Thread(target=ask, args=(f"{prompt} {tail}",))
            for prompt in prompts
        ]
        for thread in asking:
            thread.start()
        for thread in asking:
            thread.join()
        print("host tokens of the four requests:", found)
        assert not errors, f"{errors[0]!r}"
        assert sum(found) > 0, found


@pytest.mark.gpus(2)
@pytest.mark.parametrize(
    "parallel",
    [
        (
            "expert",
            {
                "data_parallel_size": 2,
                "moe_config": {"expert_parallel": True},
            },
        ),
        ("data", {"data_parallel_size": 2}),
    ],
    ids=lambda parallel: parallel[0],
)
def test_a_process_of_two_devices_keeps_the_host_kv_cache(
    deployments: Deployments, parallel: tuple[str, Config]
):
    """A main process over two devices finds a prompt's blocks in the host
    tier.

    Each of its schedulers serves one device with a queue pair of its own on
    the host's buffer pool, whose host regions keep the blocks it stores,
    which an expert union computes on both devices. A prompt whose device
    copies a flush evicted is loaded back from the host, and the answer is the
    one it gave before.
    """  # noqa: D205
    kind, options = parallel
    name = f"qwen3-live-host-{kind}"
    prompt = " ".join(f"item {index} of the list" for index in range(800))
    with _serve(deployments, name, options=merged(OPTIONS, options)) as server:
        answer = _ask(server.base_url, name, prompt, tokens=16)
        time.sleep(server.scaled(STORE_S))
        _flush(server)
        # The device holds none of the prompt's blocks after the flush, so
        # what the admission finds cached it loads from the host tier.
        loaded = chat.complete(server.base_url, _request(name, prompt, 16))
        found = cached_tokens(loaded)
        print(f"host tokens on {kind} parallel devices:", found)
        assert found > 0, loaded["usage"]
        assert chat.content(loaded) == answer


@pytest.mark.parametrize(
    "tiers",
    [
        {},
        {"host_tier": False},
        {"prefix_caching": False},
    ],
    ids=["host", "device", "uncached"],
)
def test_requests_that_outgrow_the_cache_in_decode_all_finish(
    deployments: Deployments, tiers: Config
):
    """A cache that fills while its requests decode gives one of them up.

    Admission holds what a request's prompt takes and not what it generates, so
    eight counts on a cache of 8192 tokens outgrow it as they decode. The one
    admitted last gives its units back and continues from the tokens it
    generated, so every count ends whole and none starves. A count alone that
    fills the whole cache ends at the length the cache holds, with the finish
    reason ``cache_capacity`` (``specs/scheduler-design.md`` §3.6).
    """
    name = "qwen3-live-decode-growth"
    options = merged(
        SIZED_OPTIONS, {"kv_cache_config": {"max_tokens": 8192, **tiers}}
    )
    # About 3400 tokens each (a number of three digits, its comma and its
    # space are five tokens), so eight of them need more than three caches of
    # 8192 tokens by their ends, and the cache fills while every one decodes.
    prompt = _counting_to(700)
    with _serve(deployments, name, options=options) as server:
        streams = [_count(server.base_url, name, prompt) for _ in range(8)]
        counts = []
        for stream in streams:
            stream.join()
            assert stream.error is None, f"{stream.error!r}"
            counted = numbers(stream.text)
            assert counted == list(range(1, 701)), counted[-60:]
            assert stream.reasons == ["stop"], stream.reasons
            counts.append(len(counted))
        # Eight such counts outgrow the cache by their length alone, so every
        # one ending whole is the request given up continuing.
        print("numbers of each count:", counts)
        # A count longer than the cache holds runs alone until it fills it, and
        # ends there: its next token needs a block the whole cache lacks.
        alone = _count(server.base_url, name)
        alone.join()
        assert alone.error is None, f"{alone.error!r}"
        *whole, last = numbers(alone.text)
        print(
            "numbers of a count the cache cannot hold:",
            len(whole),
            alone.reasons,
        )
        assert whole == list(range(1, len(whole) + 1)), whole[-60:]
        # The cache is full where it ends, which may be inside its last number.
        assert str(len(whole) + 1).startswith(str(last)), (whole[-5:], last)
        assert alone.reasons == ["cache_capacity"], alone.reasons


def test_a_weight_update_leaves_the_blocks_of_the_old_weights_unread(
    deployments: Deployments,
):
    """A block in the host tier belongs to the weights that computed it.

    A prompt finds its blocks in the host tier once their device copies are
    flushed. After an update to other weights the engine keys its blocks by
    the new weights, so the same prompt finds none of them, although the new
    weights name the same model: it computes its own, which the next flush
    makes it find. The other checkpoint is another copy of the files, so the
    two weights answer alike and a stale block would go unseen in the text.
    """
    other = CHECKPOINTS["qwen3-other-weights"].model
    name = "qwen3-live-weights-blocks"
    prompt = " ".join(f"item {index} of the list" for index in range(800))

    def found_in_the_host_cache(server: Server) -> int:
        """Sends the prompt on a flushed device; the host tokens it reads:
        the device holds none of its blocks after the flush.
        """  # noqa: D205
        _flush(server)
        return _cached(server.base_url, name, prompt, tokens=16)

    with _serve(deployments, name) as server:
        first = _ask(server.base_url, name, prompt, tokens=16)
        time.sleep(server.scaled(STORE_S))
        old_weights = found_in_the_host_cache(server)
        print("host tokens of the first weights:", old_weights)
        assert old_weights > 0, old_weights
        updated = admin.revision(admin.change(server, {"model": other}))
        assert updated["effect"] == "epoch", updated
        print("updated:", json.dumps(updated)[:400])
        # The blocks of the old weights are found in no tier.
        second = chat.complete(server.base_url, _request(name, prompt, 16))
        stale = cached_tokens(second)
        print("cached tokens after the update:", stale)
        assert stale == 0, second["usage"]
        assert chat.content(second) == first
        time.sleep(server.scaled(STORE_S))
        new_weights = found_in_the_host_cache(server)
        print("host tokens of the new weights:", new_weights)
        assert new_weights > 0, new_weights


def test_a_weight_update_keeps_the_stream_in_flight_going(
    deployments: Deployments,
):
    """A weight update never holds the stream in flight as long as it takes.

    A change of the model files starts a new epoch beside the serving one
    (``docs/admin.md``, Updates): the batches go on while the new checkpoint
    loads, and the stream moves to the new epoch at its switch, so the
    stream's longest silence from its last token before the update to its end
    stays below the update's duration, and the stream is whole.
    """
    other = CHECKPOINTS["qwen3-other-weights"].model
    name = "qwen3-live-weights-stream"
    with _serve(deployments, name) as server:
        stream = _start_count(server, name)
        started = time.monotonic()
        updated = admin.change(server, {"model": other})
        took = time.monotonic() - started
        assert admin.revision(updated)["effect"] == "epoch", updated.text
        stream.join()

    # The stream's silences from its last token before the update on, each
    # token's arrival its own reading.
    spans = [moment for moment in stream.arrivals if moment < started][-1:] + [
        moment for moment in stream.arrivals if moment >= started
    ]
    longest = max(
        (later - earlier for earlier, later in zip(spans, spans[1:])),
        default=0.0,
    )
    print(f"update {took:.3f} s, the stream's longest silence {longest:.3f} s")
    assert stream.error is None, f"the count failed: {stream.error!r}"
    counted = numbers(stream.text)
    assert counted == list(range(1, len(counted) + 1)), counted[-40:]
    assert len(spans) > 1, "the count ended before the update"
    assert longest < took, (
        f"the stream in flight waited {longest:.3f} s, longer than the update "
        f"took ({took:.3f} s)"
    )


def test_a_previous_epoch_in_host_storage_moves_its_weights_and_returns(
    deployments: Deployments,
):
    """``previous_epoch = "host"`` moves bytes, and a rollback brings them back.

    A target of other model files maps none of the serving epoch's weights, so
    the weights that epoch alone retains leave its GPU for host storage while
    the target serves. The target is lost on trial, and the rollback returns
    the weights and starts the epoch on them, under a stream that crosses
    both moves whole, and it answers as it did.
    """
    other = CHECKPOINTS["qwen3-other-weights"].model
    name = "qwen3-live-host-weights"
    # The target loads its checkpoint beside the serving one, and the
    # deployment's quota of the GPU holds both only past the default share.
    options = merged(
        OPTIONS,
        {
            "mem_fraction_static": 0.9,
            "update_config": {"previous_epoch": "host", "trial": HELD_TRIAL},
        },
    )
    with _serve(deployments, name, options=options) as server:
        answer = _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32)
        stream = _start_count(server, name)
        previous = admin.replica(server).epoch
        before = admin.now(server)
        with admin.changing(server, {"model": other}) as applied:
            trial = admin.wait_for_log(
                server, "update", after=before, phase="trial"
            ).fields
            assert trial["change"] == "epoch", trial

            def moved() -> bool:
                return not any(
                    sample.labels["item"] == "weights"
                    and _epochs(sample) == {previous}
                    for sample in _reservations(server)
                )

            # The weights move once the previous epoch's processes exited,
            # during the trial: its GPUs' quotas then reserve no weights of
            # the previous epoch alone, which the leader's gauges state every
            # second.
            server.wait_until(
                moved,
                ADMITTED_S,
                "the previous epoch's weights did not move",
            )
            update = _fail_the_target(server, before)
        _rolled_back(applied.result())
        print("kept:", json.dumps(update))
        # The update states the move: the bytes moved, and no host that
        # refused it.
        assert not update["host_refused"], update
        assert update["previous_epoch_kept"] == "host", update
        assert update["moved_to_host_bytes"] > 0, update
        _assert_whole(stream)
        assert _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32) == (
            answer
        )
        assert update["phase"] == "rolled_back", update


def test_step_changes_under_a_request_keep_its_stream_whole(
    deployments: Deployments,
):
    """Changes at a step boundary under a request leave its stream as a run
    without them.

    A scheduling limit takes effect at a scheduler's next step, each step
    reading one configuration, so every token the request computes is the
    one its stream delivers. The count crosses as many changes as fit under
    it, each one converged before the next is applied.
    """  # noqa: D205
    name = "qwen3-live-switches"
    switches = 0
    with _serve(deployments, name) as server:
        for _ in range(2):
            stream = _start_count(server, name)
            while not stream.ended:
                switches += 1
                size = 4 if switches % 2 else 8
                record = _apply(
                    server, {"scheduler_config": {"max_batch": size}}
                )
                assert record["effect"] == "step", record
            _assert_whole(stream)
    print("switches:", switches)
    assert switches >= 20, switches


#: Options each change of which takes effect at the engines' next step: what
#: every main process reads beside its batches, what a start reads, the
#: cache's switches turned off, and a lower share of the GPUs (the default
#: `mem_fraction_static` is 0.7). Each value differs from the deployment's.
STEP_OPTIONS: list[Config] = [
    {"worker_config": {"watchdog_timeout": "100s"}},
    {"worker_config": {"join_timeout": "100s"}},
    {"worker_config": {"compare_timeout": "100s"}},
    {"worker_config": {"report_period": "500ms"}},
    {"worker_config": {"copy_budget": "64MiB"}},
    {"device_config": {"queue_depth": 3}},
    {"profiling_config": {"nvtx_ranges": True}},
    {"kv_cache_config": {"host_tier": False}},
    {"kv_cache_config": {"prefix_caching": False}},
    {"mem_fraction_static": 0.6},
]


def test_step_options_change_with_no_new_process(deployments: Deployments):
    """A change of an option that takes effect at the next step starts no
    process: the replica serves the same epoch with the same processes, and
    a request streamed across the changes is whole.
    """  # noqa: D205
    name = "qwen3-live-step-options"
    with _serve(deployments, name) as server:
        epoch = admin.replica(server).epoch
        launched = admin.processes(server)
        stream = _start_count(server, name)
        for change in STEP_OPTIONS:
            record = _apply(server, change)
            assert record["effect"] == "step", (change, record)
            assert admin.replica(server).epoch == epoch, change
        _assert_whole(stream)
        assert admin.processes(server) == launched
