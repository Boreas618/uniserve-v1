"""A deployment's lifecycle and live updates through the admin API.

Qwen3 serves on the GPUs each case leases, as the deployment both update
modules serve (:data:`uniserve_testing.updates.PROFILE`), with a Storage
Server of the case's own: a case that reads or restarts it names the runtime
directory it serves under (the ``runtime`` fixture).

- A node (``uniserve serve`` without a model) brings deployments up
  (``PUT /admin/config``) and down through its admin API without
  restarting: each deployment's storage belongs to it, so the Storage Server
  holds nothing of a deployment brought down, and a deployment whose host
  regions the host's available storage cannot hold is refused with the sizes
  that decided it.
- A serving deployment answers its admin API, on the daemon's operator
  socket: new requests wait while generation is paused and are served once
  it continues; a flush evicts the device copies of the cached blocks;
  scheduling limits within the batch the workers were started for change in
  place while a request streams, and a change of the decode batch sizes is
  an epoch change under the same stream, which the operator commits; the
  manager attaches to the node, follows its events and detaches while it
  serves.
- A hot restart hands the deployment and its responses in flight to a
  successor daemon, and a Storage Server restart keeps it serving.
- A target that finds no storage, a lost main process, a rollback and a
  regression on trial leave the requests in flight whole.
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
import tomllib
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import httpx
import pytest
from uniserve_testing import admin, builds, chat, updates
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.builds import BuildServer
from uniserve_testing.caching import cached_tokens
from uniserve_testing.client import metric
from uniserve_testing.deployments import (
    HOST_CACHE_GIB,
    Deployments,
    Settings,
    runtime_directory,
    storage_status,
)
from uniserve_testing.devices import Lease
from uniserve_testing.plugin import environment_program
from uniserve_testing.servers import Server, environment
from uniserve_testing.updates import Stream, numbers

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

#: The options every deployment of the module serves with.
OPTIONS = updates.PROFILE


@pytest.fixture(scope="module")
def build_server(
    uniserve: Path,
    repository: Path,
    e2e_settings: Settings,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[BuildServer]:
    """The module's own Build Server, with the tree's packages installed.

    The cases install engine revisions and a slower attention package, and
    an installed revision is what a later epoch selects, so they install at
    a server of their own, which no other module's deployment resolves at.
    """
    with builds.tree_build_server(
        uniserve,
        repository,
        tmp_path_factory.mktemp("build-server"),
        scale=e2e_settings.time_scale,
    ) as server:
        yield server


#: A change of the options that only a new epoch executes: graphs of other
#: decode batch sizes.
CHANGE = "cuda_graph_config.decode_batch_sizes=[1, 2, 8]"

#: The options of a deployment whose KV cache takes the device storage its
#: share leaves, rather than a fixed token count.
SIZED_OPTIONS = tuple(
    option
    for option in OPTIONS
    if not option.startswith("kv_cache_config.max_tokens")
)

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
#: admitted and to appear in what the daemon reports, in seconds: the daemon
#: takes it within a step and its schedulers report once a second. No event
#: states a single request.
ADMITTED_S = 60

#: How long a node's deployment takes to come down after its shutdown, in
#: seconds: its requests drain within the shutdown timeout (30 s unless
#: stated), its engines then end and its storage is released.
DOWN_S = 120

#: How long a hot restart's successor takes to state its readiness record in
#: the status once it answered the handover, in seconds: it states it as it
#: begins to serve.
TAKEOVER_S = 120

#: How long a Storage Server's successor takes to serve the socket after the
#: restart command answered, in seconds: the handover waits until no copy of
#: the deployment is in flight, which a step ends.
HANDOVER_S = 120

#: How long a standby takes to park, in seconds: it starts as an update's
#: target does, and the bound is the one a target's preparation has
#: (:func:`uniserve_testing.admin.wait_for_phase`).
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
    options: tuple[str, ...] = OPTIONS,
    runtime: Path | None = None,
) -> AbstractContextManager[Server]:
    """A deployment of Qwen3 serving `name` with `options`."""
    return deployments.serve(
        "--served-model-name",
        name,
        *updates.assigned(*options),
        model=CHECKPOINTS["qwen3"].model,
        runtime=runtime,
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


def _apply(server: Server, *assignments: str, **fields: object) -> dict:
    """Apply `assignments` (``PUT /admin/config``); its result, once an
    update reached its trial or the change was made.
    """  # noqa: D205
    response = admin.apply(server, *assignments, **fields)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def _commit(server: Server) -> dict:
    """Commit the update on trial; the update once it entered ``committed``,
    which releases its kept side.
    """  # noqa: D205
    before = admin.next_event(server)
    committed = admin.operation(server, "config/commit")
    assert committed.status_code == 200, committed.text
    return admin.wait_for_phase(server, "committed", after=before)


def _change_epoch(server: Server, *assignments: str) -> dict:
    """Apply `assignments` as an epoch switch and commit its target; the
    update as it stood on trial.
    """  # noqa: D205
    result = _apply(server, *assignments)
    assert result["change"] == "epoch", result
    _commit(server)
    return result["update"]


def _held_requests(server: Server) -> int:
    """The new requests the daemon holds before an engine takes them."""
    return admin.status(server)["deployment"]["held"]


def _bytes_of(server: Server, item: str, epoch: int | None = None) -> int:
    """The bytes the deployment's quotas reserve for `item` (of `epoch`'s
    reservations, when stated).
    """  # noqa: D205
    return sum(
        entry["bytes"]
        for entry in admin.reservations(server)
        if entry["item"] == item and (epoch is None or epoch in entry["epochs"])
    )


def _failed_requests(base_url: str) -> float:
    """The requests that ended in an error so far."""
    return metric(base_url, "uniserve:request_outcomes", outcome="error")


def _flush(server: Server) -> None:
    """Evict the device copies of every cached block."""
    flushed = admin.operation(server, "flush_cache")
    assert flushed.status_code == 200, flushed.text


def _deployment(
    deployments: Deployments, name: str, *assignments: str
) -> list[str]:
    """The assignments that bring the deployment `name` up on an idle node."""
    return [
        f"model={CHECKPOINTS['qwen3'].model}",
        f"served_model_name={name}",
        f"kv_cache_config.host_gib={HOST_CACHE_GIB}",
        *OPTIONS,
        *deployments.assignments(),
        *assignments,
    ]


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


def test_a_node_serves_deployments_one_after_another_and_releases_their_storage(
    deployments: Deployments, uniserve: Path, runtime: Path
):
    # The node states no deployment of its own, so no host tier either: each
    # deployment it brings up states its own.
    with deployments.serve(name="node", host_gib=None, runtime=runtime) as node:
        available = 0
        for index in range(3):
            name = f"qwen3-live-{index}"
            # A bring-up answers once the deployment serves.
            brought = admin.apply(
                node, *_deployment(deployments, name), config=""
            )
            assert brought.status_code == 200, brought.text
            assert brought.json()["result"]["change"] == "deployment", (
                brought.text
            )
            status = admin.status(node)
            assert status["state"] == "serving", status
            assert status["deployment"]["deployment"] == name, status
            assert _ask(node.base_url, name, "Say hello.")

            # The host's Storage Server holds the deployment: its claim and
            # the weights its quota reserves.
            held = _held_of(updates.listing(uniserve, runtime), name)
            assert held["deployments"], held
            reserved = {
                entry["item"]
                for quota in held["quotas"]
                for entry in quota["reservations"]
            }
            assert "weights" in reserved, held
            # The host's available storage, as the Storage Server reads it.
            (usage,) = admin.config(node)["quotas"]
            available = usage["host_storage"]["available"]

            # The shutdown answers at once, and the deployment comes down
            # after it: the node's status states it idle once it did, and
            # the node's event log records the change with no data that
            # tells it apart from the shutdown's own.
            down = admin.operation(node, "shutdown")
            assert down.status_code == 202, down.text
            node.wait_until(
                lambda: admin.status(node)["state"] == "idle",
                DOWN_S,
                f"{name} did not come down",
            )
            status = admin.status(node)
            assert status["last_error"] is None, status
            released = _held_of(updates.listing(uniserve, runtime), name)
            assert not any(released.values()), released

        # A host tier the host's available storage cannot hold.
        too_large = 2 * available // GIB
        refused = admin.apply(
            node,
            *_deployment(
                deployments,
                "qwen3-live-too-large",
                f"kv_cache_config.host_gib={too_large}",
            ),
            config="",
        )
        assert refused.status_code != 200, refused.text
        status = admin.status(node)
        assert status["state"] == "idle", status
        error = status["last_error"] or ""
        print(f"refused: {error}")
        assert "available" in error and "reserve" in error, error
        # The node is the same process throughout.
        assert node.process.poll() is None


def test_a_serving_deployment_answers_its_admin_api(deployments: Deployments):
    name = "qwen3-live-admin"
    # The deployment keeps no host tier: what a request finds cached is on the
    # device, so the tokens its answer states as cached say what a flush
    # evicted.
    options = (*OPTIONS, "kv_cache_config.host_tier=false")
    with _serve(deployments, name, options=options) as server:
        status = admin.status(server)
        assert status["deployment"]["deployment"] == name, status
        assert status["deployment"]["engine"] == "serving", status

        # New requests wait while generation is paused.
        paused = admin.operation(server, "pause_generation")
        assert paused.status_code == 200, paused.text
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
        resumed = admin.operation(server, "continue_generation")
        assert resumed.status_code == 200, resumed.text
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

        # Scheduling limits change in place under a streaming request, and
        # other decode batch sizes by an epoch change under the same stream.
        stream = _streaming(server, name)
        update = _apply(server, "scheduler_config.max_batch=4")
        assert update["change"] == "generation", update
        _commit(server)
        before_update = len(stream.deltas)
        report = _change_epoch(server, CHANGE)
        print(f"epoch switch: {report['switch_ms']} ms")
        _assert_counts_from_one(stream, 100)
        assert len(stream.deltas) > before_update, (
            "the stream ended before the update"
        )

        # The manager attaches to the node by its operator socket, reads its
        # configuration status, follows its events and detaches. It names
        # the node by its address at the head of each line it prints, and
        # prints what the node answers as JSON after it.
        manager = str(environment_program("uniserve-manager"))
        node = server.endpoint("admin")
        configured = subprocess.run(
            [manager, "--node", node, "config"],
            env=environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        configuration = json.loads(configured.removeprefix(f"{node}: "))
        assert configuration["executing"]["state"] == "serving", configuration
        served = tomllib.loads(configuration["config"])
        assert served["served_model_name"] == name, served

        # The watch replays the events from the start, the config operations
        # among them, and detaches at an interrupt.
        with subprocess.Popen(
            [manager, "--node", node, "watch", "--from-start"],
            env=environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as watching:
            assert watching.stdout is not None
            for line in watching.stdout:
                event = json.loads(line.removeprefix(f"{node}: "))
                if (
                    event["kind"] == "operation"
                    and event["data"]["kind"] == "config"
                ):
                    break
            else:
                pytest.fail("the manager's watch ended before a config event")
            watching.send_signal(signal.SIGINT)
            _, errors = watching.communicate()
        assert watching.returncode == 0, errors
        assert _ask(server.base_url, name, "Say hello.")

        log = admin.events(server, 0, 0)
        kinds = [event["kind"] for event in log["events"]]
        assert "operation" in kinds, log


#: The streams a hot restart hands over: three counts of different lengths,
#: so their event boundaries fall at different points of the response, and
#: the number each stream must carry with no gap and no repetition.
HANDED_STREAMS = (200, 500, 900)

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
        _restart_while_streaming(
            server, name, deployments.build_server, successors
        )


def test_a_switch_after_a_hot_restart_finds_the_storage_of_the_epoch_before(
    deployments: Deployments, repository: Path
):
    """A successor daemon learns what a target maps from the process that serves.

    The epoch that serves took its storage from the one before it, under that
    epoch's name. The daemon that switched it is gone, and the next target still
    maps that storage.
    """  # noqa: E501
    name = "qwen3-live-restart-switch"
    revisions = [
        updates.engine_revision(deployments.build_server, repository)
        for _ in range(2)
    ]
    with (
        _serve(deployments, name) as server,
        updates.successors() as successors,
    ):
        answer = _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32)
        _change_epoch(server, revisions[0])
        restarted = admin.operation(server, "restart")
        assert restarted.status_code == 200, restarted.text
        successors.append(restarted.json()["result"]["pid"])
        _ask(server.base_url, name, RECOVERY_PROMPT, tokens=8)
        _change_epoch(server, revisions[1])
        assert _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32) == (
            answer
        )


def _taken_over(server: Server, pid: int) -> str:
    """The base URL of the client API the daemon `pid` serves once it took
    `server`'s deployment over in a hot restart, as its readiness record in
    the status states it.

    A successor goes on in its predecessor's instance directory and binds the
    admin sockets over its predecessor's, so `server`'s admin socket answers
    for it. It states its record once it begins to serve, after it answered
    the handover, so the change that restarted the daemon may answer first;
    no event carries the record, so the status is read until it names the
    successor.
    """  # noqa: D205

    def record() -> dict | None:
        try:
            readiness = admin.status(server).get("readiness")
        except (httpx.HTTPError, OSError):
            return None
        return readiness if readiness and readiness["pid"] == pid else None

    server.wait_until(
        lambda: record() is not None,
        TAKEOVER_S,
        f"daemon {pid} did not take the deployment over",
    )
    stated = record()
    assert stated is not None
    host, _, port = stated["endpoints"]["client"].rpartition(":")
    return f"http://{host}:{port}"


def _restart_while_streaming(
    server: Server,
    name: str,
    build_server: BuildServer | None,
    successors: list[int],
) -> None:
    engines_before = admin.engine_pids(server)
    status = admin.status(server)
    assert status["deployment"]["engine"] == "serving", status
    first_event = admin.next_event(server)

    # Three streams in flight at the restart, of different lengths so their
    # boundaries fall at different events of different requests.
    streams = [
        _count(server.base_url, name, _counting_to(last))
        for last in HANDED_STREAMS
    ]
    for stream in streams:
        stream.wait_for(
            lambda streamed: len(streamed.deltas) >= 32,
            "the counts did not stream",
        )
    restarted = admin.operation(server, "restart")
    assert restarted.status_code == 200, restarted.text
    successor = restarted.json()["result"]["pid"]
    successors.append(successor)
    assert successor != server.pid
    before_restart = [len(stream.deltas) for stream in streams]

    # The successor serves the same deployment, so it holds the same modules'
    # server: a deployment reaches its scheduling policy only through it, and
    # the successor reads the configuration the old process served rather than
    # a command line of its own (design 7.5).
    assert build_server is not None
    inherited = tomllib.loads(admin.config(server)["config"])
    assert inherited["build_config"]["socket"] == str(build_server.socket), (
        f"the successor did not inherit the modules' server: {inherited}"
    )

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

    # A one-body chat response in flight at a further restart is written by
    # its successor too, whole (design 7.5.1: every response in flight moves).
    _restart_with_a_json_body(server, name, successors)

    # The daemon the case started exits once its requests ended; the engine
    # and its workers are the same processes throughout.
    server.process.wait()
    # The one-body restart replaced `successor` in turn, so the daemon that
    # serves now is the last of them, on the same address.
    assert _taken_over(server, successors[-1]) == server.base_url
    status = admin.status(server)
    assert status["deployment"]["engine"] == "serving", status
    assert admin.engine_pids(server) == engines_before
    # The node's event log goes on in each successor: the restart its
    # predecessor recorded the start of is still in it, beside the
    # successor's own record of the deployment it serves.
    log = admin.events(server, first_event - 1, 0)
    assert not log["gap"], log
    recorded = [
        (event["kind"], (event.get("data") or {}).get("kind"))
        for event in log["events"]
    ]
    assert ("operation", "restart") in recorded, recorded
    assert ("deployment", None) in recorded, recorded

    # The successor restarts in turn, and takes a change in place.
    again = admin.operation(server, "restart")
    assert again.status_code == 200, again.text
    successors.append(again.json()["result"]["pid"])
    updates.wait_for_end(successor)
    assert _ask(server.base_url, name, "Say hello.")
    # A scheduling limit is a parameter of the policy package, so it reaches
    # the serving epoch as a generation update; `max_running_requests` is build
    # config and would switch the epoch.
    update = _apply(server, "scheduler_config.max_batch=4")
    assert update["change"] == "generation", update
    _commit(server)

    # A change of the listening address restarts the daemon on it: the new
    # address serves, and the old one closes once the replaced daemon exits.
    previous = successors[-1]
    moved = _apply(server, "admin_config.host=localhost")
    assert moved["change"] == "daemon", moved
    successors.append(moved["pid"])
    # The successor binds the new address, on a port it picks, once it holds
    # the deployment.
    moved_url = _taken_over(server, moved["pid"])
    assert _ask(moved_url, name, "Say hello.")
    updates.wait_for_end(previous)
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{server.base_url}/health", timeout=5)
    served = tomllib.loads(admin.config(server)["config"])["admin_config"]
    assert served["host"] == "localhost", served

    # A change of both the daemon's options and what an epoch executes
    # restarts the daemon on its new options first, and its successor then
    # changes the epoch in an operation of its own.
    previous = successors[-1]
    before = admin.next_event(server)
    both = _apply(server, "admin_config.host=127.0.0.1", CHANGE)
    assert both["change"] == "daemon", both
    assert both["then"] == "update", both
    successors.append(both["pid"])
    both_url = _taken_over(server, both["pid"])
    trial = admin.wait_for_phase(
        server, "trial", "abandoned", "rolled_back", after=before
    )
    assert trial["phase"] == "trial", trial
    assert trial["change"] == "epoch", trial
    # The update the successor's operation started is numbered by it.
    last = admin.status(server)["last_operation"]
    assert last["kind"] == "config" and last["state"] == "succeeded", last
    assert trial["update"] == last["id"], (trial, last)
    # The commit releases the previous epoch, and the configuration is the
    # target's from then on.
    _commit(server)
    served = tomllib.loads(admin.config(server)["config"])
    assert served["cuda_graph_config"]["decode_batch_sizes"] == [1, 2, 8]
    assert served["admin_config"]["host"] == "127.0.0.1", served
    assert _ask(both_url, name, "Say hello.")
    updates.wait_for_end(previous)


def _restart_with_a_json_body(
    server: Server, name: str, successors: list[int]
) -> None:
    """A non-streaming chat body in flight across a restart is written whole.

    The body is one response built from a collector's state, so a field lost in
    the crossing is silent: the request still answers 200 with a response that
    parses. What catches it is a body that finished, whose text is a contiguous
    count and whose usage adds up.
    """
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
    # event states a single request.
    server.wait_until(
        lambda: metric(server.base_url, "uniserve:scheduler_running") >= 1,
        ADMITTED_S,
        "the one-body request was never served",
    )
    restarted = admin.operation(server, "restart")
    assert restarted.status_code == 200, restarted.text
    successors.append(restarted.json()["result"]["pid"])
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
            [str(uniserve), "storage-server", "restart"],
            env=environment(XDG_RUNTIME_DIR=str(runtime)),
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
        assert _ask(server.base_url, name, "Say hello.")
        # The quota reserves a storage once however many epochs map it, so a
        # target that read the checkpoint again would reserve its weights
        # beside these.
        assert _bytes_of(server, "weights") == weights


def test_a_target_that_finds_no_storage_leaves_the_serving_epoch_serving(
    deployments: Deployments, lease: Lease
):
    """A target beside a full GPU ends the update and changes nothing that serves.

    A target prepares beside the serving epoch, which keeps its regions until
    it drains. With the GPU's free storage held by another process, the
    target's start finds no room and fails, the update is abandoned before any
    switch (the apply answers ``409`` with the abandoned update,
    ``docs/admin.md``), and the stream goes on with the serving epoch's regions
    as they were.
    """  # noqa: E501
    name = "qwen3-live-pressure"
    prompt = "Describe the sea in one sentence."
    with _serve(deployments, name, options=SIZED_OPTIONS) as server:
        before = _ask(server.base_url, name, prompt, 32)
        serving = admin.config(server)["executing"]["epoch"]
        cache = _bytes_of(server, "cache_segment", serving)
        stream = _streaming(server, name)
        with updates.ballast(lease) as held:
            print(f"ballast holds {held / GIB:.1f} GiB")
            # The apply waits until the update reached its trial or ended
            # before it: abandoned, it answers 409 with the update.
            response = admin.apply(server, CHANGE)
            assert response.status_code == 409, response.text
            result = response.json()["result"]
            assert result["change"] == "epoch", result
            update = result["update"]
            print("epoch switch under pressure:", json.dumps(update))
            # The target's start failed before it joined: a failed
            # preparation.
            assert update["phase"] == "abandoned", update
            assert update["failure"]["kind"] == "prepare_failed", update
        _assert_counts_from_one(stream, 200)
        # The serving epoch's quota reserves at least the KV cache it held.
        assert admin.config(server)["executing"]["epoch"] == serving
        assert _bytes_of(server, "cache_segment", serving) >= cache, (
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
        epoch = admin.config(server)["executing"]["epoch"]
        awake = _bytes_of(server, "cache_segment", epoch)
        assert awake > 0, admin.reservations(server)
        stream = _streaming(server, name)

        slept = admin.operation(server, "sleep")
        assert slept.status_code == 200, slept.text
        status = admin.status(server)
        assert status["deployment"]["asleep"] is True, status
        # The workers give back every segment of the KV cache, which the
        # GPU's quota reserves no more.
        asleep = _bytes_of(server, "cache_segment", epoch)
        assert asleep == 0, admin.reservations(server)
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

        woke = admin.operation(server, "wake_up")
        assert woke.status_code == 200, woke.text
        waiting.join()
        assert answered and answered[0], answered
        _assert_counts_from_one(stream, 200)
        # The wake creates the KV cache again, as large as it was.
        woken = _bytes_of(server, "cache_segment", epoch)
        print(f"KV cache: awake {awake} B, asleep {asleep} B, woken {woken} B")
        assert woken == awake, (awake, woken)
        # Alone, as before the sleep, the deployment answers as it did: the
        # request that waited ran beside the recomputed count, whose batch
        # changes the numerics.
        assert _ask(server.base_url, name, prompt, 32) == before
        status = admin.status(server)
        assert status["deployment"]["asleep"] is False, status
        assert status["deployment"]["paused"] is False, status


def _prepared_standby(server: Server) -> list[int]:
    """The launch generations of the parked standby, which parks once it
    captured its graphs; empty while there is none.
    """  # noqa: D205
    return admin.status(server)["deployment"]["standby"]


def _wait_for_standby(
    server: Server, replaced: list[int] | None = None
) -> list[int]:
    """Wait until a standby parked, other than `replaced`; its launch
    generations.

    No event announces a standby's parking (the event log states operations,
    the serving engine's lifecycle, update phases and deployment changes), and
    the status states its launch generations once it parked.
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
    serving = admin.status(server)["deployment"]
    standby_generations = serving["standby"]
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
    generation = admin.status(server)["deployment"]["generation"]
    promoted = bool(standby_generations) and (
        generation == standby_generations[0]
    )
    relaunched = generation not in (
        None,
        serving["generation"],
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
    options = (*OPTIONS, "worker_config.standby=true")
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
    options = (*OPTIONS, "worker_config.standby=true")
    revision = updates.engine_revision(deployments.build_server, repository)
    with _serve(deployments, name, options=options) as server:
        first = _wait_for_standby(server)
        _change_epoch(server, revision)
        _wait_for_standby(server, replaced=first)
        evidence = _lose_the_serving_engine(
            server, name, "standby-after-a-switch", standby=True
        )
        assert evidence["promoted"]


def _rollback(server: Server) -> dict:
    """Roll the update on trial back; the update once it entered
    ``rolled_back``, which serves its kept side again.
    """  # noqa: D205
    before = admin.next_event(server)
    rolled = admin.operation(server, "config/rollback", {"reason": "the check"})
    assert rolled.status_code == 200, rolled.text
    return admin.wait_for_phase(server, "rolled_back", after=before)


def test_a_relaunched_epoch_hands_its_own_storage_to_the_next_target(
    deployments: Deployments, repository: Path
):
    """A process started again keys the storage it creates by its own epoch.

    The epoch that replaces it maps that storage. So it does after a target's
    process is lost, when the first epoch's storage is gone with it, and after a
    rollback that started the epoch again on the weights it kept.
    """
    name = "qwen3-live-relaunched-storage"
    revisions = [
        updates.engine_revision(deployments.build_server, repository)
        for _ in range(3)
    ]
    with _serve(deployments, name) as server:
        answer = _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32)
        _change_epoch(server, revisions[0])
        _lose_the_serving_engine(
            server, name, "relaunched-after-a-switch", standby=False
        )
        result = _apply(
            server, revisions[1], "update_config.previous_epoch=weights"
        )
        assert result["change"] == "epoch", result
        _rollback(server)
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
    revision = updates.engine_revision(deployments.build_server, repository)
    with _serve(deployments, name) as server:
        before = admin.engine_pids(server)
        failed = _failed_requests(server.base_url)
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
        assert _failed_requests(server.base_url) == failed, "a request failed"


def _slower_target(
    build_server: BuildServer, repository: Path, directory: Path
) -> tuple[str, ...]:
    """Installs what makes an epoch step slower under the same stream.

    The target is a new revision of the tree's engine whose decode attention
    is a package of its own: the tree's trtllm-gen paged attention package
    under the backend name ``trtllm_repeated``, whose entry runs the kernel
    several times per call (``packages/paged_attention.toml``, ``repeated``).
    The target computes the same tokens, later, with no fault to report. The
    engine revision is what makes the change an epoch (a package selection
    alone is a generation update).

    Both modules are built here, for the decode keys the serving deployment's
    attention runs: an update's preparation then builds nothing, and a stream
    started after this still runs when the switch comes.

    Returns:
        The assignments that select the target.
    """
    engine = updates.engine_revision(build_server, repository)
    repeated = build_server.install(
        updates.derived(repository, "paged_attention", "repeated", directory)
    )

    # The decode keys the serving package's modules were built for, whatever
    # its revision: the store holds a module per key a deployment launched.
    decode = sorted(
        {
            module["descriptor"]
            for package in build_server.list()
            if (package["op"], package["backend"])
            == ("paged_attention", "trtllm")
            for module in package["modules"]
            if module["descriptor"].startswith("phase_decode_")
        }
    )
    assert decode, "the deployment's attention built no decode key"
    for key in decode:
        build_server.resolve(
            repeated.op, repeated.backend, repeated.revision, key=key
        )
    return (engine, f"decode_attention_backend={repeated.backend}")


@pytest.mark.parametrize("previous_epoch", ["parked", "weights", "host"])
def test_an_operator_rollback_returns_to_the_previous_epoch_under_a_request(
    deployments: Deployments, repository: Path, previous_epoch: str
):
    """A rollback by hand ends the trial with no request lost.

    The previous epoch is kept as ``update_config.previous_epoch`` states: its
    drained processes stay alive through the trial, or they exit with its
    weights kept, and the rollback starts them again on those.
    """
    name = "qwen3-live-rollback"
    revision = updates.engine_revision(deployments.build_server, repository)
    with _serve(deployments, name) as server:
        before = admin.engine_pids(server)
        previous = admin.config(server)["executing"]["epoch"]
        answer = _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32)
        stream = _start_count(server, name)
        result = _apply(
            server, revision, f"update_config.previous_epoch={previous_epoch}"
        )
        assert result["change"] == "epoch", result
        if previous_epoch != "parked":
            # The previous epoch's processes end.
            for pid in before:
                updates.wait_for_end(pid)
        else:
            assert all(updates.running(pid) for pid in before)

        # A node that cannot move the weights keeps them on its GPUs, and says
        # why.
        trial = admin.config(server)["update"]
        print("kept:", json.dumps(trial))
        if not trial["host_refused"]:
            assert trial["previous_epoch_kept"] == previous_epoch, trial
        if previous_epoch == "weights":
            # The previous epoch keeps its weights alone: each GPU's quota
            # reserves nothing else of it once its processes exited, the
            # segments the target maps being the target's. The Storage Server
            # releases a process's reservations once it saw the process end,
            # and no event states a quota's change.
            def kept() -> set[str]:
                return {
                    entry["item"]
                    for entry in admin.reservations(server)
                    if previous in entry["epochs"]
                }

            server.wait_until(
                lambda: kept() == {"weights"},
                ADMITTED_S,
                "the previous epoch kept more than its weights",
            )
        before_rollback = admin.next_event(server)
        rolled = admin.operation(
            server, "config/rollback", {"reason": "the check"}
        )
        assert rolled.status_code == 200, rolled.text
        _assert_whole(stream)
        # The previous epoch serves again once its processes stopped for the
        # trial started again; the target serves until then.
        update = admin.wait_for_phase(
            server, "rolled_back", after=before_rollback
        )
        assert update["failure"]["reason"] == "the check", update
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
    ends unavailable with the threshold's reason, the status states the rates
    that decided it, and applying it again needs ``force``. Forced without
    the threshold, the same slower target stands its trial: the threshold,
    and nothing else, rolled it back.
    """
    name = "qwen3-live-regression"
    # The step time past which the trial rolls back, as a fraction of the
    # previous epoch's steps up to the switch: above what the target's loading
    # and capture on the same GPU add to those steps, below what the repeated
    # attention adds to the target's.
    regression = 0.2
    thresholds = (
        f"update_config.rollback.step_time_regression={regression}",
        f'update_config.rollback.window="{TRIAL_WINDOW_S}s"',
    )
    slower = _slower_target(deployments.build_server, repository, tmp_path)
    with _serve(deployments, name) as server:
        answer = _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32)
        # The stream runs through the preparation, the switch, a whole window
        # of the trial and the rollback's drain, which the count outlasts only
        # when it starts with the update. The preparation's own seconds give
        # the baseline its window.
        stream = _start_count(server, name)
        before = admin.next_event(server)
        result = _apply(server, *slower, *thresholds)
        assert result["change"] == "epoch", result
        update = admin.wait_for_phase(server, "rolled_back", after=before)
        assert update["failure"]["kind"] == "step_time_regression", update
        print("rolled back:", json.dumps(update))
        # The status states the rates the decision read.
        (replica,) = update["trial"]["replicas"]
        assert replica["baseline"]["tokens_per_second"] > 0, update
        baseline = replica["baseline"]["step_us"]
        assert replica["rates"]["step_us"] > baseline * (1 + regression), update
        _assert_whole(stream)
        assert _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32) == (
            answer
        )

        # The rollback marked the target unavailable, the deployment's only
        # one: applying it again is refused without `force`.
        again = admin.apply(server, *slower)
        assert again.status_code != 200, again.text
        unavailable = admin.config(server)["unavailable"]
        assert len(unavailable) == 1, unavailable

        stream = _start_count(server, name)
        forced = _apply(server, *slower, force=True)
        assert forced["update"]["phase"] == "trial", forced

        def stood_or_ended() -> bool:
            report = admin.config(server)["update"]
            return (
                report["phase"] != "trial"
                or (report["trial_s"] or 0) >= 2 * TRIAL_WINDOW_S
            )

        # The forced target stands two windows of its trial, which its report
        # states (`trial_s`); no event marks a trial's age. The report reads
        # two windows once two windows passed, and a third bounds the reading.
        server.wait_until(
            stood_or_ended,
            3 * TRIAL_WINDOW_S,
            "the forced target's trial reported no two windows",
        )
        unthresholded = admin.config(server)["update"]
        assert unthresholded["phase"] == "trial", unthresholded
        _commit(server)
        _assert_whole(stream)


def test_a_package_update_past_a_threshold_rolls_back_in_its_epoch(
    deployments: Deployments,
):
    """A generation update stands the same trial as an epoch switch.

    The serving epoch computes a stream before the update and a trickle after
    it, so its token throughput falls past the threshold. The update rolls
    back in the same main process, with the threshold as its failure.
    """
    name = "qwen3-live-package-trial"
    with _serve(deployments, name) as server:
        stream = _start_count(server, name)
        # The baseline is the window of steps before the switch: the count's
        # own deltas say when it has streamed for one.
        stream.wait_for(
            lambda streamed: (
                streamed.arrivals[-1] - streamed.arrivals[0] > TRIAL_WINDOW_S
            ),
            "the count did not stream for a window",
        )
        before = admin.next_event(server)
        result = _apply(
            server,
            "scheduler_config.max_batch=4",
            "update_config.rollback.throughput_regression=0.5",
            f'update_config.rollback.window="{TRIAL_WINDOW_S}s"',
        )
        assert result["change"] == "generation", result
        _assert_whole(stream)

        # A trickle of small requests keeps the trial reading steps, one
        # between two reads of the event log, which end at the update's end.
        deadline = time.monotonic() + server.scaled(ROLLBACK_S)
        update = None
        while update is None:
            assert time.monotonic() < deadline, "the update was not rolled back"
            _ask(server.base_url, name, RECOVERY_PROMPT, tokens=2)
            page = admin.events(server, before, TRICKLE_S)
            assert not page["gap"], page
            for event in page["events"]:
                if event["kind"] == "update" and event["data"]["phase"] in (
                    "committed",
                    "rolled_back",
                ):
                    update = event["data"]
            before = page["next"]
        print("rolled back:", json.dumps(update))
        assert update["phase"] == "rolled_back", update
        assert update["failure"]["kind"] == "throughput_regression", update


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
        copied = metric(
            server.base_url, "uniserve:worker_stream_us", stream="copy"
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
        ("expert", ("data_parallel_size=2", "moe_config.expert_parallel=true")),
        ("data", ("data_parallel_size=2",)),
    ],
    ids=lambda parallel: parallel[0],
)
def test_a_process_of_two_devices_keeps_the_host_kv_cache(
    deployments: Deployments, parallel: tuple[str, tuple[str, ...]]
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
    with _serve(deployments, name, options=(*OPTIONS, *options)) as server:
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
        (),
        ("kv_cache_config.host_tier=false",),
        ("kv_cache_config.prefix_caching=false",),
    ],
    ids=["host", "device", "uncached"],
)
def test_requests_that_outgrow_the_cache_in_decode_all_finish(
    deployments: Deployments, tiers: tuple[str, ...]
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
    options = (*SIZED_OPTIONS, "kv_cache_config.max_tokens=8192", *tiers)
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
        updated = admin.operation(
            server, "update_weights", {"model_path": other}
        )
        assert updated.status_code == 200, updated.text
        print("updated:", updated.text[:400])
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

    With ``in_flight = keep`` the batches go on while the new checkpoint loads
    beside the KV cache, and the weights are put in place at one batch boundary
    (``docs/admin.md``, the KV cache's storage and the weights), so the
    stream's longest silence from its last token before the update to its end
    stays below the update's duration, and the stream is whole.

    The weights put in place are mapped at the addresses the captured graphs
    read, which a process subscribed to CUPTI follows by instantiating each
    graph's executable again (``profiling::storage_remapped``); that renewal
    belongs to each graph's next replay and never holds a batch for every
    bucket the runner captured.
    """
    other = CHECKPOINTS["qwen3-other-weights"].model
    name = "qwen3-live-weights-stream"
    with _serve(deployments, name) as server:
        stream = _start_count(server, name)
        started = time.monotonic()
        updated = admin.operation(
            server, "update_weights", {"model_path": other}
        )
        took = time.monotonic() - started
        assert updated.status_code == 200, updated.text
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
    the target serves. The rollback returns them and starts the epoch on them,
    under a stream that crosses both moves whole, and it answers as it did.
    """
    other = CHECKPOINTS["qwen3-other-weights"].model
    name = "qwen3-live-host-weights"
    # The target loads its checkpoint beside the serving one, and the
    # deployment's quota of the GPU holds both only past the default share.
    options = (*OPTIONS, "mem_fraction_static=0.9")
    with _serve(deployments, name, options=options) as server:
        answer = _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32)
        stream = _start_count(server, name)
        result = _apply(
            server, f"model={other}", "update_config.previous_epoch=host"
        )
        assert result["change"] == "epoch", result

        def moved_or_refused() -> bool:
            report = admin.config(server)["update"]
            return report["moved_to_host_bytes"] > 0 or bool(
                report["host_refused"]
            )

        # The weights move once the previous epoch's processes exited, during
        # the trial; its report states the bytes moved, and no event states
        # the move.
        server.wait_until(
            moved_or_refused,
            ADMITTED_S,
            "the previous epoch's weights did not move",
        )
        trial = admin.config(server)["update"]
        print("kept:", json.dumps(trial))
        assert not trial["host_refused"], trial
        assert trial["previous_epoch_kept"] == "host", trial
        update = _rollback(server)
        _assert_whole(stream)
        assert _ask(server.base_url, name, RECOVERY_PROMPT, tokens=32) == (
            answer
        )
        assert update["phase"] == "rolled_back", update


def test_package_updates_under_a_request_keep_its_stream_whole(
    deployments: Deployments,
):
    """In-place switches under a request leave its stream as a run without them.

    A generation update reaches a process that serves: its switch captures
    nothing a batch of the running request can be mistaken for, so every token
    the request computes is the one its stream delivers. The count crosses as
    many switches as fit under it, each one committed before the next starts.
    """
    name = "qwen3-live-switches"
    switches = 0
    with _serve(deployments, name) as server:
        for _ in range(2):
            stream = _start_count(server, name)
            while not stream.ended:
                switches += 1
                size = 4 if switches % 2 else 8
                result = _apply(server, f"scheduler_config.max_batch={size}")
                assert result["change"] == "generation", result
                _commit(server)
            _assert_whole(stream)
    print("switches:", switches)
    assert switches >= 20, switches
