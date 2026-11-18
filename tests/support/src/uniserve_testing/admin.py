"""A daemon's admin API (``docs/admin.md``), through its admin socket.

The daemon serves two admin routes on a Unix socket of its own, which its
readiness record states (:attr:`uniserve_testing.servers.Server.admin_socket`):

- ``/admin/policy``, the deployment's configuration, its policy package:
  ``GET`` reads the current revision's (:func:`policy`) and ``PUT``
  replaces it whole (:func:`apply`), answering once the change converged,
  rolled back or failed with the revision's record. Every write a case makes
  is an option of the package: :func:`change` applies options over the
  current package, and :func:`changing` does so beside the case.
- ``/admin/query``, the deployment's observation store, read with SQL
  (:func:`query`): the table ``stats`` holds one value of each metric series
  per second, and ``events`` every request, revision, process, span and log
  line. :func:`gauge` reads a gauge's latest values and :func:`counter` sums
  a counter; :func:`events` reads the events after a moment of the daemon's
  clock (:func:`now`), and :func:`wait_for` and :func:`wait_for_log` wait
  for one, so a case reads the clock before it causes what it waits for and
  misses no event that came first.

Beside them, what several cases read of what the deployment executes: the
state of a replica (:func:`replica`), the latest state of every process it
launched (:func:`processes`) and the main process that serves
(:func:`serving_engine`), and the daemon that serves the admin socket
(:func:`daemon_pid`), which a hot restart replaces.
"""

from __future__ import annotations

import json
import socket
import struct
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import httpx

from uniserve_testing import packages
from uniserve_testing.packages import Config, merged, without
from uniserve_testing.servers import Server, log_tail

#: The admin route of the deployment's policy package.
POLICY = "/admin/policy"

#: The admin route of the deployment's observation store.
QUERY = "/admin/query"

#: How often a wait reads the events again, in seconds: the store's series
#: are one value per second, and an event is queryable once appended.
POLL_S = 0.5

#: The span of the daemon's clock a gauge's values are read over when no
#: moment bounds them, in seconds: the leader samples its gauges every
#: second, and a scheduler or a worker reports its own once a report period
#: (a second unless stated).
SPAN_S = 3.0

#: How long a gauge takes to state a second after a given moment, in
#: seconds: the leader samples its gauges every second, and a scheduler or
#: a worker reports its own once a report period (a second unless stated).
SAMPLED_S = 30.0

#: The gauge the leader states in every second it samples, whatever the
#: deployment serves: a whole second holding it is a second the leader's
#: status gauges were sampled in (``crates/daemon/src/status.rs``).
SAMPLED = "uniserve_requests_held"

#: The states of a process that no longer runs (``process`` events).
ENDED = ("stopped", "failed")


def request(
    server: Server,
    method: str,
    path: str,
    *,
    content: bytes | None = None,
    data: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    timeout_s: float | None = None,
) -> httpx.Response:
    """`method` on the admin route `path` (``/admin/...``) of the daemon
    `server`, with the raw `content` or the form `data`, waiting at most
    `timeout_s` seconds (scaled) for the answer, or for as long as it takes
    when unset: a change answers once it ended.
    """  # noqa: D205
    transport = httpx.HTTPTransport(uds=str(server.admin_socket))
    timeout = server.scaled(timeout_s) if timeout_s is not None else None
    with httpx.Client(
        transport=transport, base_url="http://admin", timeout=timeout
    ) as client:
        return client.request(
            method,
            path,
            content=content,
            data=data,
            headers=headers,
            params=params,
        )


class Policy(NamedTuple):
    """A revision of the deployment's policy package: its configuration and
    its number (the ``ETag`` of ``GET /admin/policy``).
    """  # noqa: D205

    config: dict[str, Any]
    revision: int


def policy(server: Server, revision: int | None = None) -> Policy:
    """``GET /admin/policy[?revision=N]``: the current revision's package,
    or revision `revision`'s, with its number.
    """  # noqa: D205
    response = request(
        server,
        "GET",
        POLICY,
        params={"revision": revision} if revision is not None else None,
        timeout_s=60,
    )
    response.raise_for_status()
    return Policy(
        packages.unpacked(response.content),
        int(response.headers["etag"].strip('"')),
    )


def apply(
    server: Server,
    config: Config,
    *,
    programs: Path | None = None,
    force: bool = False,
    revision: int | None = None,
) -> httpx.Response:
    """``PUT /admin/policy`` of the package of `config` and the program
    sources under `programs`, which replaces the deployment's target whole:
    what it does not state takes its default. With `force` it applies a
    target a rollback marked unavailable; with `revision` it applies only
    while the current revision is that one (``If-Match``), answered ``412``
    otherwise.

    Answered once the change converged (``200``), its daemon restart's
    successor converges the rest (``202``), or it rolled back or was refused
    (an error status with the reason); a converged or converging answer is
    the revision's record (``revision``, ``writer``, ``effect``, ``result``,
    ``changes``, ...).
    """  # noqa: D205
    headers = {"if-match": f'"{revision}"'} if revision is not None else None
    return request(
        server,
        "PUT",
        POLICY,
        content=packages.archive(config, programs),
        headers=headers,
        params={"force": "true"} if force else None,
    )


def change(
    server: Server,
    changes: Config,
    *,
    removed: Iterable[str] = (),
    force: bool = False,
) -> httpx.Response:
    """Apply `changes` over the deployment's current package: the package
    :func:`policy` reads less the options `removed` names (dotted paths,
    such as ``profiling_config.session``, which then take their defaults)
    and with `changes` merged over it
    (:func:`uniserve_testing.packages.merged`), applied only while that
    revision is current (:func:`apply`).
    """  # noqa: D205
    current = policy(server)
    return apply(
        server,
        merged(without(current.config, *removed), changes),
        force=force,
        revision=current.revision,
    )


@contextmanager
def changing(
    server: Server, changes: Config, *, force: bool = False
) -> Iterator[Future[httpx.Response]]:
    """:func:`change` on a thread of its own while the block runs: the
    future of its answer, which arrives once the change converged or ended,
    so the case acts on the deployment while it changes (a stream across a
    switch, a fault during a trial). The block's end waits for the answer.
    """  # noqa: D205
    with ThreadPoolExecutor(max_workers=1) as pool:
        applied = pool.submit(change, server, changes, force=force)
        try:
            yield applied
        finally:
            # A failed block still waits for the change, whose answer bounds
            # the thread's life.
            applied.exception()


def revision(response: httpx.Response) -> dict[str, Any]:
    """The revision record of a change the deployment converged to: its
    answer is ``200`` with the record.
    """  # noqa: D205
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["result"] == "converged", record
    return record


def flush(server: Server) -> dict[str, Any]:
    """Revoke every cached prefix page at the engines' next step: a change
    of the package's ``flush`` number, answered once it converged; the
    revision's record.
    """  # noqa: D205
    stated = policy(server).config.get("flush", 0)
    return revision(change(server, {"flush": stated + 1}))


def query(server: Server, sql: str) -> list[dict[str, Any]]:
    """The rows of the statement `sql` over the deployment's observation
    store (``POST /admin/query``), each a mapping of its columns: a map
    column (``labels``, ``fields``) is a mapping of strings, a time an RFC
    3339 string unless the statement casts it.

    Raises:
        AssertionError: With the daemon's reason, when it refused the
            statement.
    """  # noqa: D205
    response = request(server, "POST", QUERY, data={"q": sql}, timeout_s=60)
    assert response.status_code == 200, f"{sql}: {response.text}"
    return response.json()


def literal(text: str) -> str:
    """`text` as an SQL string literal."""
    escaped = text.replace("'", "''")
    return f"'{escaped}'"


def matching(column: str, values: Mapping[str, str]) -> str:
    """The SQL condition that the map column `column` holds every entry of
    `values`; ``TRUE`` for none.
    """  # noqa: D205
    return " AND ".join(
        [
            f"{column}[{literal(key)}] = {literal(value)}"
            for key, value in values.items()
        ]
        or ["TRUE"]
    )


def now(server: Server) -> int:
    """The daemon's clock, in nanoseconds since the Unix epoch: the moment
    after which a case reads the events it causes next.
    """  # noqa: D205
    [row] = query(server, "SELECT CAST(now() AS BIGINT) AS now")
    return int(row["now"])


class Sample(NamedTuple):
    """One series' value of a metric: its labels and its value."""

    labels: dict[str, str]
    value: float


def gauge(
    server: Server, name: str, *, after: int | None = None, **labels: str
) -> list[Sample]:
    """The values of the gauge `name` in the latest whole second that holds
    a value of one of its series holding every label of `labels`: one value
    per series stated then, among the seconds after the moment `after` of
    the daemon's clock (:func:`now`, an event's time), or among the last
    :data:`SPAN_S` seconds when unstated; none when no such second holds
    one.

    The second the daemon's clock is in is left out, since its values may
    still be arriving: the leader states every gauge it samples in each
    earlier second. With `after`, the values are what the gauge stated of a
    state an event announced: the read waits, within :data:`SAMPLED_S`
    seconds (scaled), until a whole second after `after` holds a value of
    the gauge, whatever its labels, and a series no longer stated by then
    is none of them. Without it, the read waits the same way until a whole
    second of the span holds the leader's sampling (:data:`SAMPLED`), so a
    deployment that just began to serve is read once its first second was
    sampled. A gauge stated only when what it states changes is read with
    ``after=0``, over the store's whole retention.

    Raises:
        TimeoutError: When no whole second after `after` was recorded in
            time.
    """  # noqa: D205
    lower = (
        str(int(after))
        if after is not None
        else f"CAST(now() AS BIGINT) - {int(SPAN_S * 1e9)}"
    )
    window = (
        f"name = {literal(name)} AND CAST(time AS BIGINT) > {lower} "
        "AND time < date_trunc('second', now())"
    )
    # The second that must be whole before the gauge is read: one holding
    # the gauge after `after`, or one of the span holding the leader's
    # sampling.
    awaited = (
        window
        if after is not None
        else (
            f"name = {literal(SAMPLED)} AND CAST(time AS BIGINT) > {lower} "
            "AND time < date_trunc('second', now())"
        )
    )
    deadline = time.monotonic() + server.scaled(SAMPLED_S)
    while not query(
        server, f"SELECT 1 AS stated FROM stats WHERE {awaited} LIMIT 1"
    ):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"no value of {name if after is not None else SAMPLED} was "
                f"recorded after {lower} within "
                f"{server.scaled(SAMPLED_S):.0f} s"
            )
        time.sleep(POLL_S)

    rows = query(
        server,
        "SELECT CAST(time AS BIGINT) AS time, labels, value FROM stats "
        f"WHERE {window} AND {matching('labels', labels)} "
        "AND time = (SELECT max(time) FROM stats "
        f"WHERE {window} AND {matching('labels', labels)}) "
        "ORDER BY revision",
    )
    # A series is also its configuration revision: of a label set written
    # under two revisions in one second, the later revision's value holds.
    latest = {
        tuple(sorted(row["labels"].items())): Sample(
            row["labels"], float(row["value"])
        )
        for row in rows
    }
    return list(latest.values())


def counter(server: Server, name: str, **labels: str) -> float:
    """The counter `name`'s total over the store's retention, summed over
    the series that hold every label of `labels`, such as ``mode="eager"``.
    """  # noqa: D205
    [row] = query(
        server,
        f"SELECT sum(value) AS total FROM stats WHERE name = {literal(name)} "
        f"AND {matching('labels', labels)}",
    )
    return float(row["total"] or 0.0)


class Event(NamedTuple):
    """A row of ``events``: when the store recorded it (nanoseconds since
    the Unix epoch, a cursor for the events after it), its labels and its
    fields, each field that is JSON text decoded.
    """  # noqa: D205

    time: int
    labels: dict[str, str]
    fields: dict[str, Any]


def decoded(text: str | None) -> Any:
    """A field's value: the value of its JSON text (an event's field that is
    not a string is written as its JSON text), or the string itself.
    """  # noqa: D205
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def events(
    server: Server, kind: str, *, after: int = 0, **labels: str
) -> list[Event]:
    """The events of `kind` (``request``, ``revision``, ``process``,
    ``span``, ``log``) the store recorded after the moment `after`
    (:func:`now`) that hold every label of `labels`, oldest first.
    """  # noqa: D205
    rows = query(
        server,
        "SELECT CAST(time AS BIGINT) AS time, labels, fields FROM events "
        f"WHERE kind = {literal(kind)} AND {matching('labels', labels)} "
        f"AND CAST(time AS BIGINT) > {int(after)} ORDER BY 1",
    )
    return [
        Event(
            int(row["time"]),
            row["labels"],
            {key: decoded(value) for key, value in row["fields"].items()},
        )
        for row in rows
    ]


def wait_for(
    server: Server,
    kind: str,
    matches: Callable[[Event], bool],
    *,
    after: int,
    timeout_s: float,
    what: str,
    **labels: str,
) -> Event:
    """The first event of `kind` holding `labels` after the moment `after`
    that `matches`, within `timeout_s` seconds (scaled). `after` is the
    daemon's clock (:func:`now`) the caller read before it caused the event,
    so an event that came first is not missed.

    Raises:
        TimeoutError: Naming `what`, with the daemon's log tail.
    """  # noqa: D205
    deadline = time.monotonic() + server.scaled(timeout_s)
    while True:
        for event in events(server, kind, after=after, **labels):
            if matches(event):
                return event
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{what} within {server.scaled(timeout_s):.0f} s"
                f"{log_tail(server.log_path)}"
            )
        time.sleep(POLL_S)


def logs(
    server: Server, source: str, *, after: int = 0, **labels: str
) -> list[Event]:
    """The ``log`` events of the writer `source` (``node``, ``engine``,
    ``update``, ``profile``) after the moment `after` holding `labels`,
    oldest first.
    """  # noqa: D205
    return events(server, "log", after=after, source=source, **labels)


def wait_for_log(
    server: Server,
    source: str,
    *,
    after: int,
    timeout_s: float = 900,
    what: str | None = None,
    labels: Mapping[str, str] | None = None,
    **fields: str | tuple[str, ...],
) -> Event:
    """The first ``log`` event of the writer `source` after the moment
    `after` holding `labels` whose every field of `fields` is the value
    stated, or one of the values a tuple states: an update entering one of
    its phases (``wait_for_log(server, "update", after=t, phase=("trial",
    "rolled_back"))``), an engine's lifecycle change, a profiling session's
    state.
    """  # noqa: D205

    def held(event: Event) -> bool:
        return all(
            event.fields.get(key)
            in (wanted if isinstance(wanted, tuple) else (wanted,))
            for key, wanted in fields.items()
        )

    stated = ", ".join(f"{key}={value}" for key, value in fields.items())
    return wait_for(
        server,
        "log",
        held,
        after=after,
        timeout_s=timeout_s,
        what=what or f"no {source} log of {stated}",
        source=source,
        **(labels or {}),
    )


class Replica(NamedTuple):
    """A replica as the leader states it each second: the lifecycle state of
    its engine (``serving``, ``parked``, ``draining``, ...), the epoch it
    executes, the launch generation of the engine that serves (``None``
    while none does) and those of its parked standby's members, member 0
    first (the one a promotion serves; empty without a standby).
    """  # noqa: D205

    state: str | None
    epoch: int | None
    generation: int | None
    standby: list[int]


def replica(server: Server, ordinal: int = 0) -> Replica:
    """Replica `ordinal`'s :class:`Replica`, from the leader's gauges."""
    at = str(ordinal)
    states = [
        sample.labels["state"]
        for sample in gauge(server, "uniserve_replica_state", replica=at)
        if sample.value == 1
    ]
    epochs = gauge(server, "uniserve_replica_epoch", replica=at)
    generations = gauge(server, "uniserve_replica_generation", replica=at)
    standby = sorted(
        gauge(server, "uniserve_replica_standby_generation", replica=at),
        key=lambda sample: int(sample.labels["member"]),
    )
    return Replica(
        state=states[0] if states else None,
        epoch=int(epochs[0].value) if epochs else None,
        generation=int(generations[0].value) if generations else None,
        standby=[int(sample.value) for sample in standby],
    )


def processes(server: Server, **labels: str) -> list[dict[str, Any]]:
    """The latest state of every process the deployment launched whose
    ``process`` events hold `labels` (``host``, ``group``, ``rank``), one
    launch each: its labels and its fields (``generation``, ``pid``,
    ``epoch``, ``state``, ``reason``) in one mapping.
    """  # noqa: D205
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in events(server, "process", **labels):
        launch = (
            event.labels.get("host"),
            event.labels.get("group"),
            event.labels.get("rank"),
            event.fields.get("generation"),
        )
        latest[launch] = {**event.labels, **event.fields}
    return list(latest.values())


def serving_engine(server: Server) -> dict[str, Any]:
    """The main process that serves: the latest state (:func:`processes`)
    of the launch of group ``engine`` whose generation replica 0 serves.
    """  # noqa: D205
    generation = replica(server).generation
    assert generation is not None, "no main process serves"
    (serving,) = [
        process
        for process in processes(server, group="engine")
        if process["generation"] == generation
    ]
    return serving


def daemon_pid(server: Server) -> int:
    """The process of the daemon that serves `server`'s admin socket now: the
    credentials of its listener (``SO_PEERCRED``). A hot restart's successor
    binds the socket over its predecessor's, so this names the successor
    once it serves there.
    """  # noqa: D205
    credentials = struct.Struct("3i")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(server.admin_socket))
        pid, _, _ = credentials.unpack(
            connection.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, credentials.size
            )
        )
    return pid
