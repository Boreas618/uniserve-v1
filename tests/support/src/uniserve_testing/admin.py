"""A daemon's admin API (``docs/admin.md``), through its operator socket.

The daemon serves its admin routes on a Unix socket of its own, which its
readiness record states (:attr:`uniserve_testing.servers.Server.admin_socket`).
The deployment's configuration is its policy package: ``GET /admin/policy``
reads the current revision's (:func:`policy`) and ``PUT /admin/policy``
replaces it whole (:func:`apply`), answering once the change converged,
rolled back or failed with the revision's record; :func:`change` applies
options over the current package, and :func:`changing` does so beside the
case. What the deployment executes and owns is the status's ``report``
(:func:`report`). Each other write is an operation that answers once it
ended, and the node's event log (``GET /admin/events``) states every
operation, every change of the serving engine's lifecycle and every phase
an update enters, so a test waits for those on the log
(:func:`wait_for_event`) instead of reading the status repeatedly.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import httpx

from uniserve_testing import packages
from uniserve_testing.packages import Config, merged
from uniserve_testing.servers import Server, log_tail

#: How long one read of the event log waits for an event, in seconds; the
#: daemon caps it at ``admin_config.event_log.longest_wait``.
EVENT_WAIT_S = 30

#: The admin route of the deployment's policy package.
POLICY = "/admin/policy"


def request(
    server: Server,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> httpx.Response:
    """`method` on the admin route `path` (``/admin/...``) of the daemon
    `server`, with the JSON `body` or the raw `content`, waiting at most
    `timeout_s` seconds (scaled) for the answer, or for as long as it takes
    when unset: an operation answers once it ended.
    """  # noqa: D205
    transport = httpx.HTTPTransport(uds=str(server.admin_socket))
    timeout = server.scaled(timeout_s) if timeout_s is not None else None
    with httpx.Client(
        transport=transport, base_url="http://admin", timeout=timeout
    ) as client:
        return client.request(
            method,
            path,
            json=body,
            content=content,
            headers=headers,
            params=params,
        )


def operation(
    server: Server, path: str, body: dict[str, Any] | None = None
) -> httpx.Response:
    """``POST /admin/<path>`` with `body`, answered once the operation ends."""
    return request(server, "POST", f"/admin/{path}", body=body or {})


def status(server: Server) -> dict[str, Any]:
    """``GET /admin/status``: what the node serves and its operations."""
    response = request(server, "GET", "/admin/status", timeout_s=60)
    response.raise_for_status()
    return response.json()


def report(server: Server) -> dict[str, Any]:
    """What the serving deployment executes and owns, the status's
    ``report``: the first replica's configuration (``config``) and the epoch
    it executes (``executing``), every replica's (``replicas``), the update
    in flight or the last one (``update``), the targets a rollback marked
    unavailable (``unavailable``), the epoch records (``epochs``), the
    node's processes (``processes``) and the GPUs' quotas (``quotas``).
    """  # noqa: D205
    stated = status(server)
    assert stated["state"] == "serving", stated
    assert stated["report"] is not None, stated
    return stated["report"]


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
    server: Server, changes: Config, *, force: bool = False
) -> httpx.Response:
    """Apply `changes` over the deployment's current package: the package
    :func:`policy` reads with `changes` merged over it
    (:func:`uniserve_testing.packages.merged`), applied only while that
    revision is current (:func:`apply`).
    """  # noqa: D205
    current = policy(server)
    return apply(
        server,
        merged(current.config, changes),
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


def events(server: Server, after: int, wait_s: float) -> dict[str, Any]:
    """``GET /admin/events``: the events after sequence number `after`,
    waiting up to `wait_s` seconds for one.
    """  # noqa: D205
    response = request(
        server,
        "GET",
        "/admin/events",
        params={"after": after, "wait": int(wait_s)},
        timeout_s=wait_s + 30,
    )
    response.raise_for_status()
    return response.json()


def wait_for_event(
    server: Server,
    matches: Callable[[dict[str, Any]], bool],
    timeout_s: float,
    what: str,
    *,
    after: int,
) -> dict[str, Any]:
    """The first event after sequence number `after` that `matches`, read
    from the node's event log as it arrives, within `timeout_s` seconds
    (scaled). `after` is a ``next_event`` the caller read before it caused
    the event (:func:`status`), so an event that came first is not missed.

    Raises:
        TimeoutError: Naming `what`, with the daemon's log tail.
        AssertionError: When events after `after` left the log unread.
    """  # noqa: D205
    deadline = time.monotonic() + server.scaled(timeout_s)
    while (remaining := deadline - time.monotonic()) > 0:
        page = events(server, after, min(EVENT_WAIT_S, remaining))
        assert not page.get("gap"), f"events after {after} left the log"
        for event in page["events"]:
            if matches(event):
                return event
        after = page["next"]
    raise TimeoutError(
        f"{what} within {server.scaled(timeout_s):.0f} s"
        f"{log_tail(server.log_path)}"
    )


def next_event(server: Server) -> int:
    """The sequence number of the node's next event (``next_event``)."""
    return int(status(server)["next_event"])


def wait_for_phase(
    server: Server, *phases: str, after: int, timeout_s: float = 900
) -> dict[str, Any]:
    """The update as it stood when it entered one of `phases`
    (``preparing``, ``switching``, ``trial``, ``committed``,
    ``rolled_back``, ``abandoned``), from its ``update`` event.
    """  # noqa: D205
    event = wait_for_event(
        server,
        lambda event: (
            event["kind"] == "update" and event["data"]["phase"] in phases
        ),
        timeout_s,
        f"the update did not reach {' or '.join(phases)}",
        after=after,
    )
    return event["data"]


def engine_processes(server: Server) -> list[dict[str, Any]]:
    """The deployment's main processes on the leader's node: the rows of
    the report's ``processes`` of group ``engine``, each with its ``pid``,
    launch ``generation`` and ``state``.
    """  # noqa: D205
    return [
        process
        for process in report(server)["processes"]
        if process["group"] == "engine"
    ]


def serving_engine(server: Server) -> dict[str, Any]:
    """The main process that serves: the row of :func:`engine_processes`
    whose launch generation ``GET /admin/status`` names.
    """  # noqa: D205
    generation = status(server)["deployment"]["generation"]
    assert generation is not None, "no main process serves"
    (serving,) = [
        process
        for process in engine_processes(server)
        if process["generation"] == generation
    ]
    return serving


def engine_pids(server: Server) -> set[int]:
    """The process ids of the deployment's main processes on the leader."""
    return {process["pid"] for process in engine_processes(server)}


def quotas(server: Server) -> list[dict[str, Any]]:
    """Every GPU quota of every host of the deployment, as the report states
    them (``quotas``), each holding its ``limit`` and its ``reservations``
    (each with its ``item``, ``epochs`` and ``bytes``).
    """  # noqa: D205
    return [
        quota for node in report(server)["quotas"] for quota in node["quotas"]
    ]


def reservations(server: Server) -> list[dict[str, Any]]:
    """Every reservation of every GPU quota of the deployment."""
    return [
        entry for quota in quotas(server) for entry in quota["reservations"]
    ]
