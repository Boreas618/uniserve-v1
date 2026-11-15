"""A daemon's admin API (``docs/admin.md``), through its operator socket.

The daemon serves its admin routes on a Unix socket of its own, which its
readiness record states (:attr:`uniserve_testing.servers.Server.admin_socket`).
Each write is an operation that answers once it ended unless asked not to
wait, and the node's event log (``GET /admin/events``) states every
operation, every change of the serving engine's lifecycle and every phase
an update enters, so a test waits for those on the log (:func:`wait_for_event`)
instead of reading the status repeatedly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from uniserve_testing.servers import Server, log_tail

#: How long one read of the event log waits for an event, in seconds; the
#: daemon caps it at ``admin_config.event_log.longest_wait``.
EVENT_WAIT_S = 30


def request(
    server: Server,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> httpx.Response:
    """`method` on the admin route `path` (``/admin/...``) of the daemon
    `server`, waiting at most `timeout_s` seconds (scaled) for the answer, or
    for as long as it takes when unset: an operation answers once it ended.
    """  # noqa: D205
    transport = httpx.HTTPTransport(uds=str(server.admin_socket))
    timeout = server.scaled(timeout_s) if timeout_s is not None else None
    with httpx.Client(
        transport=transport, base_url="http://admin", timeout=timeout
    ) as client:
        return client.request(method, path, json=body, params=params)


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


def config(server: Server) -> dict[str, Any]:
    """``GET /admin/config``: the configuration, the epoch that executes
    (``executing``), the update in flight or the last one (``update``), the
    epoch records (``epochs``), the node's processes (``processes``) and the
    quotas (``quotas``).
    """  # noqa: D205
    response = request(server, "GET", "/admin/config", timeout_s=60)
    response.raise_for_status()
    return response.json()


def apply(
    server: Server, *assignments: str, **fields: object
) -> httpx.Response:
    """``PUT /admin/config`` with `assignments` (``set``) and the request's
    other `fields` (``force``, ``in_flight``, ``dry_run``, ``deadline_s``,
    ``wait``): answered, for an update, once it reached its trial or ended
    before one.
    """  # noqa: D205
    return request(
        server,
        "PUT",
        "/admin/config",
        body={"set": list(assignments), **fields},
    )


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
    ``GET /admin/config``'s ``processes`` of group ``engine``, each with its
    ``pid``, launch ``generation`` and ``state``.
    """  # noqa: D205
    return [
        process
        for process in config(server)["processes"]
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
    """Every GPU quota of every host of the deployment, as ``GET
    /admin/config`` reports them (``quotas``), each holding its ``limit``
    and its ``reservations`` (each with its ``item``, ``epochs`` and
    ``bytes``).
    """  # noqa: D205
    return [
        quota for node in config(server)["quotas"] for quota in node["quotas"]
    ]


def reservations(server: Server) -> list[dict[str, Any]]:
    """Every reservation of every GPU quota of the deployment."""
    return [
        entry for quota in quotas(server) for entry in quota["reservations"]
    ]
