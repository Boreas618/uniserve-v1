"""The profiling cases' Qwen3 deployment, the decode load they profile, and
the session calls of its admin API (``/admin/profile``, ``docs/profiling.md``).

A profiling module serves one deployment for all of its cases (:func:`serve`,
on the module's lease, with the module's own options over
:data:`OPTIONS`); its cases keep decode steps running with :class:`Load`,
start sessions with :func:`start`, end them with :func:`stop` or wait for
their windows to end with :func:`ended`, read the fence probes' counts
with :func:`fence_steps`, and read a session's file with :func:`table`.
"""  # noqa: D205

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

from uniserve_testing import admin, chat
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments
from uniserve_testing.packages import Config, merged
from uniserve_testing.servers import Server

#: The model name the deployment serves under.
NAME = "qwen3-profiling"

#: The options of every profiling deployment, below its module's own: Qwen3
#: under :data:`NAME`, few captured graphs (decode batches up to 8, prefill
#: buckets of 512 tokens, which bound a step's token work), so that it starts
#: quickly and a session's decode steps replay them, and the reasoning
#: delimiters streamed as content.
OPTIONS: dict[str, Any] = {
    "model": CHECKPOINTS["qwen3"].model,
    "served_model_name": NAME,
    "max_num_batched_tokens": 512,
    "chat_config": {"reasoning_parser": "none"},
    "cuda_graph_config": {
        "prefill_token_sizes": [512],
        "decode_batch_sizes": [1, 2, 4, 8],
    },
}

#: The admin route of the profiling sessions.
PROFILE = "/admin/profile"

#: How long a session's window takes to end by itself, in seconds (scaled):
#: its workers record the window's steps at the pace of the load's decode
#: steps and then drain what CUPTI recorded into their files, which a
#: timeline of every graph node or a full PC sampling buffer makes large.
WINDOW_S = 600.0

#: What the load asks: a count whose answer decodes for as long as a stream
#: reads it, greedily and without a thinking section.
LOAD_REQUEST = chat.body(
    NAME,
    "Count from 1 to 3000, separated by commas and spaces. "
    "Write only the numbers.",
    2048,
    temperature=0,
    stream=True,
    **chat.NO_THINKING,
)


@contextmanager
def serve(deployments: Deployments, config: Config) -> Iterator[Server]:
    """Serve Qwen3 with `config` over :data:`OPTIONS` until the block ends."""
    with deployments.serve(merged(OPTIONS, config)) as server:
        yield server


class Load:
    """Requests that stream the count on threads of their own while the
    block runs, each asked again when it ends: decode steps throughout.

    The block's end stops every stream at its next line and raises what a
    stream raised.
    """  # noqa: D205

    def __init__(self, server: Server, streams: int = 2) -> None:
        """`streams` concurrent streams to the deployment `server`."""
        self._url = f"{server.base_url}/v1/chat/completions"
        self._streams = streams
        self._stop = threading.Event()
        # The streams that read their first line, or ended before one, which
        # `decoding` waits on.
        self._begun = 0
        self._changed = threading.Condition()
        self._executor = ThreadPoolExecutor(max_workers=streams)
        self._futures: list[Future[None]] = []

    def __enter__(self) -> Load:
        """Start the streams."""
        self._futures = [
            self._executor.submit(self._stream) for _ in range(self._streams)
        ]
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the streams and raise the first error one of them raised."""
        self._stop.set()
        self._executor.shutdown(wait=True)
        for future in self._futures:
            future.result()

    def decoding(self) -> None:
        """Return once every stream read its first line, so that the
        deployment decodes the load's requests from now on.
        """  # noqa: D205
        with self._changed:
            self._changed.wait_for(lambda: self._begun == self._streams)

    def _begin(self) -> None:
        with self._changed:
            self._begun += 1
            self._changed.notify_all()

    def _stream(self) -> None:
        """Stream the count, again and again, until the load stops."""
        begun = False
        try:
            while not self._stop.is_set():
                with httpx.stream(
                    "POST",
                    self._url,
                    json=LOAD_REQUEST,
                    timeout=chat.COMPLETION_S,
                ) as response:
                    response.raise_for_status()
                    for _ in response.iter_lines():
                        if not begun:
                            begun = True
                            self._begin()
                        if self._stop.is_set():
                            break
        finally:
            # A stream that failed before its first line releases a waiter
            # too; the block's end raises its error.
            if not begun:
                self._begin()


def start(server: Server, request: dict[str, Any]) -> dict[str, Any]:
    """Start a session of `request` (``POST /admin/profile``), answered once
    every worker prepared its collectors; the session as it started, with
    each worker's answer (``workers``).
    """  # noqa: D205
    response = admin.request(server, "POST", PROFILE, body=request)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def stop(server: Server) -> dict[str, Any]:
    """End the running session (``DELETE /admin/profile``), answered once
    every worker wrote its file and the KV cache grew back into the storage
    the session held; the session as it ended.
    """  # noqa: D205
    response = admin.request(server, "DELETE", PROFILE)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def session(server: Server) -> dict[str, Any] | None:
    """The running session, or the last that ended (``GET
    /admin/profile``); ``None`` before any.
    """  # noqa: D205
    response = admin.request(server, "GET", PROFILE)
    response.raise_for_status()
    return response.json()["session"]


def ended(server: Server, timeout_s: float = WINDOW_S) -> dict[str, Any]:
    """The session once it ended by itself (at its window's end or by its
    guard), within `timeout_s` seconds (scaled).

    No event of the node's log announces a session's end, so its state is
    read every second.
    """  # noqa: D205
    last: dict[str, Any] = {}

    def is_ended() -> bool:
        current = session(server) or {}
        last.clear()
        last.update(current)
        return current.get("state") == "ended"

    server.wait_until(is_ended, timeout_s, "the profiling session did not end")
    return last


def fence_steps(server: Server, stage: str, module: str) -> int:
    """The steps of `stage` the fence probes counted for the module classes
    the pattern `module` names, as a request names them (``*`` matching any
    run), or for the steps' own spans with ``step``: the counts of the
    histogram ``uniserve:worker_fence_step_seconds`` (``docs/profiling.md``,
    Fence probes), summed over its matching label sets.
    """  # noqa: D205
    text = httpx.get(f"{server.base_url}/metrics", timeout=None).text
    total = 0
    for family in text_string_to_metric_families(text):
        if family.name != "uniserve:worker_fence_step_seconds":
            continue
        for sample in family.samples:
            if (
                sample.name.endswith("_count")
                and sample.labels.get("stage") == stage
                and fnmatchcase(sample.labels.get("module", ""), module)
            ):
                total += int(sample.value)
    return total


def table(path: Path, name: str) -> list[dict[str, Any]]:
    """Every row of the table `name` of the session file `path`, opened
    read-only, as a mapping of its columns; none when the file has no such
    table. The file's tables and their identifiers are the schema its
    ``schema_version`` states (``crates/engine/src/profiling/schema.sql``).
    """  # noqa: D205
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }
        if name not in tables:
            return []
        return [dict(row) for row in db.execute(f'SELECT * FROM "{name}"')]
