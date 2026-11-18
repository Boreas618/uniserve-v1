"""The profiling cases' Qwen3 deployment, the decode load they profile, and
its profiling sessions (``docs/profiling.md``).

A session is an option of the deployment's package, the table
``[profiling_config.session]``: while the package states it, the serving
epoch's workers run it, a change of it ends the running session and starts
the new one, and its removal ends it. Each state a session enters is a
``log`` event of source ``profile`` (:mod:`uniserve_testing.admin`).

A profiling module serves one deployment for all of its cases (:func:`serve`,
on the module's lease, with the module's own options over
:data:`OPTIONS`); its cases keep decode steps running with :class:`Load`,
start sessions with :func:`start`, end them with :func:`stop` or wait for
their windows to end with :func:`ended`, read a session's latest state with
:func:`latest`, read the fence probes' counts with :func:`fence_steps`, and
read a session's file with :func:`table`.
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

from uniserve_testing import admin, chat
from uniserve_testing.admin import Event
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

#: The option of the package that states the session the workers run.
SESSION = "profiling_config.session"

#: How long a stated session takes to start, in seconds (scaled): every
#: worker prepares its collectors at its next step boundary.
START_S = 120.0

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


def start(server: Server, request: dict[str, Any]) -> Event:
    """State the session of `request` (:data:`SESSION`), which ends the one
    the package stated before; its ``profile`` event once every worker
    prepared its collectors (``state`` ``running``), whose fields hold each
    worker's answer (``workers``) and whose ``session`` label names it.
    """  # noqa: D205
    after = admin.now(server)
    admin.revision(
        admin.change(server, {"profiling_config": {"session": request}})
    )
    return admin.wait_for_log(
        server,
        "profile",
        after=after,
        timeout_s=START_S,
        what="the profiling session did not start",
        state="running",
    )


def stop(server: Server, started: Event) -> dict[str, Any]:
    """End the session `started` (:func:`start`) by removing it from the
    package; the session once it ended, every worker having written its file
    and the KV cache grown back into the storage the session held.
    """  # noqa: D205
    admin.revision(admin.change(server, {}, removed=(SESSION,)))
    return ended(server, started, START_S)


def ended(
    server: Server, started: Event, timeout_s: float = WINDOW_S
) -> dict[str, Any]:
    """The session `started` (:func:`start`) once it ended, by itself (at
    its window's end or by its guard), by its removal or with its epoch,
    within `timeout_s` seconds (scaled).
    """  # noqa: D205
    return admin.wait_for_log(
        server,
        "profile",
        after=started.time,
        timeout_s=timeout_s,
        what="the profiling session did not end",
        labels={"session": started.labels["session"]},
        state="ended",
    ).fields


def latest(server: Server, started: Event) -> dict[str, Any]:
    """The latest state of the session `started` (:func:`start`), from its
    ``profile`` events: its ``state``, ``guard`` actions, ``files``, ...
    """  # noqa: D205
    states = admin.logs(
        server,
        "profile",
        after=started.time - 1,
        session=started.labels["session"],
    )
    return states[-1].fields


def fence_steps(server: Server, phase: str, module: str) -> int:
    """The steps of `phase` the fence probes counted for the module classes
    the pattern `module` names, as a request names them (``*`` matching any
    run), or for the steps' own spans with ``step``: the observations of the
    histogram ``uniserve_step_device_seconds`` (``docs/profiling.md``, Fence
    probes), summed over its matching series.
    """  # noqa: D205
    rows = admin.query(
        server,
        "SELECT labels['module'] AS module, "
        "sum(histogram['count']) AS steps FROM stats "
        "WHERE name = 'uniserve_step_device_seconds' "
        f"AND labels['phase'] = {admin.literal(phase)} GROUP BY 1",
    )
    return sum(
        int(row["steps"] or 0)
        for row in rows
        if fnmatchcase(row["module"] or "", module)
    )


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
