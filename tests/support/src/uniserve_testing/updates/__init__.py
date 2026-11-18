"""What the live-update cases share.

The update modules (``test_live_updates_http.py`` and
``test_generation_update_http.py``) serve one Qwen3 deployment and change it
while it serves. They share:

- :data:`PROFILE`, the options of the deployment they serve, so the two
  modules' numbers describe one deployment;
- :func:`derived`, a revision of a serving package that a checked-in
  mutation of ``tests/python/e2e/packages`` declares, and
  :func:`engine_revision`, a new revision of the tree's engine built at the
  run's Build Server, which changes an epoch's engine and nothing else;
  :func:`modules`, what that Build Server built;
- :class:`Stream`, a streamed completion read on a thread of its own that
  wakes its waiters at each frame, so a case acts a number of tokens into a
  response without polling it, and :func:`numbers`, a count's numbers;
- :func:`wait_for_end`, :func:`running` and :func:`terminate`, which
  follow a process through a pidfd (a hot restart's successor daemon, an
  engine the daemon supervises), and :func:`successors`, which ends the
  daemons hot restarts started;
- :func:`listing`, what a Storage Server holds;
- :func:`ballast`, a process that holds a GPU's free storage
  (:mod:`uniserve_testing.updates.ballast`).
"""

from __future__ import annotations

import json
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from uniserve_testing import builds, chat
from uniserve_testing.builds import BuildServer
from uniserve_testing.devices import Lease
from uniserve_testing.servers import environment

#: The options of the deployment both update modules serve: few captured
#: graphs, so that an epoch's engines start in about a minute, and a small
#: device KV cache. Its attention package and its step budget are the
#: deployment's defaults: the installed package of the highest priority for
#: each key, and the widest prefill graph it captures.
PROFILE: dict[str, Any] = {
    "dtype": "bfloat16",
    "kv_cache_config": {"max_tokens": 32768},
    "chat_config": {"reasoning_parser": "none"},
    "cuda_graph_config": {
        "prefill_token_sizes": [512],
        "decode_batch_sizes": [1, 2, 4, 8],
    },
}

#: The checked-in package mutations, relative to the repository: a TOML file
#: per serving package, each declaring the revisions a case derives from it
#: (:func:`uniserve_testing.builds.derived_package`).
MUTATIONS = Path("tests", "python", "e2e", "packages")

#: The program :func:`ballast` runs.
BALLAST = "uniserve_testing.updates.ballast"


def derived(
    repository: Path, mutations: str, variant: str, directory: Path
) -> Path:
    """The revision `variant` of the mutation file ``<mutations>.toml`` of
    :data:`MUTATIONS`, written into `directory`.
    """  # noqa: D205
    return builds.derived_package(
        repository,
        repository / MUTATIONS / f"{mutations}.toml",
        variant,
        directory,
    )


def engine_revision(
    build_server: BuildServer, repository: Path
) -> dict[str, Any]:
    """The options that select a new revision of the tree's engine.

    The revision is installed and its executable built now, at
    `build_server`. It runs the same code as the serving one under another
    revision, which is a change of the epoch's engine and of nothing else,
    and an update to it builds nothing while it prepares: a request started
    after this still streams when the switch comes.
    """
    installed = build_server.install(repository / builds.ENGINE_PACKAGE)
    build_server.resolve(installed.op, installed.backend, installed.revision)
    return {"worker_config": {"engine_revision": installed.revision}}


def listing(uniserve: Path, runtime: Path) -> dict[str, Any]:
    """What the Storage Server under the runtime directory `runtime` holds,
    as ``storage-server list --json`` prints it
    (``storage::protocol::Listing``): its ``tensors``, the bytes its scopes
    hold (``device_bytes``, ``host_bytes``), its ``deployments``, its
    ``children`` and ``transfer`` server, its ``quotas`` and the
    ``buffer_pools`` of the deployments with a session.
    """  # noqa: D205
    listed = subprocess.run(
        [str(uniserve), "storage-server", "list", "--json"],
        env=environment(XDG_RUNTIME_DIR=str(runtime)),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(listed.stdout)


class Stream:
    """A streamed chat completion, read on a thread of its own.

    It records each content delta as it arrives, with the moment it arrived
    (``time.monotonic``), every finish reason its chunks state, and what its
    reader raised, which a refusal, a timeout and a stall otherwise all show
    as a stream that produced nothing. A caller waits on the stream itself
    (:meth:`wait_for`), which wakes at each frame and when the stream ends.
    """

    def __init__(self, base_url: str, request: dict[str, Any]) -> None:
        """Start streaming `request` (its ``stream`` set) at the chat
        completions route of the client API at `base_url`.
        """  # noqa: D205
        self.deltas: list[str] = []
        self.arrivals: list[float] = []
        self.reasons: list[str] = []
        #: The data frames read, whether or not they carry content.
        self.frames = 0
        #: What the reader raised, unless the caller closed the stream.
        self.error: BaseException | None = None
        self.ended = False
        # Every field above changes under this condition, which wakes the
        # waiters at each frame and at the end.
        self._changed = threading.Condition()
        self._closing = False
        self._response: httpx.Response | None = None
        self._reader = threading.Thread(
            target=self._read, args=(base_url, request), daemon=True
        )
        self._reader.start()

    @property
    def text(self) -> str:
        """The content streamed so far."""
        with self._changed:
            return "".join(self.deltas)

    def wait_for(self, predicate: Callable[[Stream], bool], what: str) -> None:
        """Wait until `predicate` holds of the stream, reading it at each
        frame; the stream's own read timeout bounds a stream that stalls.

        Raises:
            AssertionError: Naming `what`, when the stream ended first, with
                what its reader raised.
        """  # noqa: D205
        with self._changed:
            self._changed.wait_for(lambda: predicate(self) or self.ended)
            if predicate(self):
                return
            raise AssertionError(
                f"{what}: the stream ended after {len(self.deltas)} deltas "
                f"(reader error: {self.error!r})"
            )

    def join(self) -> None:
        """Wait until the stream ended."""
        self._reader.join()

    def close(self) -> None:
        """End the stream from the caller's side.

        The reader ends at its next read of the closed response, and what it
        raises then is the caller's doing, not the server's.
        """
        with self._changed:
            self._closing = True
            response = self._response
        if response is not None:
            response.close()

    def _read(self, base_url: str, request: dict[str, Any]) -> None:
        try:
            with httpx.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=request,
                timeout=chat.COMPLETION_S,
            ) as response:
                if response.status_code != 200:
                    raise AssertionError(
                        f"the stream was refused: {response.status_code}: "
                        f"{response.read()[:400]!r}"
                    )
                with self._changed:
                    if self._closing:
                        return
                    self._response = response
                for line in response.iter_lines():
                    if line == "data: [DONE]":
                        break
                    if line.startswith("data: "):
                        self._take(json.loads(line.removeprefix("data: ")))
        except BaseException as error:  # noqa: BLE001 - kept for the waiter
            with self._changed:
                if not self._closing:
                    self.error = error
        finally:
            with self._changed:
                self.ended = True
                self._changed.notify_all()

    def _take(self, chunk: dict[str, Any]) -> None:
        """Record one data frame's deltas and finish reasons."""
        moment = time.monotonic()
        with self._changed:
            self.frames += 1
            for choice in chunk.get("choices", ()):
                content = (choice.get("delta") or {}).get("content")
                if content:
                    self.deltas.append(content)
                    self.arrivals.append(moment)
                if choice.get("finish_reason"):
                    self.reasons.append(choice["finish_reason"])
            self._changed.notify_all()


def numbers(text: str) -> list[int]:
    """The integers `text` writes, in order: a count's numbers."""
    return [int(word) for word in re.findall(r"\d+", text)]


def wait_for_end(pid: int) -> None:
    """Return once process `pid` ended.

    The process is followed through a pidfd, which becomes readable when it
    ends, whoever its parent is; a process that already ended and was reaped
    has no pidfd to open, and is ended. A UniServe process ends within its own
    bounds, so the wait has none of its own: the case's bound (pytest-timeout)
    covers one that hangs.
    """
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        select.select([descriptor], [], [])
    finally:
        os.close(descriptor)


def running(pid: int) -> bool:
    """Whether process `pid` runs now, as a pidfd of it states."""
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        readable, _, _ = select.select([descriptor], [], [], 0)
        return not readable
    finally:
        os.close(descriptor)


def terminate(pid: int) -> None:
    """End process `pid` with SIGTERM and return once it ended.

    The signal goes through a pidfd, so it reaches the process the caller
    named even when its number was reused after it ended. A UniServe process
    ends on SIGTERM within its own bounds (:func:`wait_for_end`).
    """
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        select.select([descriptor], [], [])
    except ProcessLookupError:
        # It ended between the pidfd's opening and the signal.
        pass
    finally:
        os.close(descriptor)


@contextmanager
def successors() -> Iterator[list[int]]:
    """The daemons hot restarts start in place of a case's daemon, each ended
    when the block ends: the caller appends each successor's process id as
    the restart answers it.

    A successor outlives the daemon that started it (``POST
    /admin/restart``), so stopping the launched server does not end it.
    """  # noqa: D205
    started: list[int] = []
    try:
        yield started
    finally:
        for pid in reversed(started):
            terminate(pid)


@contextmanager
def ballast(lease: Lease, leave: int) -> Iterator[int]:
    """Hold the free storage of the leased GPU but ``leave`` bytes in a process
    of its own until the block ends; the bytes it holds.

    The process (:mod:`uniserve_testing.updates.ballast`) sees the leased
    GPU alone, holds what the device lets it allocate of its free storage
    but ``leave`` bytes, and ends when its input closes, which frees them.
    """  # noqa: D205
    process = subprocess.Popen(
        [sys.executable, "-m", BALLAST, str(leave)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=environment(CUDA_VISIBLE_DEVICES=lease.visible),
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        line = process.stdout.readline()
        assert line.strip().isdigit(), f"the ballast process answered {line!r}"
        yield int(line)
    finally:
        process.stdin.close()
        process.wait()
