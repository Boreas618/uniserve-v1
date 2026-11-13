"""Runs a timeline load: requests over a stated time, with changes.

A point whose load states ``duration_s`` issues requests from the load's
start, t = 0, until t = ``duration_s`` and then waits for every stream to
end. Request ``n`` takes row ``n mod len(rows)``. At an unbounded request
rate the load is a closed loop of ``max_concurrency`` streams, each issuing
its next request when its previous one ends; at a finite rate, arrivals are
Poisson as in ``run_load``, within the concurrency limit.

Meanwhile the point's changes (``types.Change``) act on the deployment, each
at its time, and the load runs on while they do. Issuing continues past
``duration_s`` while a change has not ended, and for ``settle_s`` seconds
after the last one ended. When the deployment states an admin listener, its
event log (``GET /admin/events``) is read throughout the load; a change that
waits for an update's phase finds it there.

``run_timeline`` fills a record the caller owns as the load proceeds, so a
load that fails leaves what it observed. Times ending in ``_s`` are seconds
from the load's start:

- ``load_start``: the ``time.perf_counter()`` reading of t = 0, the clock of
  every request record, and ``clock_offset``, ``time.time()`` minus it;
- ``issue_end_s``: when issuing stopped;
- ``changes``: per change, its declaration, ``issued_s`` and ``ended_s``,
  and what answered it: an HTTP ``status`` and ``answer``, the signalled
  ``pid`` and its ``cmdline``, or a command's ``returncode`` and
  ``output``; ``error`` when it failed;
- ``events``: each admin event as served, with ``received_s`` and its own
  ``time`` as ``at_s``; a failed read of the log leaves its ``error``.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from signal import Signals
from typing import TYPE_CHECKING, Any

import aiohttp
import numpy as np

from ..types import Change, Example, LoadConfig
from .arrival import LoadResult, Submit, T, limit, measured, warm_up

if TYPE_CHECKING:
    from ..server import ManagedDeployment

# The phases in which an update has ended (the daemon's `Phase::ended`): a
# change waiting for another phase of it waits no longer.
_ENDED_PHASES = frozenset({"committed", "rolled_back", "abandoned"})

# Seconds one read of the event log waits for a new event, and between
# reads while the listener does not answer (a daemon restarting or lost).
_EVENT_WAIT_S = 10.0
_EVENT_RETRY_S = 1.0


async def run_timeline(
    rows: list[Example],
    *,
    load: LoadConfig,
    changes: Sequence[Change],
    submit: Submit[T],
    deployment: ManagedDeployment | None,
    record: dict[str, Any],
    warmup_requests: int,
    warmup_rows: list[Example] | None = None,
    inspect_warmup: Callable[[list[T]], None] | None = None,
    before_measure: Callable[[], Awaitable[None]] | None = None,
) -> LoadResult:
    """Run excluded warmup, then the timeline load and its changes.

    Warmup is ``run_load``'s. ``deployment`` is what the changes act on and
    whose admin events are read; without one, a change fails.

    Returns:
        Warmup and measured outputs, the latter in issue order, and the
        measured window.

    Raises:
        ValueError: If ``load`` states no ``duration_s``.
        WarmupFailure: If a warmup request failed; nothing is measured.
        BaseException: An exception from a submission, or the cancellation
            of this call, after every request, change and the event reader
            have been cancelled and awaited.
    """
    duration = load.duration_s
    if duration is None:
        raise ValueError("a timeline load states load.duration_s")
    limited = limit(submit, load.max_concurrency)
    warmup_outputs = await warm_up(
        rows,
        limited,
        warmup_requests=warmup_requests,
        warmup_rows=warmup_rows,
        inspect_warmup=inspect_warmup,
    )
    if before_measure is not None:
        await before_measure()

    start = time.perf_counter()
    offset = time.time() - start
    # A change's record starts as the fields its declaration states.
    entries = {
        change.name: {
            key: value
            for key, value in asdict(change).items()
            if value is not None and value != ()
        }
        for change in changes
    }
    record.update(
        load_start=start,
        clock_offset=offset,
        changes=list(entries.values()),
        events=[],
    )

    def since() -> float:
        return time.perf_counter() - start

    # Issuing runs to `duration_s`, and on while a change is pending and for
    # `settle_s` after the last one ended.
    pending = len(changes)
    last_end = 0.0

    def issuing() -> bool:
        now = since()
        return now < duration or pending > 0 or now < last_end + load.settle_s

    listener = deployment.admin_socket if deployment is not None else None
    arrived = asyncio.Condition()
    finished = {change.name: asyncio.Event() for change in changes}

    async def phase_reached(phase: str, after_s: float) -> float:
        """Wait for the first event from `after_s` reporting `phase`."""
        if listener is None:
            raise RuntimeError("no admin listener reports the update's phases")
        async with arrived:
            while True:
                for event in record["events"]:
                    if event.get("at_s", -math.inf) < after_s:
                        continue
                    reached = (event.get("data") or {}).get("phase")
                    if reached == phase:
                        return float(event["at_s"])
                    if reached in _ENDED_PHASES:
                        raise RuntimeError(
                            f"the update ended {reached} before {phase}"
                        )
                await arrived.wait()

    async def time_of(change: Change) -> float:
        """Return when `change` acts, once its reference is known."""
        if change.at_s is not None:
            return change.at_s
        assert change.after is not None
        await finished[change.after].wait()
        reference = entries[change.after]
        if "error" in reference:
            raise RuntimeError(f"change {change.after} failed")
        if change.phase is None:
            return reference["ended_s"] + change.delay_s
        # The event may never come; the wait is bounded like a request.
        async with asyncio.timeout(load.request_timeout_s):
            reached = await phase_reached(change.phase, reference["issued_s"])
        return reached + change.delay_s

    async def apply(change: Change) -> None:
        """Wait for the change's time, act, and record its answer."""
        nonlocal pending, last_end
        entry = entries[change.name]
        try:
            at = await time_of(change)
            await asyncio.sleep(max(0.0, at - since()))
            entry["issued_s"] = since()
            entry.update(await _act(change, deployment, load.request_timeout_s))
        except Exception as error:  # noqa: BLE001 - recorded as the change's.
            entry["error"] = f"{type(error).__name__}: {error}"
        finally:
            entry["ended_s"] = since()
            pending -= 1
            last_end = max(last_end, entry["ended_s"])
            finished[change.name].set()

    outputs: dict[int, T] = {}
    counter = itertools.count()

    async def stream() -> None:
        """Issue one closed-loop stream's requests, one after another."""
        while issuing():
            index = next(counter)
            row = rows[index % len(rows)]
            outputs[index] = await submit(row, time.perf_counter())

    async def arrive(index: int, scheduled: float) -> None:
        outputs[index] = await limited(rows[index % len(rows)], scheduled)

    background = [asyncio.create_task(apply(change)) for change in changes]
    if listener is not None:
        background.append(
            asyncio.create_task(_read_events(listener, record, start, arrived))
        )
    requests: list[asyncio.Task[None]] = []
    try:
        if math.isinf(load.request_rate):
            assert load.max_concurrency is not None
            requests = [
                asyncio.create_task(stream())
                for _ in range(load.max_concurrency)
            ]
        else:
            # Exponential gaps from the generator `run_point` seeds.
            while issuing():
                index = next(counter)
                requests.append(
                    asyncio.create_task(arrive(index, time.perf_counter()))
                )
                gap = float(np.random.exponential(1.0 / load.request_rate))
                await asyncio.sleep(gap)
        await asyncio.gather(*requests)
    finally:
        # Every change has ended once issuing stopped; the reader runs until
        # the last stream ended, and is stopped here on every path.
        for task in [*requests, *background]:
            task.cancel()
        await asyncio.gather(*requests, *background, return_exceptions=True)
    record["issue_end_s"] = max(duration, last_end + load.settle_s)
    return measured(
        warmup_outputs, [outputs[index] for index in sorted(outputs)], offset
    )


async def _act(
    change: Change, deployment: ManagedDeployment | None, timeout_s: float
) -> dict[str, Any]:
    """Perform a change's action and return what answered it.

    A failed action raises, or, when it was answered, returns its answer
    with an ``error``.
    """
    if deployment is None:
        raise RuntimeError("no deployment to change")
    if change.path is not None:
        listener = deployment.admin_socket
        if listener is None:
            raise RuntimeError("the deployment states no admin listener")
        connector = aiohttp.UnixConnector(path=str(listener))
        assert change.method is not None
        async with aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as session:
            async with session.request(
                change.method,
                f"http://localhost{change.path}",
                json=change.body,
            ) as response:
                text = await response.text(errors="replace")
        answer: dict[str, Any] = {"status": response.status}
        try:
            answer["answer"] = json.loads(text)
        except ValueError:
            answer["answer"] = text
        if not 200 <= response.status < 300:
            answer["error"] = f"HTTP {response.status}"
        return answer

    if change.signal is not None:
        assert change.process is not None
        pattern = re.compile(change.process)
        matches = []
        for pid in deployment.pids():
            try:
                arguments = Path(f"/proc/{pid}/cmdline").read_bytes()
            except OSError:
                continue
            joined = arguments.rstrip(b"\0").replace(b"\0", b" ")
            line = joined.decode(errors="replace")
            if pattern.search(line):
                matches.append((pid, line))
        if len(matches) != 1:
            raise RuntimeError(
                f"{len(matches)} processes of the deployment match "
                f"{change.process!r}: {matches}"
            )
        pid, line = matches[0]
        os.kill(pid, Signals[change.signal])
        return {"pid": pid, "cmdline": line}

    # A command runs as the deployment's processes do: in their working
    # directory, with their environment and runtime directory.
    launch = deployment.servers[0].launch
    process = await asyncio.create_subprocess_exec(
        *change.command,
        cwd=launch.working_directory,
        env={**os.environ, **launch.environment},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        async with asyncio.timeout(timeout_s):
            output, _ = await process.communicate()
    finally:
        # A command that outlives its change, timed out or cancelled, ends
        # with it.
        if process.returncode is None:
            process.kill()
            await process.wait()
    answer = {
        "returncode": process.returncode,
        "output": output.decode(errors="replace"),
    }
    if process.returncode != 0:
        answer["error"] = f"exit status {process.returncode}"
    return answer


async def _read_events(
    listener: Path,
    record: dict[str, Any],
    start: float,
    arrived: asyncio.Condition,
) -> None:
    """Read the admin event log into ``record["events"]`` until cancelled.

    Every event the log keeps is read, from its first; ``at_s`` places an
    event's own time (seconds since the Unix epoch) on the load's clock.
    While the listener does not answer, the first error of each outage is
    recorded and the read is retried every ``_EVENT_RETRY_S``.
    """
    offset = record["clock_offset"]
    after = 0
    failing = False
    connector = aiohttp.UnixConnector(path=str(listener))
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            try:
                async with session.get(
                    "http://localhost/admin/events",
                    params={"after": after, "wait": _EVENT_WAIT_S},
                    timeout=aiohttp.ClientTimeout(total=_EVENT_WAIT_S + 10),
                ) as response:
                    response.raise_for_status()
                    page = await response.json()
            except (aiohttp.ClientError, OSError, ValueError) as error:
                if not failing:
                    record["events"].append(
                        {
                            "error": f"{type(error).__name__}: {error}",
                            "received_s": time.perf_counter() - start,
                        }
                    )
                failing = True
                await asyncio.sleep(_EVENT_RETRY_S)
                continue

            failing = False
            received = time.perf_counter() - start
            for event in page.get("events", []):
                event = {**event, "received_s": received}
                if isinstance(event.get("time"), int | float):
                    event["at_s"] = event["time"] - offset - start
                record["events"].append(event)
            after = max(after, int(page.get("next", after)))
            async with arrived:
                arrived.notify_all()
