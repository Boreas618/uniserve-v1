"""Convert profiling sessions' files into a trace Perfetto opens.

The trace is the Trace Event Format's JSON (``traceEvents``), which the
Perfetto UI and ``chrome://tracing`` read. Every worker file becomes a
process named by its rank; its records become tracks of that process:

- each CUDA stream's kernels, copies, fills and whole-graph launches;
- each host thread's runtime and driver API calls and its NVTX ranges;
- the worker's steps, and each device's fence probe intervals (one slice
  per probed component instance's call, nested as the calls nest);
- PM samples as counter tracks.

Every time is placed on CUPTI's clock through the file's ``clocks``
(:meth:`~uniserve_eval.profile.reader.ProfileFile.to_cupti`), so the
workers' files of one session line up; the trace starts at the earliest
record.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .reader import STEP_BEGIN, ProfileFile

# Track ids within a worker's process, above any Linux thread id.
_STREAM = 1 << 32
_STEPS = 2 << 32
_FENCE = 3 << 32

# `CUpti_ActivityFlag` of an NVTX mark.
_INSTANT, _START, _END = 1, 2, 4


def convert(paths: Iterable[str | Path]) -> dict[str, Any]:
    """The trace of the files at ``paths``."""
    events: list[dict[str, Any]] = []
    for path in paths:
        with ProfileFile(path) as profile:
            events.extend(_file_events(profile))
    timed = [event for event in events if "ts" in event]
    origin = min((event["ts"] for event in timed), default=0)
    for event in timed:
        event["ts"] = (event["ts"] - origin) / 1000.0
        if "dur" in event:
            event["dur"] = event["dur"] / 1000.0
    return {"traceEvents": events, "displayTimeUnit": "ns"}


def write(paths: Iterable[str | Path], output: str | Path) -> int:
    """Write the trace of ``paths`` to ``output``; returns its events."""
    trace = convert(paths)
    Path(output).write_text(json.dumps(trace))
    return len(trace["traceEvents"])


def _slice(
    pid: int, tid: int, name: str, start: int, end: int, **args: Any
) -> dict[str, Any]:
    return {
        "name": name,
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": start,
        "dur": max(end - start, 0),
        "args": args,
    }


def _track(pid: int, tid: int, name: str) -> dict[str, Any]:
    return {
        "name": "thread_name",
        "ph": "M",
        "pid": pid,
        "tid": tid,
        "args": {"name": name},
    }


def _file_events(profile: ProfileFile) -> list[dict[str, Any]]:
    pid = profile.rank
    meta = profile.metadata
    events: list[dict[str, Any]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "args": {
                "name": f"rank {pid} (world {meta.get('world')}, "
                f"device {meta.get('device')}, session {meta.get('session')})"
            },
        }
    ]
    streams: set[int] = set()
    threads: set[int] = set()

    def device_slice(table: str, row: dict, name: str, **args: Any) -> None:
        stream = int(row["stream"])
        streams.add(stream)
        events.append(
            _slice(
                pid,
                _STREAM + stream,
                name,
                int(row["start"]),
                int(row["end"]),
                correlation=row["correlation"],
                **args,
            )
        )

    for row in profile.rows("kernel"):
        device_slice(
            "kernel",
            row,
            profile.name(row["name"]),
            grid=[row["grid_x"], row["grid_y"], row["grid_z"]],
            block=[row["block_x"], row["block_y"], row["block_z"]],
            graph=row["graph"],
            graph_node=row["graph_node"],
        )
    for row in profile.rows("copy"):
        device_slice(
            "copy", row, "copy", bytes=row["bytes"], kind=row["copy_kind"]
        )
    for row in profile.rows("fill"):
        device_slice(
            "fill", row, "fill", bytes=row["bytes"], value=row["value"]
        )
    for row in profile.rows("graph"):
        device_slice("graph", row, f"graph {row['graph']}", graph=row["graph"])
    for row in profile.rows("api"):
        thread = int(row["thread"])
        threads.add(thread)
        events.append(
            _slice(
                pid,
                thread,
                profile.name(row["name"]),
                int(row["start"]),
                int(row["end"]),
                correlation=row["correlation"],
                result=row["result"],
            )
        )

    # NVTX ranges pair their start and end marks by id; an instantaneous
    # mark is an instant event.
    starts: dict[int, dict] = {}
    for row in profile.rows("marker", order="time"):
        flags, thread = int(row["flags"]), int(row["object_1"] or 0)
        if flags & _START:
            starts[int(row["id"])] = row
        elif flags & _END and int(row["id"]) in starts:
            start = starts.pop(int(row["id"]))
            thread = int(start["object_1"] or 0)
            threads.add(thread)
            events.append(
                _slice(
                    pid,
                    thread,
                    profile.name(start["name"]),
                    int(start["time"]),
                    int(row["time"]),
                    domain=profile.name(start["domain"]),
                )
            )
        elif flags & _INSTANT:
            threads.add(thread)
            events.append(
                {
                    "name": profile.name(row["name"]),
                    "ph": "i",
                    "s": "t",
                    "pid": pid,
                    "tid": thread,
                    "ts": int(row["time"]),
                }
            )

    if profile.count("steps"):
        events.append(_track(pid, _STEPS, "steps"))
    for row in profile.rows("steps"):
        if row["end"] is None:
            continue
        events.append(
            _slice(
                pid,
                _STEPS,
                f"{row['stage']} {row['step']}",
                int(row["begin"]),
                int(row["end"]),
                selected=bool(row["selected"]),
            )
        )

    devices: set[int] = set()
    for interval in profile.fence_intervals():
        device = interval["device"]
        offset = profile.offset("globaltimer", device)
        if offset is None:
            continue
        devices.add(device)
        events.append(
            _slice(
                pid,
                _FENCE + device,
                interval["component"],
                interval["start"] + offset.nanoseconds,
                interval["end"] + offset.nanoseconds,
                path=interval["path"],
                step=interval["step"],
            )
        )
    # The steps' spans between their markers.
    begins: dict[tuple[int, int], int] = {}
    for row in profile.rows("fence", order="device, time"):
        device, probe = int(row["device"]), int(row["probe"])
        offset = profile.offset("globaltimer", device)
        if probe < STEP_BEGIN or offset is None:
            continue
        key = (device, int(row["step"]))
        if probe == STEP_BEGIN:
            begins[key] = int(row["time"])
        elif key in begins:
            devices.add(device)
            events.append(
                _slice(
                    pid,
                    _FENCE + device,
                    f"step {key[1]}",
                    begins.pop(key) + offset.nanoseconds,
                    int(row["time"]) + offset.nanoseconds,
                )
            )

    for row in profile.rows("pm_sample", order="sample"):
        events.append(
            {
                "name": row["metric"],
                "ph": "C",
                "pid": pid,
                "ts": int(row["start"]),
                "args": {"value": row["value"]},
            }
        )

    events.extend(
        _track(pid, _STREAM + stream, f"stream {stream}")
        for stream in sorted(streams)
    )
    events.extend(
        _track(pid, thread, f"thread {thread}") for thread in sorted(threads)
    )
    events.extend(
        _track(pid, _FENCE + device, f"fence probes (device {device})")
        for device in sorted(devices)
    )
    return events
