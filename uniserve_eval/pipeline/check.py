"""Checks a timeline run of changes against its control run.

`uniserve-eval check-change <run> --control <control>` reads two completed
bundles of timeline points (`load.duration_s`): the run, whose changes acted
on the deployment, and its control, the same load and timeline without
them. It computes the pass under change of the heavy cases' protocol
(`artifacts/buffer-pool/heavy/protocol.md`):

1. no failed request: each answers 200 and ends with `finish_reason =
   "length"` after the server counted exactly its requested tokens;
2. no closed connection: no request ended in a transport failure or in a
   stream that stopped without its terminal event;
3. no token lost or repeated: each stream carried exactly its requested
   tokens, and they agree by the near-tie rule (`near_tie_mismatch`) with
   the control's first successful output of the same row;
4. new requests admitted throughout: each request submitted within the
   change's window received its first token no later than the window's end
   plus the control's 99th-percentile time to first token;
5. the buffer pool's invariants held: the check-failure counter the command
   names reads 0 in every scrape after the run's window, and the replay
   verdict file, a JSON object, has `verdict` `pass`.

The change's window runs from the first change's issue to the latest of
every change's end and every admin event recorded from that issue on. A
request is submitted when the load issued it, and its first token arrives
with the first chunk that carried a token id.

Reported beside them, with no pass line: the longest gap between two tokens
of any stream, over the whole run and over gaps that meet the window; the
time to first token at p50 and p99 of the requests submitted in the window;
each the same for the control over the same window; and each change's
times.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..metrics import percentile
from ..server_metrics import parse_samples

# Classifiers of a request whose connection closed under it: its transport
# failed, or its stream ended before its terminal event.
_CLOSED = frozenset({"transport_failure", "response_missing_terminal"})

_ITEMS = (
    ("no_failed_request", "1. No failed request"),
    ("no_closed_connection", "2. No closed connection"),
    ("no_token_lost_or_repeated", "3. No token lost or repeated"),
    ("admitted_throughout", "4. New requests admitted throughout"),
    ("invariants_hold", "5. The buffer pool's invariants hold"),
)


def change_report(
    run: Path, control: Path, *, check_metric: str, replay: Path
) -> dict[str, Any]:
    """Compute the pass under change of `run` against `control`.

    Args:
        run: The bundle of a timeline point whose changes were applied.
        control: The bundle of the same load without changes.
        check_metric: The Prometheus name of the Storage Server's
            check-failure counter, summed over its series in each scrape.
        replay: The verdict file of the run's trace replay.

    Returns:
        The report: `pass`, the window, each item's verdict with its count
        and first failing request, the reported measures and the changes.

    Raises:
        OSError: A file cannot be read.
        ValueError: A file is not of a timeline bundle, or the run issued
            no change.
        KeyError: A record lacks a field of the evaluator's records.
    """
    records, timeline = _bundle(run)
    controls, _ = _bundle(control)
    window = _window(timeline)
    start, end = window

    def request_fault(record: dict[str, Any]) -> str | None:
        # The server's own count of the tokens it generated.
        counted = (
            record["output_len"]
            if record["output_len_source"] == "server_usage"
            else None
        )
        if (
            record["success"]
            and record["status_code"] == 200
            and record["finish_reason"] == "length"
            and counted == record["requested_output_len"] > 0
        ):
            return None
        return (
            f"{record['classifier']}, status {record['status_code']}, "
            f"finish {record['finish_reason']}, {counted} tokens counted"
        )

    def connection_fault(record: dict[str, Any]) -> str | None:
        if record["classifier"] not in _CLOSED:
            return None
        return f"{record['classifier']}: {record['error']}"

    items: dict[str, dict[str, Any]] = {}
    items["no_failed_request"] = _item(records, request_fault)
    items["no_closed_connection"] = _item(records, connection_fault)

    # The reference of a row is its first successful control output.
    references: dict[str, dict[str, Any]] = {}
    for record in controls:
        if record["success"] and record["token_ids"]:
            references.setdefault(record["row"], record)

    def token_fault(record: dict[str, Any]) -> str | None:
        count, wanted = len(record["token_ids"]), record["requested_output_len"]
        if count != wanted:
            return f"{count} tokens streamed of {wanted}"
        reference = references.get(record["row"])
        if reference is None:
            return "the control has no output of its row"
        position = near_tie_mismatch(record, reference)
        if position is None:
            return None
        return f"disagrees with the control at token {position}"

    items["no_token_lost_or_repeated"] = _item(records, token_fault)

    ttfts = [
        record["first_s"] - record["send_s"]
        for record in controls
        if record["success"] and record["first_s"] is not None
    ]
    deadline = end + percentile(ttfts, 99)

    def admission_fault(record: dict[str, Any]) -> str | None:
        first = record["first_s"]
        if first is not None and first <= deadline:
            return None
        return f"first token at {_seconds(first)}"

    items["admitted_throughout"] = _item(
        [record for record in records if start <= record["submit_s"] <= end],
        admission_fault,
    )
    items["admitted_throughout"]["deadline_s"] = deadline

    scrapes = json.loads((run / "server_metrics.json").read_text())["after"]
    failures = {
        origin: _metric_total(text, check_metric)
        for origin, text in scrapes.items()
    }
    verdict = json.loads(replay.read_text()).get("verdict")
    items["invariants_hold"] = {
        "pass": bool(failures)
        and all(count == 0 for count in failures.values())
        and verdict == "pass",
        "check_failures": failures,
        "replay": verdict,
    }

    return {
        "run": str(run),
        "control": str(control),
        "pass": all(item["pass"] for item in items.values()),
        "window_s": [start, end],
        "items": items,
        "measures": {
            "run": _measures(records, window),
            "control": _measures(controls, window),
        },
        "changes": [
            {
                key: change.get(key)
                for key in ("name", "issued_s", "ended_s", "error")
            }
            for change in timeline["changes"]
        ],
    }


def near_tie_mismatch(
    first: dict[str, Any], second: dict[str, Any]
) -> int | None:
    """Return where two greedy outputs of one prompt disagree, if they do.

    The near-tie rule is vLLM's `check_logprobs_close` with
    `near_tie_tol = 0`: two outputs agree when their token ids are equal up
    to the first position where they differ and, there, each side's token
    is among the other side's candidates (`top_logprobs`). Later positions,
    and positions past the shorter output, are not compared.

    Returns:
        The position of a divergence outside the candidates, else `None`.
    """
    pairs = zip(first["token_ids"], second["token_ids"], strict=False)
    for position, (one, other) in enumerate(pairs):
        if one == other:
            continue
        near = one in _candidates(second, position)
        near = near and other in _candidates(first, position)
        return None if near else position
    return None


def render_change_report(report: dict[str, Any]) -> str:
    """Render a `change_report` as the run's `result.md`."""
    start, end = report["window_s"]
    lines = [
        f"# Pass under change: {Path(report['run']).name}",
        "",
        f"Run `{report['run']}` against the control `{report['control']}`. "
        f"The change's window runs from {start:.3f} s to {end:.3f} s after "
        f"the load's start. Verdict: **{_verdict(report['pass'])}**.",
        "",
        "| Item | Verdict | Detail |",
        "| --- | --- | --- |",
    ]
    for key, title in _ITEMS:
        item = report["items"][key]
        if key == "invariants_hold":
            detail = (
                f"check failures {item['check_failures']}; replay "
                f"{item['replay']}"
            )
        else:
            detail = f"{item['count']} of {item['of']} requests fail"
            if item["first"]:
                detail += f"; first: {item['first']}"
        if key == "admitted_throughout":
            detail += f"; deadline {item['deadline_s']:.3f} s"
        lines.append(f"| {title} | {_verdict(item['pass'])} | {detail} |")

    run, control = report["measures"]["run"], report["measures"]["control"]
    lines += [
        "",
        "| Measure (no pass line) | Run | Control |",
        "| --- | --- | --- |",
    ]
    for key, title in (
        ("max_gap_ms", "Longest gap between two tokens of any stream (ms)"),
        ("max_gap_window_ms", "Longest gap meeting the window (ms)"),
        ("ttft_p50_ms", "TTFT p50, requests submitted in the window (ms)"),
        ("ttft_p99_ms", "TTFT p99, requests submitted in the window (ms)"),
    ):
        lines.append(f"| {title} | {run[key]:.3f} | {control[key]:.3f} |")
    for key, title in (
        ("window_requests", "Requests submitted in the window"),
        ("requests", "Requests"),
    ):
        lines.append(f"| {title} | {run[key]} | {control[key]} |")

    lines += ["", "Changes, in seconds from the load's start:", ""]
    for change in report["changes"]:
        issued, ended = change["issued_s"], change["ended_s"]
        line = f"- `{change['name']}`: issued {_seconds(issued)}, ended "
        line += _seconds(ended)
        if change["error"]:
            line += f"; failed: {change['error']}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _bundle(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a timeline bundle's records, timed from its load's start.

    Each record gains `index` (its issue order), `row`, `submit_s`,
    `send_s`, `token_s` and `first_s`, in seconds from the load's start.
    """
    timeline = json.loads((path / "timeline.json").read_text())
    if "load_start" not in timeline:
        raise ValueError(f"{path} is not a bundle of a timeline load")
    start = timeline["load_start"]
    records = []
    lines = (path / "requests.jsonl").read_text().splitlines()
    for index, line in enumerate(item for item in lines if item.strip()):
        record = json.loads(line)
        sent = record["client_send_time"]
        submitted = record["scheduled_time"]
        tokens = [
            time - start for time in record["token_times"] if time is not None
        ]
        first = record["first_text_time"]
        record.update(
            index=index,
            row=record["example"]["id"],
            submit_s=(sent if submitted is None else submitted) - start,
            send_s=sent - start,
            token_s=tokens,
            first_s=tokens[0]
            if tokens
            else (None if first is None else first - start),
        )
        records.append(record)
    return records, timeline


def _window(timeline: dict[str, Any]) -> tuple[float, float]:
    """Return the change's window, in seconds from the load's start."""
    changes = timeline["changes"]
    issued = [change["issued_s"] for change in changes if "issued_s" in change]
    if not issued:
        raise ValueError("the run issued no change")
    start = min(issued)
    ends = [change["ended_s"] for change in changes if "ended_s" in change]
    ends += [
        event["at_s"]
        for event in timeline["events"]
        if event.get("at_s", start - 1) >= start
    ]
    return start, max(ends)


def _item(
    records: list[dict[str, Any]],
    fault: Callable[[dict[str, Any]], str | None],
) -> dict[str, Any]:
    """Apply `fault` (a record's failure, or `None`) to every record."""
    reasons = [(record, fault(record)) for record in records]
    failing = [(record, reason) for record, reason in reasons if reason]
    first = None
    if failing:
        record, reason = failing[0]
        first = (
            f"request {record['index']} (row {record['row']}, submitted at "
            f"{record['submit_s']:.3f} s): {reason}"
        )
    return {
        "pass": not failing,
        "count": len(failing),
        "of": len(records),
        "first": first,
    }


def _measures(
    records: list[dict[str, Any]], window: tuple[float, float]
) -> dict[str, float]:
    """Return the reported measures of one run over `window`."""
    start, end = window
    gaps, window_gaps = [0.0], [0.0]
    for record in records:
        times = record["token_s"]
        for before, after in zip(times, times[1:], strict=False):
            gaps.append(after - before)
            if after >= start and before <= end:
                window_gaps.append(after - before)
    submitted = [
        record for record in records if start <= record["submit_s"] <= end
    ]
    ttfts = [
        record["first_s"] - record["send_s"]
        for record in submitted
        if record["first_s"] is not None
    ]
    return {
        "max_gap_ms": max(gaps) * 1000,
        "max_gap_window_ms": max(window_gaps) * 1000,
        "ttft_p50_ms": percentile(ttfts, 50) * 1000,
        "ttft_p99_ms": percentile(ttfts, 99) * 1000,
        "window_requests": len(submitted),
        "requests": len(records),
    }


def _candidates(record: dict[str, Any], position: int) -> set[Any]:
    """Return the token ids of a record's candidates at a position."""
    if position >= len(record["top_logprobs"]):
        return set()
    return {pair[0] for pair in record["top_logprobs"][position]}


def _metric_total(text: str | None, name: str) -> float | None:
    """Return the sum of a counter's series in a scrape, if it has one."""
    if text is None:
        return None
    values = [
        value
        for key, value in parse_samples(text).items()
        if key.split("{", 1)[0] == name
    ]
    return sum(values) if values else None


def _seconds(value: float | None) -> str:
    return "never" if value is None else f"{value:.3f} s"


def _verdict(passed: bool) -> str:
    return "pass" if passed else "FAIL"
