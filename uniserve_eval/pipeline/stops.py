"""Checks a timeline run in which a process was stopped, against its control.

`uniserve-eval check-stop <run> --control <control> --headroom-tokens N`
reads two completed timeline bundles (`load.duration_s`): the run, in which
a process of the deployment was stopped with `SIGSTOP` and continued with
`SIGCONT` (the heavy case E2 stops the host's Storage Server,
`artifacts/buffer-pool/heavy/protocol.md`), and its control, the same load
without the signals. A stop window runs from a `SIGSTOP` change's signal to
the signal of the next `SIGCONT` change to the same process, in seconds from
the load's start; the control's window is the same interval of its own load.
A token arrives with its text-bearing event (`text_times`), and a gap is the
interval between two consecutive tokens of one stream, the evaluator's
inter-token latency. The items:

1. No failed request: `check._request_fault` holds for every request.
2. The decode step time is unchanged while stopped, per window: the gaps
   whose later token arrived in the window, the decode steps that completed
   while the process was stopped as the client saw them, have their p50 and
   p99 within the minimum to the maximum of the control's gaps of the same
   window; and the window's longest silence, the longest part of it in
   which no stream received a token, is at most that maximum. A step that
   ran in the window lasted at least as long as the silence it spans, so a
   step held up by the stopped process shows there even when none of its
   gaps ends in the window.
3. Left out only after the headroom, and scheduled again: between two
   consecutive scrapes (`scrapes.jsonl`), the counter of calls left out for
   late blocks (`--counter`, its series summed per origin) rises only by
   calls a stop window explains, and for a window only when some stream
   received at least `--headroom-tokens` tokens in it: a request holds that
   many tokens of blocks beyond its next call when it asks for more, so its
   blocks run out only after it decoded them while the process was
   stopped. A call left out at time c is counted in the scheduler's report
   at most one statistics period (`REPORT_PERIOD_S`) later and read by the
   first scrape after that report; window [a, b] explains a rise between
   scrapes k - 1 and k when scrape k was answered at or after a and scrape
   k - 1 was sent at most `b + 2 * REPORT_PERIOD_S`: a request whose blocks
   ran out while the process was stopped receives them within one period
   of the signal that continued it, and its last left-out call is reported
   within one more. And every request in flight at a window's end (a token
   before it and its stream not ended by it) completed: item 1 holds for
   it.

Reported with no pass line: per window, its duration, both runs' gaps
(count, minimum, p50, p99, maximum), the control's longest silence, the most
tokens one stream received in it, the requests in flight at its end, the
longest time from its end to the next token of one of them, and the
counter's rise it explains; the counter's rise over the run; the longest gap
of any stream, and the time to first token at p50 and p99 of the requests
submitted in each window, beside the control's; and, when given, the
Storage Server's check failures from its listing and the trace replay's
verdict line.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..metrics import percentile
from .check import _bundle, _item, _metric_total, _request_fault, _verdict

#: The scheduler's statistics period: each report counts the calls left out
#: since the previous one (`crates/engine/src/scheduler/stats.rs`,
#: `STATS_PERIOD`; `specs/scheduler-design.md` §7.5, one snapshot a second).
REPORT_PERIOD_S = 1.0

#: The daemon's counter of calls left out of a step for late blocks.
COUNTER = "uniserve:scheduler_calls_without_blocks_total"

_ITEMS = (
    ("no_failed_request", "1. No failed request"),
    ("step_time_unchanged", "2. Decode step time within the control's"),
    ("left_out_after_headroom", "3. Left out only after the headroom"),
)


def stop_report(
    run: Path,
    control: Path,
    *,
    headroom_tokens: int,
    counter: str = COUNTER,
    listing: Path | None = None,
    replay: Path | None = None,
) -> dict[str, Any]:
    """Compute E2's pass of `run` against `control`.

    Args:
        run: The bundle of a timeline point that stopped and continued a
            process of the deployment.
        control: The bundle of the same load without the signals.
        headroom_tokens: The tokens of blocks a decoding request holds
            beyond its next call when it asks for more blocks.
        counter: The Prometheus name of the left-out counter.
        listing: The Storage Server's listing (`storage-server list`)
            taken after the load, whose check failures are reported.
        replay: The output of the run's trace replay, whose first line is
            reported.

    Returns:
        The report: `pass`, the windows, each item's verdict with its
        detail, and the reported measures.

    Raises:
        OSError: A file cannot be read.
        ValueError: A file is not of a timeline bundle, a change failed, or
            the run has no stop window.
        KeyError: A record lacks a field of the evaluator's records.
    """
    records, timeline = _bundle(run)
    controls, control_timeline = _bundle(control)
    _add_text_times(records, timeline)
    _add_text_times(controls, control_timeline)
    windows = _windows(timeline)

    items: dict[str, dict[str, Any]] = {}
    items["no_failed_request"] = _item(records, _request_fault)

    # Item 2, window by window.
    measured = []
    for start, end in windows:
        gaps = _gaps(records, start, end)
        control_gaps = _gaps(controls, start, end)
        low = min(control_gaps, default=None)
        high = max(control_gaps, default=None)
        p50 = percentile(gaps, 50) if gaps else None
        p99 = percentile(gaps, 99) if gaps else None
        silence = _silence(records, start, end)
        within = (
            low is not None
            and high is not None
            and p50 is not None
            and p99 is not None
            and low <= p50 <= high
            and low <= p99 <= high
            and silence <= high
        )
        measured.append(
            {
                "window_s": [start, end],
                "duration_ms": (end - start) * 1000,
                "pass": within,
                "run": _spread(gaps),
                "control": _spread(control_gaps),
                "silence_ms": silence * 1000,
                "control_silence_ms": _silence(controls, start, end) * 1000,
                "most_tokens": _most_tokens(records, start, end),
                **_in_flight(records, end),
            }
        )
    items["step_time_unchanged"] = {
        "pass": all(window["pass"] for window in measured),
        "windows": measured,
    }

    items["left_out_after_headroom"] = _left_out(
        run, timeline, records, measured, counter, headroom_tokens
    )

    report: dict[str, Any] = {
        "run": str(run),
        "control": str(control),
        "pass": all(item["pass"] for item in items.values()),
        "windows_s": [list(window) for window in windows],
        "headroom_tokens": headroom_tokens,
        "counter": counter,
        "items": items,
        "measures": {
            "run": _measures(records, windows),
            "control": _measures(controls, windows),
        },
    }
    if listing is not None:
        text = listing.read_text(errors="replace")
        report["check_failures"] = [
            int(found) for found in re.findall(r"(\d+) check failures", text)
        ]
    if replay is not None:
        lines = replay.read_text(errors="replace").splitlines()
        report["replay"] = lines[0] if lines else ""
    return report


def render_stop_report(report: dict[str, Any]) -> str:
    """Render a `stop_report` as the run's `result.md`."""
    lines = [
        f"# E2, the process stopped: {Path(report['run']).name}",
        "",
        f"Run `{report['run']}` against the control `{report['control']}`. "
        f"Verdict: **{_verdict(report['pass'])}**.",
        "",
        "| Item | Verdict | Detail |",
        "| --- | --- | --- |",
    ]
    for key, title in _ITEMS:
        item = report["items"][key]
        if key == "no_failed_request":
            detail = f"{item['count']} of {item['of']} requests fail"
            if item["first"]:
                detail += f"; first: {item['first']}"
        elif key == "step_time_unchanged":
            detail = "; ".join(
                f"window at {window['window_s'][0]:.3f} s: "
                f"p50 {_ms(window['run']['p50_ms'])}, "
                f"p99 {_ms(window['run']['p99_ms'])}, "
                f"silence {window['silence_ms']:.3f} ms, against "
                f"{_ms(window['control']['min_ms'])} to "
                f"{_ms(window['control']['max_ms'])}"
                for window in item["windows"]
            )
        else:
            detail = (
                f"{len(item['unexplained'])} rises no window explains, "
                f"{len(item['before_headroom'])} for a window shorter than "
                f"the headroom; {item['in_flight_failed']} of "
                f"{item['in_flight']} requests in flight at a window's end "
                "failed"
            )
            if item["first"]:
                detail += f"; first: {item['first']}"
        lines.append(f"| {title} | {_verdict(item['pass'])} | {detail} |")

    lines += [
        "",
        "Per window (times in seconds from the load's start, gaps in ms):",
        "",
        "| Window | Duration (ms) | Run gaps: n, min, p50, p99, max | "
        "Control gaps: n, min, p50, p99, max | Silence run / control (ms) | "
        "Most tokens of a stream | In flight at end | Longest wait after "
        "end (ms) | Counter rise explained |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rises = report["items"]["left_out_after_headroom"]["rises_by_window"]
    for window, rise in zip(
        report["items"]["step_time_unchanged"]["windows"], rises, strict=True
    ):
        start, end = window["window_s"]
        lines.append(
            f"| {start:.3f} to {end:.3f} | {window['duration_ms']:.1f} | "
            f"{_spread_text(window['run'])} | "
            f"{_spread_text(window['control'])} | "
            f"{window['silence_ms']:.3f} / "
            f"{window['control_silence_ms']:.3f} | {window['most_tokens']} | "
            f"{window['in_flight']} | {_ms(window['longest_wait_ms'])} | "
            f"{rise:g} |"
        )

    item = report["items"]["left_out_after_headroom"]
    run, control = report["measures"]["run"], report["measures"]["control"]
    lines += [
        "",
        f"Counter `{report['counter']}`: rose by {item['total_rise']:g} over "
        f"the run, in {item['scrapes']} scrapes; headroom "
        f"{report['headroom_tokens']} tokens.",
        "",
        "| Measure (no pass line) | Run | Control |",
        "| --- | --- | --- |",
        f"| Longest gap between two tokens of any stream (ms) | "
        f"{run['max_gap_ms']:.3f} | {control['max_gap_ms']:.3f} |",
    ]
    for index, (mine, theirs) in enumerate(
        zip(run["windows"], control["windows"], strict=True)
    ):
        for key, title in (
            ("ttft_p50_ms", "TTFT p50"),
            ("ttft_p99_ms", "TTFT p99"),
        ):
            lines.append(
                f"| {title}, requests submitted in window {index + 1} (ms) | "
                f"{_ms(mine[key])} | {_ms(theirs[key])} |"
            )
        lines.append(
            f"| Requests submitted in window {index + 1} | "
            f"{mine['submitted']} | {theirs['submitted']} |"
        )
    lines.append(f"| Requests | {run['requests']} | {control['requests']} |")
    if "check_failures" in report:
        lines += [
            "",
            "Storage Server check failures (its listing after the load): "
            f"{report['check_failures']}.",
        ]
    if "replay" in report:
        lines += ["", f"Trace replay: `{report['replay']}`."]
    return "\n".join(lines) + "\n"


def _add_text_times(
    records: list[dict[str, Any]], timeline: dict[str, Any]
) -> None:
    """Give each record its tokens' arrivals and its stream's end.

    `text_s` and `end_s` are seconds from the load's start; a stream with no
    final event ends with its last token.
    """
    start = timeline["load_start"]
    for record in records:
        record["text_s"] = [time - start for time in record["text_times"]]
        final = record["final_event_time"]
        record["end_s"] = (
            final - start
            if final is not None
            else (record["text_s"][-1] if record["text_s"] else None)
        )


def _windows(timeline: dict[str, Any]) -> list[tuple[float, float]]:
    """Return the run's stop windows, in seconds from its load's start.

    Raises:
        ValueError: If a change failed or was not issued, a `SIGSTOP` has
            no later `SIGCONT` to its process, or there is no `SIGSTOP`.
    """
    changes = timeline["changes"]
    for change in changes:
        if "error" in change or "issued_s" not in change:
            raise ValueError(
                f"change {change.get('name')} did not apply: "
                f"{change.get('error', 'not issued')}"
            )
    signals = sorted(
        (change for change in changes if "signal" in change),
        key=lambda change: change["issued_s"],
    )
    windows = []
    for index, stop in enumerate(signals):
        if stop["signal"] != "SIGSTOP":
            continue
        continued = [
            change
            for change in signals[index + 1 :]
            if change["signal"] == "SIGCONT" and change["pid"] == stop["pid"]
        ]
        if not continued:
            raise ValueError(f"{stop['name']} is not continued by a SIGCONT")
        windows.append((stop["issued_s"], continued[0]["issued_s"]))
    if not windows:
        raise ValueError("the run stopped no process")
    return windows


def _gaps(
    records: list[dict[str, Any]], start: float, end: float
) -> list[float]:
    """Return the gaps, in seconds, whose later token arrived in the window."""
    gaps = []
    for record in records:
        times = record["text_s"]
        for before, after in zip(times, times[1:], strict=False):
            if start <= after <= end:
                gaps.append(after - before)
    return gaps


def _silence(records: list[dict[str, Any]], start: float, end: float) -> float:
    """Return the longest part of the window in which no token arrived."""
    arrivals = sorted(
        time
        for record in records
        for time in record["text_s"]
        if start <= time <= end
    )
    edges = [start, *arrivals, end]
    return max(after - before for before, after in zip(edges, edges[1:]))


def _most_tokens(
    records: list[dict[str, Any]], start: float, end: float
) -> int:
    """Return the most tokens one stream received in the window."""
    return max(
        (
            sum(start <= time <= end for time in record["text_s"])
            for record in records
        ),
        default=0,
    )


def _in_flight(records: list[dict[str, Any]], end: float) -> dict[str, Any]:
    """Describe the requests in flight at a window's end.

    A request is in flight when a token of it arrived before `end` and its
    stream had not ended by then; `longest_wait_ms` is the longest time from
    `end` to the next token of one of them.
    """
    waiting = [record for record in records if _is_in_flight(record, end)]
    waits = [
        min(time for time in record["text_s"] if time > end) - end
        for record in waiting
        if any(time > end for time in record["text_s"])
    ]
    return {
        "in_flight": len(waiting),
        "in_flight_failed": sum(
            _request_fault(record) is not None for record in waiting
        ),
        "longest_wait_ms": max(waits) * 1000 if waits else None,
    }


def _is_in_flight(record: dict[str, Any], end: float) -> bool:
    """Whether a token of a request arrived before `end` and it ran on."""
    times = record["text_s"]
    return bool(times) and times[0] < end and record["end_s"] > end


def _left_out(
    run: Path,
    timeline: dict[str, Any],
    records: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    counter: str,
    headroom_tokens: int,
) -> dict[str, Any]:
    """Compute item 3 from the run's scrapes and records."""
    start = timeline["load_start"]
    lines = (run / "scrapes.jsonl").read_text().splitlines()
    scrapes = [json.loads(line) for line in lines if line.strip()]

    # Each origin's readings of the counter, in tick order; a scrape that
    # failed or lacks the counter is no reading, and the next rise spans it.
    readings: dict[str, list[tuple[float, float, float]]] = {}
    for scrape in sorted(scrapes, key=lambda scrape: scrape["tick"]):
        value = _metric_total(scrape["text"], counter)
        if value is not None:
            readings.setdefault(scrape["origin"], []).append(
                (scrape["sent"] - start, scrape["received"] - start, value)
            )

    unexplained: list[str] = []
    before_headroom: list[str] = []
    rises_by_window = [0.0] * len(windows)
    total = 0.0
    for origin, series in readings.items():
        total += series[-1][2] - series[0][2]
        for previous, current in zip(series, series[1:], strict=False):
            rise = current[2] - previous[2]
            if rise <= 0:
                continue
            where = (
                f"{origin}: +{rise:g} between the scrapes sent at "
                f"{previous[0]:.3f} s and answered at {current[1]:.3f} s"
            )
            explaining = [
                index
                for index, window in enumerate(windows)
                if current[1] >= window["window_s"][0]
                and previous[0] <= window["window_s"][1] + 2 * REPORT_PERIOD_S
            ]
            if not explaining:
                unexplained.append(where)
                continue
            index = explaining[0]
            rises_by_window[index] += rise
            if windows[index]["most_tokens"] < headroom_tokens:
                before_headroom.append(where)

    in_flight = sum(window["in_flight"] for window in windows)
    in_flight_failed = sum(window["in_flight_failed"] for window in windows)
    faults = unexplained + before_headroom
    if not readings:
        faults.append(f"no scrape reports {counter}")
    return {
        "pass": not faults and in_flight_failed == 0,
        "scrapes": len(scrapes),
        "total_rise": total,
        "rises_by_window": rises_by_window,
        "unexplained": unexplained,
        "before_headroom": before_headroom,
        "in_flight": in_flight,
        "in_flight_failed": in_flight_failed,
        "first": faults[0] if faults else None,
    }


def _measures(
    records: list[dict[str, Any]], windows: list[tuple[float, float]]
) -> dict[str, Any]:
    """Return one run's reported measures over the stop windows."""
    gaps = [
        after - before
        for record in records
        for before, after in zip(
            record["text_s"], record["text_s"][1:], strict=False
        )
    ]
    per_window = []
    for start, end in windows:
        ttfts = [
            record["first_s"] - record["send_s"]
            for record in records
            if start <= record["submit_s"] <= end
            and record["first_s"] is not None
        ]
        per_window.append(
            {
                "submitted": sum(
                    start <= record["submit_s"] <= end for record in records
                ),
                "ttft_p50_ms": percentile(ttfts, 50) * 1000 if ttfts else None,
                "ttft_p99_ms": percentile(ttfts, 99) * 1000 if ttfts else None,
            }
        )
    return {
        "max_gap_ms": max(gaps, default=0.0) * 1000,
        "requests": len(records),
        "windows": per_window,
    }


def _spread(gaps: list[float]) -> dict[str, Any]:
    """Return a set of gaps' count, extremes and percentiles, in ms."""
    if not gaps:
        return {
            "count": 0,
            "min_ms": None,
            "p50_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(gaps),
        "min_ms": min(gaps) * 1000,
        "p50_ms": percentile(gaps, 50) * 1000,
        "p99_ms": percentile(gaps, 99) * 1000,
        "max_ms": max(gaps) * 1000,
    }


def _spread_text(spread: dict[str, Any]) -> str:
    return ", ".join(
        [str(spread["count"])]
        + [_ms(spread[key]) for key in ("min_ms", "p50_ms", "p99_ms", "max_ms")]
    )


def _ms(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}"
