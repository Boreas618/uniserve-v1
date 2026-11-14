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
   the first successful output of the same row in the control of the
   configuration the request ended on;
4. new requests admitted throughout: each request submitted within the
   change's window received its first token no later than the window's end
   plus the control's 99th-percentile time to first token;
5. the buffer pool's invariants held: the check-failure counter the command
   names reads 0 in every scrape after the run's window, and the replay
   verdict file, a JSON object, has `verdict` `pass`;
6. with the switch named and a block size (case E4): the first admission
   on the target of each prompt the predecessor served before the switch
   reports as cached (`usage.prompt_tokens_details.cached_tokens`) at
   least the whole blocks of the prompt less its last token, which an
   admission always computes.

The change's window runs from the first change's issue to the latest of
every change's end and every admin event recorded from that issue on. A
request is submitted when the load issued it, its first token arrives with
the first chunk that carried a token id, and it ends with its last event.
A request a cancel change disconnected (`CANCELLED`) is counted and left out
of every item.

The control is the run's own: the same load and timeline without changes,
on the configuration the run started in. A change that moves the deployment
to another configuration names it (`Change.configuration`), and the command
is given that configuration's control, started in it. A request ended on
the configuration that served when it was submitted, unless a change to
another one overlapped its life: an epoch's target admits every request
from its promotion, inside the switch, while the predecessor goes on with
the requests it holds until they end or its drain's deadline migrates them,
so a request in flight across the switch ended on either side, which the
records do not tell apart. It agrees when it agrees with the control of a
configuration it may have ended on.

Reported beside them, with no pass line: the longest gap between two tokens
of any stream, over the whole run and over gaps that meet the window; the
time to first token at p50 and p99 of the requests submitted in the window;
each the same for the control over the same window; and each change's
times.

`uniserve-eval check-faults <run> --reference <alone>` computes the pass of
case E6 instead, whose changes are faults: every request not cancelled
gives, by the near-tie rule, the tokens of its row served alone (the
bundle of a point that served each row once with `max_concurrency = 1`),
and item 5.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..metrics import percentile
from ..server_metrics import parse_samples
from ..types import CANCELLED

# Classifiers of a request whose connection closed under it: its transport
# failed, or its stream ended before its terminal event.
_CLOSED = frozenset({"transport_failure", "response_missing_terminal"})

_ITEMS = (
    ("no_failed_request", "1. No failed request"),
    ("no_closed_connection", "2. No closed connection"),
    ("no_token_lost_or_repeated", "3. No token lost or repeated"),
    ("admitted_throughout", "4. New requests admitted throughout"),
    ("invariants_hold", "5. The buffer pool's invariants hold"),
    ("cached_across_switch", "6. Prompts cached across the switch"),
)

_FAULT_ITEMS = (
    (
        "same_tokens_as_alone",
        "1. Every request not cancelled gives its row's tokens served alone",
    ),
    ("invariants_hold", "2. The buffer pool's invariants hold"),
)


def change_report(
    run: Path,
    control: Path,
    *,
    check_metric: str,
    replay: Path,
    configurations: Mapping[str, Path] | None = None,
    switch: str | None = None,
    block_size: int | None = None,
) -> dict[str, Any]:
    """Compute the pass under change of `run` against `control`.

    Args:
        run: The bundle of a timeline point whose changes were applied.
        control: The bundle of the same load without changes, on the
            configuration the run started in.
        check_metric: The Prometheus name of the Storage Server's
            check-failure counter, summed over its series in each scrape.
        replay: The verdict file of the run's trace replay.
        configurations: Per configuration a change of the run moves the
            deployment to, by name, the bundle of the same load started in
            it: that configuration's control.
        switch: The name of the change that switched the deployment to its
            target: item 6, case E4's, is computed when it is given, with
            `block_size`, the KV cache's block size in tokens.
        block_size: See `switch`.

    Returns:
        The report: `pass`, the window, each item's verdict with its count
        and first failing request, the cancelled requests, the reported
        measures and the changes.

    Raises:
        OSError: A file cannot be read.
        ValueError: A file is not of a timeline bundle, the run issued no
            change, or a change of the run moves the deployment to a
            configuration `configurations` does not name.
        KeyError: A record lacks a field of the evaluator's records.
    """
    if (switch is None) != (block_size is None):
        raise ValueError("item 6 takes both the switch and the block size")
    records, timeline = _bundle(run)
    controls, _ = _bundle(control)
    window = _window(timeline)
    start, end = window
    cancelled = [
        record for record in records if record["classifier"] == CANCELLED
    ]
    records = [
        record for record in records if record["classifier"] != CANCELLED
    ]

    def connection_fault(record: dict[str, Any]) -> str | None:
        if record["classifier"] not in _CLOSED:
            return None
        return f"{record['classifier']}: {record['error']}"

    items: dict[str, dict[str, Any]] = {}
    items["no_failed_request"] = _item(records, _request_fault)
    items["no_closed_connection"] = _item(records, connection_fault)

    # Per configuration, keyed by its name (`None` for the one the run
    # started in), each row's reference in that configuration's control.
    references: dict[str | None, dict[str, dict[str, Any]]] = {
        None: _first_outputs(controls)
    }
    for name, path in (configurations or {}).items():
        references[name] = _first_outputs(_bundle(path)[0])
    moves = _moves(timeline)
    unknown = sorted({name for _, _, name in moves} - references.keys())
    if unknown:
        raise ValueError(
            f"the run's changes move the deployment to {unknown}, of which "
            "no control is given"
        )

    def token_fault(record: dict[str, Any]) -> str | None:
        return _token_fault(
            record,
            [
                (_control_label(name), references[name])
                for name in _ended_on(record, moves)
            ],
        )

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

    items["invariants_hold"] = _invariants(run, check_metric, replay)
    if switch is not None and block_size is not None:
        items["cached_across_switch"] = _cached_across_switch(
            records, timeline, switch, block_size
        )

    return {
        "run": str(run),
        "control": str(control),
        "controls": {
            name: str(path) for name, path in (configurations or {}).items()
        },
        "pass": all(item["pass"] for item in items.values()),
        "window_s": [start, end],
        "cancelled": len(cancelled),
        "items": items,
        "measures": {
            "run": _measures(records, window),
            "control": _measures(controls, window),
        },
        "changes": _changes(timeline),
    }


def fault_report(
    run: Path, reference: Path, *, check_metric: str, replay: Path
) -> dict[str, Any]:
    """Compute case E6's pass of `run`, whose changes are faults.

    Args:
        run: The bundle of a timeline point whose faults were applied.
        reference: The bundle of a point that served each row once with
            nothing else running (`max_concurrency = 1`, no `duration_s`).
        check_metric: As `change_report`'s.
        replay: As `change_report`'s.

    Returns:
        The report: `pass`, each item's verdict, the cancelled requests and
        the changes.

    Raises:
        OSError: A file cannot be read.
        ValueError: `run` is not of a timeline bundle, or `reference` is not
            a completed bundle of rows served alone.
        KeyError: A record lacks a field of the evaluator's records.
    """
    records, timeline = _bundle(run)
    alone = _first_outputs(_alone(reference))
    cancelled = [
        record for record in records if record["classifier"] == CANCELLED
    ]
    records = [
        record for record in records if record["classifier"] != CANCELLED
    ]

    def fault(record: dict[str, Any]) -> str | None:
        return _request_fault(record) or _token_fault(
            record, [("the row served alone", alone)]
        )

    items = {
        "same_tokens_as_alone": _item(records, fault),
        "invariants_hold": _invariants(run, check_metric, replay),
    }
    return {
        "run": str(run),
        "reference": str(reference),
        "pass": all(item["pass"] for item in items.values()),
        "cancelled": len(cancelled),
        "items": items,
        "changes": _changes(timeline),
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
    ]
    for name, path in report["controls"].items():
        lines.append(f"- The control of configuration `{name}`: `{path}`.")
    lines += [
        f"- Requests a change cancelled, left out of the items: "
        f"{report['cancelled']}.",
        "",
        *_item_rows(report["items"], _ITEMS),
    ]

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

    lines += ["", *_change_lines(report["changes"])]
    return "\n".join(lines) + "\n"


def render_fault_report(report: dict[str, Any]) -> str:
    """Render a `fault_report` as the run's `result.md`."""
    lines = [
        f"# Faults under load: {Path(report['run']).name}",
        "",
        f"Run `{report['run']}` against its rows served alone "
        f"`{report['reference']}`. Verdict: **{_verdict(report['pass'])}**.",
        "",
        f"- Requests a change cancelled, left out of the items: "
        f"{report['cancelled']}.",
        "",
        *_item_rows(report["items"], _FAULT_ITEMS),
        "",
        *_change_lines(report["changes"]),
    ]
    return "\n".join(lines) + "\n"


def _item_rows(
    items: dict[str, dict[str, Any]], titles: tuple[tuple[str, str], ...]
) -> list[str]:
    """Return the table of a report's items, in the order of `titles`."""
    lines = ["| Item | Verdict | Detail |", "| --- | --- | --- |"]
    for key, title in titles:
        item = items.get(key)
        if item is None:
            continue
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
        if key == "cached_across_switch":
            detail += f"; {item['served']} prompts served before the switch"
        lines.append(f"| {title} | {_verdict(item['pass'])} | {detail} |")
    return lines


def _change_lines(changes: list[dict[str, Any]]) -> list[str]:
    """Return the list of a report's changes and their times."""
    lines = ["Changes, in seconds from the load's start:", ""]
    for change in changes:
        issued, ended = change["issued_s"], change["ended_s"]
        line = f"- `{change['name']}`: issued {_seconds(issued)}, ended "
        line += _seconds(ended)
        if change["error"]:
            line += f"; failed: {change['error']}"
        lines.append(line)
    return lines


def _changes(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """Return each change of a timeline with its times and any failure."""
    return [
        {
            key: change.get(key)
            for key in ("name", "issued_s", "ended_s", "error")
        }
        for change in timeline["changes"]
    ]


def _bundle(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a timeline bundle's records, timed from its load's start."""
    timeline = json.loads((path / "timeline.json").read_text())
    if "load_start" not in timeline:
        raise ValueError(f"{path} is not a bundle of a timeline load")
    return _records(path, timeline["load_start"]), timeline


def _alone(path: Path) -> list[dict[str, Any]]:
    """Read the records of a bundle of rows served alone.

    Raises:
        ValueError: The bundle is not completed, or its point ran requests
            side by side (`max_concurrency` other than 1) or as a timeline.
    """
    state = json.loads((path / "run.json").read_text())
    load = state.get("workload", {}).get("load", {})
    if (
        state.get("status") != "completed"
        or load.get("max_concurrency") != 1
        or load.get("duration_s") is not None
    ):
        raise ValueError(
            f"{path} is not a completed bundle of rows served alone "
            "(max_concurrency = 1, no duration_s)"
        )
    return _records(path, 0.0)


def _records(path: Path, start: float) -> list[dict[str, Any]]:
    """Read a bundle's records, timed from `start` (a `perf_counter` time).

    Each record gains `index` (its issue order), `row`, `submit_s`,
    `send_s`, `token_s`, `first_s` and `end_s` (its last event, or its last
    token when it has none), in seconds from `start`.
    """
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
        final = record["final_event_time"]
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
        record["end_s"] = (
            final - start
            if final is not None
            else max(tokens, default=record["submit_s"])
        )
        records.append(record)
    return records


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


def _request_fault(record: dict[str, Any]) -> str | None:
    """Return how a request failed: anything but 200, `length` and its count.

    The count is the server's own (its usage) of the tokens it generated,
    which must be the request's limit.
    """
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


def _token_fault(
    record: dict[str, Any],
    references: list[tuple[str, dict[str, dict[str, Any]]]],
) -> str | None:
    """Return how a request's tokens fail every one of `references`.

    The stream must carry exactly the requested tokens, which must agree by
    the near-tie rule with the output of the same row in one of
    `references`, each a label and its outputs by row.
    """
    count, wanted = len(record["token_ids"]), record["requested_output_len"]
    if count != wanted:
        return f"{count} tokens streamed of {wanted}"
    faults = []
    for label, outputs in references:
        reference = outputs.get(record["row"])
        if reference is None:
            faults.append(f"{label} has no output of its row")
            continue
        position = near_tie_mismatch(record, reference)
        if position is None:
            return None
        faults.append(f"disagrees with {label} at token {position}")
    return "; ".join(faults)


def _control_label(configuration: str | None) -> str:
    """Name the control of a configuration, `None` being the run's own."""
    if configuration is None:
        return "the control"
    return f"the control of {configuration}"


def _first_outputs(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return each row's first successful output that carried tokens."""
    outputs: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["success"] and record["token_ids"]:
            outputs.setdefault(record["row"], record)
    return outputs


def _moves(timeline: dict[str, Any]) -> list[tuple[float, float, str]]:
    """Return the applied changes to another configuration, by their end.

    Each is its issue and end, in seconds from the load's start, and the
    configuration it moved the deployment to. A change that failed is the
    point's failure (`changes_applied`) and moves nothing here.
    """
    moves = [
        (change["issued_s"], change["ended_s"], change["configuration"])
        for change in timeline["changes"]
        if change.get("configuration")
        and "issued_s" in change
        and "ended_s" in change
        and "error" not in change
    ]
    return sorted(moves, key=lambda move: move[1])


def _ended_on(
    record: dict[str, Any], moves: list[tuple[float, float, str]]
) -> list[str | None]:
    """Return the configurations a request may have ended on.

    The first is the one that served when it was submitted: the target of
    the last move that ended by then (`None`, the run's own, before any).
    Then each target of a move that overlapped the request's life, from
    the move's issue to its end.
    """
    submitted, ended = record["submit_s"], record["end_s"]
    serving: str | None = None
    for _, end, name in moves:
        if end <= submitted:
            serving = name
    names = [serving]
    for issued, end, name in moves:
        if end > submitted and issued < ended and name not in names:
            names.append(name)
    return names


def _invariants(run: Path, check_metric: str, replay: Path) -> dict[str, Any]:
    """Return item 5: the check-failure count 0 and the replay's `pass`.

    The count is read from every scrape taken after the run's window, and
    one scrape at least must report it.
    """
    scrapes = json.loads((run / "server_metrics.json").read_text())["after"]
    failures = {
        origin: _metric_total(text, check_metric)
        for origin, text in scrapes.items()
    }
    verdict = json.loads(replay.read_text()).get("verdict")
    return {
        "pass": bool(failures)
        and all(count == 0 for count in failures.values())
        and verdict == "pass",
        "check_failures": failures,
        "replay": verdict,
    }


def _cached_across_switch(
    records: list[dict[str, Any]],
    timeline: dict[str, Any],
    name: str,
    block_size: int,
) -> dict[str, Any]:
    """Return item 6, case E4's: what the target found of the predecessor's.

    The switch is the change `name`. The predecessor served a prompt when a
    request of its row received a token before the switch was issued: its
    prompt's whole blocks were committed by then. The target serves from
    the switch's end until the next change is issued; the first request of
    each such row submitted then is the first admission of that prompt on
    the target, and it must report as cached at least the whole blocks of
    its prompt less the last token, `prompt_len - 1` rounded down to
    `block_size`: an admission computes a prompt's last token whatever is
    cached, so a hit never covers it. The item fails when no such
    admission happened: it has no evidence.

    Raises:
        ValueError: The run has no applied change `name`.
    """
    changes = [
        change
        for change in timeline["changes"]
        if "issued_s" in change and "ended_s" in change
    ]
    switch = next(
        (change for change in changes if change["name"] == name), None
    )
    if switch is None:
        raise ValueError(f"the run applied no change {name!r}")
    target_end = min(
        (
            change["issued_s"]
            for change in changes
            if change["issued_s"] > switch["issued_s"]
        ),
        default=math.inf,
    )
    served = {
        record["row"]
        for record in records
        if record["first_s"] is not None
        and record["first_s"] < switch["issued_s"]
    }
    firsts: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda record: record["submit_s"]):
        if (
            record["row"] in served
            and switch["ended_s"] <= record["submit_s"] < target_end
        ):
            firsts.setdefault(record["row"], record)

    def fault(record: dict[str, Any]) -> str | None:
        whole = (record["prompt_len"] - 1) // block_size * block_size
        cached = record["cached_prompt_tokens"]
        if cached is not None and cached >= whole:
            return None
        return f"{cached} prompt tokens cached, below the {whole} of its blocks"

    admissions = sorted(firsts.values(), key=lambda record: record["index"])
    item = _item(admissions, fault)
    item["served"] = len(served)
    if not admissions:
        item["pass"] = False
        item["first"] = (
            "no prompt the predecessor served was admitted on the target"
        )
    return item


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
