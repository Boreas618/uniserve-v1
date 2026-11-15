"""Per-second scrapes of a timeline load, and `check-stop`.

A timeline point runs against a stand-in server, a real process that
`uniserve-eval run` launches, which streams tokens and serves a `/metrics`
exposition counting its own scrapes; the point's bundle must hold one scrape
of it per second over the load.

`check-stop` is held to E2's pass on bundles written here as the evaluator
writes them: four streams decode one token per 10 ms step, with a fixed
pattern of client-side jitter, and the run stops a process from 3.0 s to
3.3 s. In the clean run the steps go on while it is stopped, two streams are
left out after the headroom of 8 tokens and resume after the stop, and the
left-out counter rises in the scrape after it; each other run changes one
thing the protocol forbids.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from prometheus_client.parser import text_string_to_metric_families

pytestmark = pytest.mark.unit

_COUNTER = "uniserve:scheduler_calls_without_blocks_total"


def _sample(exposition: str, name: str) -> float:
    """The value of the unlabelled sample ``name`` in an exposition."""
    for family in text_string_to_metric_families(exposition):
        for sample in family.samples:
            if sample.name == name and not sample.labels:
                return sample.value
    raise AssertionError(f"the exposition holds no {name}: {exposition!r}")


def test_a_timeline_load_scrapes_its_server_every_second(
    tmp_path: Path,
    unused_tcp_port: int,
    evaluate: Callable[[Sequence[str]], int],
) -> None:
    """The bundle holds one scrape per second from the load's start."""
    port = unused_tcp_port
    (tmp_path / "rows.jsonl").write_text(
        "".join(
            json.dumps({"id": f"row-{i}", "prompt": f"p {i}", "output_len": 10})
            + "\n"
            for i in range(4)
        ),
        encoding="utf-8",
    )
    # The profile states the port the stand-in listens on, since the
    # evaluator probes the address its profile names.
    command = [
        sys.executable,
        "-m",
        "uniserve_testing.standins.server",
        "--port",
        str(port),
    ]
    profile = tmp_path / "profile.toml"
    profile.write_text(
        f"""
[servers.stand-in]
port = {port}
command = {json.dumps(command)}

[benchmarks.scraped]
server = "stand-in"
task = "text"
model = "model"
dataset = "jsonl"
dataset_path = "rows.jsonl"

[benchmarks.scraped.load]
num_prompts = 4
duration_s = 2.5
max_concurrency = 2
warmup_requests = 0

[benchmarks.scraped.metrics]
output_throughput = "higher"
""",
        encoding="utf-8",
    )

    argv = ["--config", str(profile), "run", "scraped"]
    assert evaluate([*argv, "--output-root", str(tmp_path / "out")]) == 0

    bundle = tmp_path / "out" / "scraped"
    timeline = json.loads((bundle / "timeline.json").read_text())
    start = timeline["load_start"]
    lines = (bundle / "scrapes.jsonl").read_text().splitlines()
    scrapes = sorted(
        (json.loads(line) for line in lines), key=lambda item: item["tick"]
    )
    records = [
        json.loads(line)
        for line in (bundle / "requests.jsonl").read_text().splitlines()
    ]

    # One scrape per tick, the first at the load's start and each a second
    # after the one before, until the last stream ended.
    assert [scrape["tick"] for scrape in scrapes] == list(range(len(scrapes)))
    assert all(
        scrape["origin"] == f"http://127.0.0.1:{port}" for scrape in scrapes
    )
    assert abs(scrapes[0]["sent"] - start) < 0.2
    for scrape in scrapes:
        assert abs(scrape["sent"] - scrapes[0]["sent"] - scrape["tick"]) < 0.2
        assert scrape["sent"] <= scrape["received"]
    last_end = max(record["final_event_time"] for record in records)
    assert scrapes[-1]["sent"] > last_end - 1.0
    assert len(scrapes) >= 3

    # Each scrape keeps the exposition the server answered, in order: the
    # stand-in's count of the scrapes it answered rises by each.
    counts = [
        _sample(scrape["text"], "stand_in_scrapes_total")
        for scrape in sorted(scrapes, key=lambda item: item["received"])
    ]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)

    # The records keep each token's arrival.
    for record in records:
        assert len(record["text_times"]) == 10
        assert record["text_times"] == sorted(record["text_times"])


# The bundles `check-stop` reads. The load starts at perf-counter time
# _START; times below are seconds from it.
_START = 1000.0
_STEP = 0.010
_STREAMS = 4
_TOKENS = 600
_STOP, _CONT = 3.0, 3.3

# The runs in which two streams are left out and the counter rises.
_LEFT_OUT = {"clean", "early", "late", "failed", "short"}


def _arrival(stream: int, step: int) -> float:
    """A token's arrival: its step's output plus the client's jitter."""
    jitter = 0.001 * ((step * 7 + stream * 5) % 3)
    return step * _STEP + 0.0005 * stream + jitter


def _stream_times(stream: int, kind: str) -> list[float]:
    """The arrival of each of a stream's tokens in a run of `kind`.

    In every run but the control and `slow`, the process is stopped over
    [_STOP, _CONT]. `clean`, `early`, `late`, `failed` and `short`: the
    steps go on, and streams 0 and 1 are left out after 8 tokens of the
    stop until 20 ms after it. `held`: no step completes from 50 ms into the
    stop until it ends. `slow`: the steps of the stop take 30 ms.
    """
    times: list[float] = []
    step = 100
    while len(times) < _TOKENS:
        time = _arrival(stream, step)
        stopped = _STOP <= time <= _CONT
        if kind == "held" and _STOP + 0.05 < time <= _CONT:
            step = round(_CONT / _STEP) + 1
            continue
        if kind in _LEFT_OUT and stream < 2 and stopped:
            if sum(_STOP <= item for item in times) >= 8:
                step = round((_CONT + 0.02) / _STEP)
                continue
        times.append(time)
        step += 3 if kind == "slow" and stopped else 1
    return times


def _write_bundle(root: Path, kind: str) -> Path:
    """Write one run's bundle as `uniserve-eval run` writes a timeline's.

    `short` records a stop of 50 ms, fewer steps than the headroom.
    """
    bundle = root / kind
    bundle.mkdir()
    changes = []
    if kind != "control":
        stop = _STOP
        cont = _CONT if kind != "short" else _STOP + 0.05
        changes = [
            {
                "name": "stop",
                "at_s": stop,
                "signal": "SIGSTOP",
                "process": "storage-server serve",
                "issued_s": stop,
                "ended_s": stop + 0.0001,
                "pid": 77,
                "cmdline": "uniserve storage-server serve",
            },
            {
                "name": "continue",
                "after": "stop",
                "delay_s": cont - stop,
                "signal": "SIGCONT",
                "process": "storage-server serve",
                "issued_s": cont,
                "ended_s": cont + 0.0001,
                "pid": 77,
                "cmdline": "uniserve storage-server serve",
            },
        ]
    (bundle / "timeline.json").write_text(
        json.dumps(
            {
                "load_start": _START,
                "clock_offset": 0.0,
                "changes": changes,
                "events": [],
                "issue_end_s": 7.0,
            }
        )
    )

    records = []
    for stream in range(_STREAMS):
        times = _stream_times(stream, kind)
        failed = kind == "failed" and stream == 0
        records.append(
            {
                "example": {"id": f"row-{stream}"},
                "scheduled_time": _START + 0.5,
                "client_send_time": _START + 0.5,
                "first_text_time": _START + times[0],
                "final_event_time": _START + times[-1] + 0.001,
                "token_times": [],
                "text_times": [_START + time for time in times],
                "success": not failed,
                "status_code": 200,
                "classifier": "response_missing_terminal" if failed else "ok",
                "error": "stream ended" if failed else None,
                "finish_reason": None if failed else "length",
                "output_len": _TOKENS,
                "output_len_source": "server_usage",
                "requested_output_len": _TOKENS,
            }
        )
    (bundle / "requests.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    # Scrapes every second from 0 s; the counter's rise is placed by kind:
    # in the scrape after the stop, before it, or long after it.
    rise_tick = {"early": 2, "late": 7}.get(kind, 4)
    scrapes = []
    for tick in range(9):
        left_out = 34 if kind in _LEFT_OUT and tick >= rise_tick else 0
        text = f'{_COUNTER}{{engine="0"}} {left_out}\n'
        scrapes.append(
            {
                "origin": "http://127.0.0.1:18081",
                "tick": tick,
                "sent": _START + tick + 0.01,
                "received": _START + tick + 0.02,
                "text": text,
            }
        )
    (bundle / "scrapes.jsonl").write_text(
        "".join(json.dumps(scrape) + "\n" for scrape in scrapes)
    )
    return bundle


@pytest.fixture(scope="module")
def bundles(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the control and every run; return their directory."""
    root = tmp_path_factory.mktemp("stops")
    for kind in (
        "control",
        "clean",
        "held",
        "slow",
        "early",
        "late",
        "failed",
        "short",
    ):
        _write_bundle(root, kind)
    (root / "listing.txt").write_text(
        "buffer pool of Qwen3-30B-A3B: 0 check failures\n"
    )
    (root / "replay.txt").write_text("ok    replay: 12 inputs; model: ok\n")
    return root


@pytest.mark.parametrize(
    ("kind", "failing"),
    [
        ("clean", set()),
        ("held", {"step_time_unchanged"}),
        ("slow", {"step_time_unchanged"}),
        ("early", {"left_out_after_headroom"}),
        ("late", {"left_out_after_headroom"}),
        ("failed", {"no_failed_request", "left_out_after_headroom"}),
        ("short", {"left_out_after_headroom"}),
    ],
)
def test_check_stop_holds_a_run_to_e2s_pass(
    bundles: Path,
    capsys: pytest.CaptureFixture[str],
    evaluate: Callable[[Sequence[str]], int],
    kind: str,
    failing: set[str],
) -> None:
    """A clean run passes; each forbidden behaviour fails its own items.

    `held` keeps every gap that ends in the window normal and fails on the
    window's silence; `short` lets the counter rise for a stop of fewer
    steps than the headroom; `failed` fails a request in flight at the
    stop's end.
    """
    result = bundles / f"{kind}.md"
    status = evaluate(
        [
            "check-stop",
            str(bundles / kind),
            "--control",
            str(bundles / "control"),
            "--headroom-tokens",
            "8",
            "--listing",
            str(bundles / "listing.txt"),
            "--replay",
            str(bundles / "replay.txt"),
            "--result",
            str(result),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    failed = {key for key, item in report["items"].items() if not item["pass"]}
    assert failed == failing
    assert report["pass"] is (not failing)
    assert status == (2 if failing else 0)
    assert report["check_failures"] == [0]
    assert report["replay"].startswith("ok")
    assert result.read_text().startswith(f"# E2, the process stopped: {kind}")


def test_check_stop_measures_the_window_as_the_client_saw_it(
    bundles: Path,
    capsys: pytest.CaptureFixture[str],
    evaluate: Callable[[Sequence[str]], int],
) -> None:
    """The clean run's window: its steps, left-out streams and their wait."""
    evaluate(
        [
            "check-stop",
            str(bundles / "clean"),
            "--control",
            str(bundles / "control"),
            "--headroom-tokens",
            "8",
            "--result",
            str(bundles / "measures.md"),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    (window,) = report["items"]["step_time_unchanged"]["windows"]
    assert window["window_s"] == [_STOP, _CONT]
    assert window["duration_ms"] == pytest.approx(300.0)

    # Every stream was in flight at the stop's end; the two left-out ones
    # waited until 20 ms after it for their next token.
    assert window["in_flight"] == _STREAMS and window["in_flight_failed"] == 0
    assert window["longest_wait_ms"] == pytest.approx(20.0, abs=3.0)
    assert window["most_tokens"] == pytest.approx(30, abs=1)

    # The steps went on at 10 ms with the control's jitter, so their gaps
    # equal the control's; the counter's rise is the window's.
    assert window["run"]["p50_ms"] == pytest.approx(10.0, abs=1.5)
    assert window["run"]["max_ms"] <= window["control"]["max_ms"]
    item = report["items"]["left_out_after_headroom"]
    assert item["rises_by_window"] == [34] and item["total_rise"] == 34
