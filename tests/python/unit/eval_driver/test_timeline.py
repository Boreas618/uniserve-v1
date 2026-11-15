"""Timeline loads with changes, and `check-change`, against a stand-in.

The stand-in is a real process that `uniserve-eval run` launches from the
profile's command, :mod:`uniserve_testing.standins.server`. Like
`uniserve serve`, it serves an OpenAI-compatible chat API, here streaming
each request's tokens with their ids and top-5 logprobs, and an admin
listener on a Unix socket in its runtime directory, where an update moves
through `switching` and `trial` and is `committed` or `rolled_back`; it
starts a detached Storage Server there and an engine in a process group of
its own. A point's requests carry a `fault` the stand-in reads: it strikes
the point's third request with a lost token, a repeated token, a failed
answer, a closed connection, or a divergence inside or outside the top 5;
`late` instead holds the first token of every request that arrives during
the update's switch, `slow` spaces every request's tokens by 0.1 s, and
`uncached` reports nothing cached once an update switched.

The stand-in serves configuration `A` until an update sets
`configuration=B`, and `A` again after a rollback; a request keeps the
configuration it arrived in, or names its own (`configuration` in its
body), and B's tokens differ from A's at every position. Its usage counts
64 prompt tokens, all but the last cached when it served the row before.
It logs the prompt of each stream its client closed to `disconnected.txt`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from uniserve_testing.standins.server import token

from uniserve_eval.config import load_config

pytestmark = pytest.mark.unit

_TOKENS = 8
_ROWS = 6

# The update most points apply: its switch at 0.3 s and its commit 0.2 s
# after the switch's end, as the admin events report it.
_UPDATE = """
[[benchmarks.{name}.changes]]
name = "switch"
at_s = 0.3
method = "PUT"
path = "/admin/config"
body = {{ set = ["scheduler_config.policy=fifo"], wait = true }}

[[benchmarks.{name}.changes]]
name = "commit"
after = "switch"
phase = "trial"
delay_s = 0.2
method = "POST"
path = "/admin/config/commit"
body = {{}}
"""

# The clean point also runs a command in the deployment's environment, which
# prints the runtime directory that environment names, and stops its Storage
# Server for 50 ms.
_CLEAN = """
[[benchmarks.{name}.changes]]
name = "runtime"
at_s = 0.05
command = ["printenv", "XDG_RUNTIME_DIR"]

[[benchmarks.{name}.changes]]
name = "stop"
at_s = 0.1
signal = "SIGSTOP"
process = "storage-server serve"

[[benchmarks.{name}.changes]]
name = "continue"
after = "stop"
delay_s = 0.05
signal = "SIGCONT"
process = "storage-server serve"
"""

# An epoch to configuration B at 0.3 s, rolled back to A 0.2 s after the
# switch's end.
_MOVE = """
[[benchmarks.{name}.changes]]
name = "switch"
at_s = 0.3
method = "PUT"
path = "/admin/config"
body = {{ set = ["configuration=B"], wait = true }}
configuration = "B"

[[benchmarks.{name}.changes]]
name = "rollback"
after = "switch"
phase = "trial"
delay_s = 0.2
method = "POST"
path = "/admin/config/rollback"
body = {{}}
configuration = "A"
"""

# Two of the four streams disconnected at once, inside the update's window.
_CANCEL = """
[[benchmarks.{name}.changes]]
name = "cancel"
at_s = 0.35
cancel = 2
"""

# The engine, in a process group of its own, continued by a signal found by
# its command line among the deployment's processes.
_ENGINE = """
[[benchmarks.{name}.changes]]
name = "continue"
at_s = 0.1
signal = "SIGCONT"
process = "uniserve-engine"
"""

# Each point's fault and changes, in run order. The control applies no
# change; `alone` serves each row once with nothing else running, the
# reference of case E6; `control_b` is the control of configuration B.
_POINTS: dict[str, tuple[str | None, str]] = {
    "control": (None, ""),
    "alone": (None, ""),
    "control_b": (None, ""),
    "clean": (None, _UPDATE + _CLEAN),
    "tie_inside": ("tie_inside", _UPDATE),
    "tie_outside": ("tie_outside", _UPDATE),
    "lost": ("lost", _UPDATE),
    "repeat": ("repeat", _UPDATE),
    "fail": ("fail", _UPDATE),
    "close": ("close", _UPDATE),
    "late": ("late", _UPDATE),
    "moved": (None, _MOVE),
    "cancel": ("slow", _UPDATE + _CANCEL),
    "engine": (None, _ENGINE),
    "uncached": ("uncached", _UPDATE),
}


def _point(name: str, fault: str | None, changes: str) -> str:
    """Return a point over 4 streams with its changes.

    Every point but `alone` is a timeline of 1 s; `alone` sends each row
    once, one request at a time.
    """
    body = "logprobs = true, top_logprobs = 5, return_token_ids = true, "
    body += "return_tokens_as_token_ids = true"
    if fault:
        body += f', fault = "{fault}"'
    if name == "control_b":
        body += ', configuration = "B"'
    load = "duration_s = 1.0\nsettle_s = 0.2\nmax_concurrency = 4"
    if name == "alone":
        load = "max_concurrency = 1"
    point = f"""
[benchmarks.{name}]
server = "stand-in"
task = "text"
model = "model"
dataset = "jsonl"
dataset_path = "rows.jsonl"

[benchmarks.{name}.load]
num_prompts = {_ROWS}
{load}
warmup_requests = 0

[benchmarks.{name}.sampling]
extra_body = {{ {body} }}

[benchmarks.{name}.metrics]
output_throughput = "higher"
"""
    return point + changes.format(name=name)


def _records(bundle: Path) -> list[dict]:
    lines = (bundle / "requests.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _pid(path: Path) -> int:
    """The pid a stand-in's pid file records first."""
    return int(path.read_text().split()[0])


@pytest.fixture(scope="module")
def runs(
    tmp_path_factory: pytest.TempPathFactory,
    unused_tcp_port_factory: Callable[[], int],
    evaluate: Callable[[Sequence[str]], int],
) -> Path:
    """Run every point once on one stand-in deployment; return its tree."""
    root = tmp_path_factory.mktemp("timeline")
    (root / "rows.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": f"row-{i}",
                    "prompt": f"prompt {i}",
                    "output_len": _TOKENS,
                }
            )
            + "\n"
            for i in range(_ROWS)
        ),
        encoding="utf-8",
    )
    # The profile states the port the stand-in listens on, since the
    # evaluator probes the address its profile names.
    port = unused_tcp_port_factory()
    command = [
        sys.executable,
        "-m",
        "uniserve_testing.standins.server",
        "--port",
        str(port),
    ]
    server = f"""
[servers.stand-in]
port = {port}
command = {json.dumps(command)}
admin_socket = "uniserve/admin-{port}.sock"

[suites.all]
points = {json.dumps(list(_POINTS))}
"""
    profile = root / "profile.toml"
    profile.write_text(
        server
        + "".join(_point(name, *point) for name, point in _POINTS.items()),
        encoding="utf-8",
    )
    (root / "replay-pass.json").write_text('{"verdict": "pass"}')
    (root / "replay-fail.json").write_text('{"verdict": "fail"}')

    # The failed request and the closed connection fail their own points'
    # validation, so the run ends with status 2 after every point ran.
    argv = ["--config", str(profile), "run", "all", "--reuse-deployment"]
    argv += ["--keep-going", "--output-root", str(root / "results")]
    assert evaluate(argv) == 2
    return root


@pytest.fixture
def check(
    runs: Path,
    capsys: pytest.CaptureFixture[str],
    evaluate: Callable[[Sequence[str]], int],
) -> Callable[..., tuple[int, dict, set[str]]]:
    """Run a check command of a point's bundle with the passing evidence.

    The returned function takes the command, the point and the command's
    other options, and returns its status, report and failed items; the
    report is empty when the command could not check.
    """

    def run(
        command: str, point: str, *options: str
    ) -> tuple[int, dict, set[str]]:
        status = evaluate(
            [
                command,
                str(runs / "results" / point),
                *options,
                "--check-metric",
                "stand_in_check_failures_total",
                "--replay",
                str(runs / "replay-pass.json"),
                "--result",
                str(runs / f"{command}-{point}.md"),
            ]
        )
        out = capsys.readouterr().out
        report = json.loads(out) if out else {}
        failed = {
            key
            for key, item in report.get("items", {}).items()
            if not item["pass"]
        }
        return status, report, failed

    return run


def test_a_timeline_load_applies_its_changes_while_streams_run(
    runs: Path,
) -> None:
    """Each change acts at its time, on the load's clock, under load."""
    bundle = runs / "results" / "clean"
    timeline = json.loads((bundle / "timeline.json").read_text())
    start = timeline["load_start"]
    changes = {change["name"]: change for change in timeline["changes"]}
    records = _records(bundle)

    # The update's call is answered once its switch ended; its commit is
    # issued 0.2 s after the trial event the event log reported.
    switch, commit = changes["switch"], changes["commit"]
    assert switch["issued_s"] >= 0.3 and switch["status"] == 200
    assert switch["answer"] == {"kind": "config", "state": "succeeded"}
    (trial,) = [
        event
        for event in timeline["events"]
        if event.get("data", {}).get("phase") == "trial"
        and event["at_s"] >= switch["issued_s"]
    ]
    assert trial["at_s"] <= switch["ended_s"]
    assert commit["issued_s"] >= trial["at_s"] + 0.2
    assert commit["status"] == 200 and "error" not in commit

    # Streams ran across the call: some request was issued before it and
    # received tokens after it.
    assert any(
        record["scheduled_time"] - start
        < switch["issued_s"]
        < record["token_times"][-1] - start
        for record in records
    )

    # The signals went to the deployment's Storage Server, 50 ms apart, and
    # the command ran in the deployment's runtime directory.
    storage = _pid(runs / "storage-server.pid")
    assert changes["stop"]["pid"] == changes["continue"]["pid"] == storage
    runtime = changes["runtime"]["output"].strip()
    assert changes["runtime"]["returncode"] == 0
    assert f"--socket {runtime}/uniserve/" in changes["stop"]["cmdline"]
    assert changes["continue"]["issued_s"] >= changes["stop"]["ended_s"] + 0.05

    # Issuing ran to the load's duration, and the point records its changes
    # as applied.
    issued = [record["scheduled_time"] - start for record in records]
    assert timeline["issue_end_s"] >= 1.0
    assert 0.8 < max(issued) < timeline["issue_end_s"] + 0.01
    summary = json.loads((bundle / "summary.json").read_text())
    assert summary["validation"]["checks"]["changes_applied"] is True


def test_the_records_hold_every_token_of_every_stream(runs: Path) -> None:
    """Each record carries its stream's ids, candidates, times and usage."""
    records = _records(runs / "results" / "clean")

    # Rows recur in issue order: request n took row n mod 6.
    assert len(records) > _ROWS
    assert [record["example"]["id"] for record in records] == [
        f"row-{index % _ROWS}" for index in range(len(records))
    ]
    for record in records:
        prompt = record["example"]["prompt"]
        ids = [token(prompt, position) for position in range(_TOKENS)]
        assert record["status_code"] == 200
        assert record["finish_reason"] == "length"
        assert record["token_ids"] == ids
        assert [top[0] for top in record["top_logprobs"]] == [
            [chosen, -0.1] for chosen in ids
        ]
        assert all(len(top) == 5 for top in record["top_logprobs"])
        assert record["token_times"] == sorted(record["token_times"])
        assert record["first_text_time"] == record["token_times"][0]
        # Eight token chunks and the final one with the usage, whose
        # cached tokens are the whole prompt's but its last token, which an
        # admission always computes: the control served every row before.
        assert len(record["chunk_times"]) == _TOKENS + 1
        assert record["cached_prompt_tokens"] == 63


@pytest.mark.parametrize(
    ("point", "failing"),
    [
        ("clean", set()),
        ("tie_inside", set()),
        ("tie_outside", {"no_token_lost_or_repeated"}),
        ("lost", {"no_token_lost_or_repeated"}),
        ("repeat", {"no_token_lost_or_repeated"}),
        ("fail", {"no_failed_request", "no_token_lost_or_repeated"}),
        (
            "close",
            {
                "no_failed_request",
                "no_closed_connection",
                "no_token_lost_or_repeated",
            },
        ),
        ("late", {"admitted_throughout"}),
    ],
)
def test_check_change_holds_a_run_to_the_pass_under_change(
    runs: Path,
    capsys: pytest.CaptureFixture[str],
    evaluate: Callable[[Sequence[str]], int],
    point: str,
    failing: set[str],
) -> None:
    """A clean run passes; each fault fails the items it violates."""
    result = runs / f"{point}.md"
    status = evaluate(
        [
            "check-change",
            str(runs / "results" / point),
            "--control",
            str(runs / "results" / "control"),
            "--check-metric",
            "stand_in_check_failures_total",
            "--replay",
            str(runs / "replay-pass.json"),
            "--result",
            str(result),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    failed = {key for key, item in report["items"].items() if not item["pass"]}
    assert failed == failing
    assert report["pass"] is (not failing)
    assert status == (2 if failing else 0)
    assert result.read_text().startswith(f"# Pass under change: {point}")


def test_a_cancel_disconnects_streams_in_flight_and_they_go_on(
    runs: Path, check: Callable[..., tuple[int, dict, set[str]]]
) -> None:
    """A cancel closes two in-flight streams at once by the client.

    Their requests are recorded as cancelled, the server sees the two
    streams closed, the streams issue their next requests, and the point
    and its check leave the cancelled requests out.
    """
    bundle = runs / "results" / "cancel"
    timeline = json.loads((bundle / "timeline.json").read_text())
    start = timeline["load_start"]
    (cancel,) = [c for c in timeline["changes"] if c["name"] == "cancel"]
    records = _records(bundle)

    assert "error" not in cancel and len(cancel["cancelled"]) == 2
    cancelled = [records[index] for index in cancel["cancelled"]]
    assert [record["classifier"] for record in records].count("cancelled") == 2
    for record in cancelled:
        assert record["classifier"] == "cancelled"
        assert record["scheduled_time"] - start < cancel["issued_s"]
        assert record["final_event_time"] - start >= cancel["issued_s"]
    disconnected = (runs / "disconnected.txt").read_text().splitlines()
    assert sorted(disconnected) == sorted(
        record["example"]["prompt"] for record in cancelled
    )
    # Every stream ran on: the two cancelled ones issued their next
    # requests at once, and the two others when their requests ended.
    after = [
        record
        for record in records
        if record["scheduled_time"] - start > cancel["issued_s"]
    ]
    assert len(after) >= 4

    summary = json.loads((bundle / "summary.json").read_text())
    assert summary["validation"]["valid"] is True
    assert summary["cancelled_count"] == 2
    assert summary["request_count"] == len(records) - 2

    control = str(runs / "results" / "control")
    status, report, failed = check(
        "check-change", "cancel", "--control", control
    )
    assert (status, failed, report["cancelled"]) == (0, set(), 2)


def test_a_signal_reaches_a_process_a_server_started_in_its_own_group(
    runs: Path,
) -> None:
    """A signal finds a server's descendant outside the server's group.

    The daemon starts each engine as the leader of a process group of its
    own; the deployment's processes include it, so a signal found by its
    command line reaches it.
    """
    timeline = json.loads(
        (runs / "results" / "engine" / "timeline.json").read_text()
    )
    changes = {change["name"]: change for change in timeline["changes"]}
    engine = _pid(runs / "engine.pid")

    resume = changes["continue"]
    assert "error" not in resume
    assert resume["pid"] == engine
    assert "uniserve-engine" in resume["cmdline"]


def test_check_change_compares_a_request_with_its_configurations_control(
    runs: Path, check: Callable[..., tuple[int, dict, set[str]]]
) -> None:
    """A request is compared with the control of its configuration.

    A run moved to B and rolled back to A passes against each
    configuration's control; B's requests fail against A's control, and a
    configuration without a control cannot be checked.
    """
    records = _records(runs / "results" / "moved")
    on_b = [
        record
        for record in records
        if record["token_ids"][0] != token(record["example"]["prompt"], 0)
    ]
    assert on_b, "some request ended on configuration B"

    control = str(runs / "results" / "control")
    control_b = str(runs / "results" / "control_b")
    status, report, failed = check(
        "check-change",
        "moved",
        "--control",
        control,
        "--control-of",
        f"B={control_b}",
        "--control-of",
        f"A={control}",
    )
    assert (status, failed) == (0, set())
    assert report["controls"] == {"B": control_b, "A": control}

    status, _, failed = check(
        "check-change",
        "moved",
        "--control",
        control,
        "--control-of",
        f"B={control}",
        "--control-of",
        f"A={control}",
    )
    assert (status, failed) == (2, {"no_token_lost_or_repeated"})

    status, report, _ = check("check-change", "moved", "--control", control)
    assert (status, report) == (1, {})


@pytest.mark.parametrize(
    ("point", "passed"), [("clean", True), ("uncached", False)]
)
def test_check_change_with_a_block_size_requires_the_prompts_cached(
    check: Callable[..., tuple[int, dict, set[str]]],
    runs: Path,
    point: str,
    passed: bool,
) -> None:
    """With a block size, the target must find the prompts cached.

    The first admission after the switch of each prompt served before it
    must report cached the whole blocks of its prompt less the last token
    (case E4's item); a target
    that found nothing fails that item alone.
    """
    control = str(runs / "results" / "control")
    status, report, failed = check(
        "check-change",
        point,
        "--control",
        control,
        "--switch",
        "switch",
        "--block-size",
        "16",
    )
    item = report["items"]["cached_across_switch"]
    assert item["of"] > 0 and item["served"] > 0
    assert item["pass"] is passed
    assert failed == (set() if passed else {"cached_across_switch"})
    assert status == (0 if passed else 2)


def test_rows_served_alone_are_the_reference_of_faults(
    runs: Path, check: Callable[..., tuple[int, dict, set[str]]]
) -> None:
    """Rows served alone are the reference `check-faults` holds a run to.

    A point of one stream serves each row once with nothing else running;
    `check-faults` holds every request not cancelled to its row's tokens
    there, and refuses a reference that ran requests side by side.
    """
    records = _records(runs / "results" / "alone")
    assert [record["example"]["id"] for record in records] == [
        f"row-{index}" for index in range(_ROWS)
    ]
    for before, after in zip(records, records[1:], strict=False):
        assert after["client_send_time"] >= before["final_event_time"]

    alone = str(runs / "results" / "alone")
    status, report, failed = check(
        "check-faults", "cancel", "--reference", alone
    )
    assert (status, failed, report["cancelled"]) == (0, set(), 2)
    assert (
        (runs / "check-faults-cancel.md")
        .read_text()
        .startswith("# Faults under load: cancel")
    )

    status, _, failed = check(
        "check-faults", "tie_outside", "--reference", alone
    )
    assert (status, failed) == (2, {"same_tokens_as_alone"})

    control = str(runs / "results" / "control")
    status, report, _ = check("check-faults", "clean", "--reference", control)
    assert (status, report) == (1, {})


@pytest.mark.parametrize(
    ("metric", "replay"),
    [
        ("stand_in_check_failures_total", "replay-fail.json"),
        ("absent_check_failures_total", "replay-pass.json"),
    ],
)
def test_check_change_requires_the_invariants_evidence(
    runs: Path,
    capsys: pytest.CaptureFixture[str],
    evaluate: Callable[[Sequence[str]], int],
    metric: str,
    replay: str,
) -> None:
    """A failed replay, or a check count no scrape reports, fails item 5."""
    status = evaluate(
        [
            "check-change",
            str(runs / "results" / "clean"),
            "--control",
            str(runs / "results" / "control"),
            "--check-metric",
            metric,
            "--replay",
            str(runs / replay),
            "--result",
            str(runs / "invariants.md"),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    failed = {key for key, item in report["items"].items() if not item["pass"]}
    assert status == 2 and failed == {"invariants_hold"}


@pytest.mark.parametrize(
    ("load", "admin", "changes", "message"),
    [
        ("", True, _UPDATE, "require load.duration_s"),
        ("duration_s = 1.0", False, _UPDATE, "states no admin_socket"),
        (
            "duration_s = 1.0",
            True,
            '[[benchmarks.p.changes]]\nname = "commit"\nafter = "switch"\n'
            'method = "POST"\npath = "/admin/config/commit"\n',
            "must name an earlier change",
        ),
        (
            "duration_s = 1.0",
            True,
            '[[benchmarks.p.changes]]\nname = "kill"\nat_s = 1.0\n'
            'signal = "SIGKILL"\n',
            "signal and process go together",
        ),
        (
            "duration_s = 1.0\nrequest_rate = 5.0",
            True,
            '[[benchmarks.p.changes]]\nname = "cut"\nat_s = 1.0\ncancel = 2\n',
            "streams of a closed loop",
        ),
        (
            "duration_s = 1.0",
            True,
            '[[benchmarks.p.changes]]\nname = "cut"\nat_s = 1.0\ncancel = 5\n',
            "streams of a closed loop",
        ),
        (
            "duration_s = 1.0",
            True,
            '[[benchmarks.p.changes]]\nname = "run"\nat_s = 1.0\n'
            'command = ["true"]\nhost = "elsewhere"\n',
            "host names the host of a signal's process",
        ),
    ],
)
def test_a_profile_rejects_changes_it_cannot_apply(
    tmp_path: Path, load: str, admin: bool, changes: str, message: str
) -> None:
    """A change needs a timeline, its listener, and an earlier reference."""
    socket_line = 'admin_socket = "admin.sock"' if admin else ""
    profile = tmp_path / "profile.toml"
    profile.write_text(
        f"""
[servers.s]
port = 8000
command = ["server"]
{socket_line}

[benchmarks.p]
server = "s"
task = "text"
model = "model"
dataset = "jsonl"
dataset_path = "rows.jsonl"

[benchmarks.p.load]
max_concurrency = 4
{load}

[benchmarks.p.metrics]
output_throughput = "higher"
"""
        + changes.format(name="p"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(profile)
