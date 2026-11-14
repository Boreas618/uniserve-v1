"""Timeline loads with changes, and `check-change`, against a stand-in.

The stand-in is a real process that `uniserve-eval run` launches from the
profile's command. Like `uniserve serve`, it serves an OpenAI-compatible
chat API, here streaming each request's tokens with their ids and top-5
logprobs, and an admin listener on a Unix socket in its runtime directory,
where an update moves through `switching` and `trial` and is `committed`
or `rolled_back`; and it starts a detached Storage Server there. A point's
requests carry a `fault` the stand-in reads: it strikes the point's third
request with a lost token, a repeated token, a failed answer, a closed
connection, or a divergence inside or outside the top 5; `late` instead
holds the first token of every request that arrives during the update's
switch, `slow` spaces every request's tokens by 0.1 s, and `uncached`
reports nothing cached once an update switched.

The stand-in serves configuration `A` until an update sets
`configuration=B`, and `A` again after a rollback; a request keeps the
configuration it arrived in, or names its own (`configuration` in its
body), and B's tokens differ from A's at every position. Its usage counts
64 prompt tokens, all but the last cached when it served the row before.
It logs the prompt of each stream its client closed to `disconnected.txt`.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import uuid
import zlib
from pathlib import Path

import pytest

from uniserve_eval.cli import build_parser
from uniserve_eval.config import load_config

pytestmark = pytest.mark.unit

_TOKENS = 8
_ROWS = 6

# Tells this module's stand-in Storage Server apart from every other process
# on the host, so a signal found by pattern on the host reaches only it.
_MARK = uuid.uuid4().hex

# The stand-in server; its arguments are the port it listens on and the mark
# its Storage Server's command line carries.
_SERVER = r"""
import asyncio, json, os, subprocess, sys, time, zlib

from aiohttp import web

port, mark = int(sys.argv[1]), sys.argv[2]
services = os.path.join(os.environ["XDG_RUNTIME_DIR"], "uniserve")
os.makedirs(services, exist_ok=True)
storage = subprocess.Popen(
    [
        sys.executable, "-c", "import time; time.sleep(300)",
        "storage-server", "serve",
        "--socket", os.path.join(services, "storage-server.sock"),
        "--stand-in", mark,
    ],
    start_new_session=True,
)
with open("storage-server.pid", "w") as handle:
    handle.write(str(storage.pid))

events, faults, switching = [], {}, False
configuration, switched, seen = "A", False, set()
appended = asyncio.Condition()


def token(row, position, served="A"):
    key = f"{row}/{position}" if served == "A" else f"{served}/{row}/{position}"
    return 1000 + zlib.crc32(key.encode()) % 50000


def chunk(event):
    return b"data: " + json.dumps(event).encode() + b"\n\n"


async def record(phase):
    events.append({
        "seq": len(events) + 1, "time": time.time(), "kind": "deployment",
        "message": f"update 1: {phase}", "data": {"phase": phase},
    })
    async with appended:
        appended.notify_all()


async def chat(request):
    body = await request.json()
    row, count = body["messages"][0]["content"], body["max_completion_tokens"]
    fault = body.get("fault")
    faults[fault] = faults.get(fault, 0) + 1
    struck = fault is not None and faults[fault] == 3
    if struck and fault == "fail":
        return web.json_response({"error": {"message": "failed"}}, status=500)
    held = fault == "late" and switching
    served = body.get("configuration", configuration)
    cached = row in seen and not (fault == "uncached" and switched)
    seen.add(row)

    response = web.StreamResponse(headers={"content-type": "text/event-stream"})
    await response.prepare(request)
    await asyncio.sleep(1.0 if held else 0.02)
    positions = list(range(count))
    if struck and fault == "lost":
        positions.remove(4)
    if struck and fault == "repeat":
        positions.insert(4, 4)
    try:
        for index, position in enumerate(positions):
            chosen = token(row, position, served)
            others = [chosen + 100000 * rank for rank in range(1, 5)]
            if struck and position == 4 and fault == "tie_inside":
                chosen, others[0] = others[0], chosen
            if struck and position == 4 and fault == "tie_outside":
                chosen, others = 7, [8, 9, 10, 11]
            top = [
                {"token": f"token_id:{item}", "logprob": -0.1 - rank}
                for rank, item in enumerate([chosen, *others])
            ]
            entry = {
                "token": f"token_id:{chosen}", "logprob": -0.1,
                "top_logprobs": top,
            }
            choice = {
                "index": 0, "delta": {"content": f" {chosen}"},
                "token_ids": [chosen], "logprobs": {"content": [entry]},
            }
            await response.write(chunk({"choices": [choice]}))
            if struck and fault == "close" and index == 2:
                request.transport.close()
                return response
            await asyncio.sleep(0.1 if fault == "slow" else 0.02)
    except (ConnectionResetError, asyncio.CancelledError):
        with open("disconnected.txt", "a") as handle:
            handle.write(row + "\n")
        raise
    usage = {
        "prompt_tokens": 64, "completion_tokens": count,
        "prompt_tokens_details": {"cached_tokens": 63 if cached else 0},
    }
    final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
    await response.write(chunk({**final, "usage": usage}) + b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def change(request):
    global switching, switched, configuration
    body = await request.json()
    switching, switched = True, True
    for setting in body.get("set", []):
        name, _, value = setting.partition("=")
        if name == "configuration":
            configuration = value
    await record("switching")
    await asyncio.sleep(0.2)
    switching = False
    await record("trial")
    return web.json_response({"kind": "config", "state": "succeeded"})


async def commit(request):
    await request.read()
    await record("committed")
    return web.json_response({"kind": "commit", "state": "succeeded"})


async def rollback(request):
    global configuration
    await request.read()
    configuration = "A"
    await record("rolled_back")
    return web.json_response({"kind": "rollback", "state": "succeeded"})


async def read_events(request):
    after = int(request.query.get("after", "0"))
    async with appended:
        try:
            await asyncio.wait_for(
                appended.wait_for(lambda: len(events) > after),
                float(request.query.get("wait", "0")),
            )
        except asyncio.TimeoutError:
            pass
    page = {"events": events[after:], "next": max(after, len(events))}
    return web.json_response({**page, "gap": False})


async def metrics(request):
    return web.Response(text='stand_in_check_failures_total{pool="0"} 0\n')


async def version(request):
    return web.json_response({"version": "stand-in"})


async def main():
    admin = web.Application()
    admin.router.add_put("/admin/config", change)
    admin.router.add_post("/admin/config/commit", commit)
    admin.router.add_post("/admin/config/rollback", rollback)
    admin.router.add_get("/admin/events", read_events)
    api = web.Application()
    api.router.add_post("/v1/chat/completions", chat)
    api.router.add_get("/metrics", metrics)
    api.router.add_get("/version", version)
    admin_runner, api_runner = web.AppRunner(admin), web.AppRunner(api)
    await admin_runner.setup()
    await api_runner.setup()
    path = os.path.join(services, f"admin-{port}.sock")
    await web.UnixSite(admin_runner, path).start()
    # The client listener last: the evaluator's readiness probe.
    await web.TCPSite(api_runner, "127.0.0.1", port).start()
    await asyncio.Event().wait()


asyncio.run(main())
"""

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

# The clean point also runs a command in the deployment's environment and
# stops its Storage Server for 50 ms.
_CLEAN = """
[[benchmarks.{name}.changes]]
name = "runtime"
at_s = 0.05
command = ["sh", "-c", "echo $XDG_RUNTIME_DIR"]

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

# The Storage Server stopped for 50 ms through another host, found there by
# the mark its command line carries; `ssh` is the stand-in's.
_REMOTE = """
[[benchmarks.{name}.changes]]
name = "stop"
at_s = 0.1
signal = "SIGSTOP"
process = "stand-in {mark}"
host = "standin-host"

[[benchmarks.{name}.changes]]
name = "continue"
after = "stop"
delay_s = 0.05
signal = "SIGCONT"
process = "stand-in {mark}"
host = "standin-host"
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
    "remote": (None, _REMOTE),
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
    return point + changes.format(name=name, mark=_MARK)


def _command(argv: list[str]) -> int:
    """Run one evaluator command and return its exit status."""
    args = build_parser().parse_args(argv)
    try:
        args.function(args)
    except SystemExit as exit_info:
        return int(exit_info.code or 0)
    return 0


def _token(row: str, position: int) -> int:
    """The stand-in's token at a position of a prompt's output."""
    return 1000 + zlib.crc32(f"{row}/{position}".encode()) % 50000


def _records(bundle: Path) -> list[dict]:
    lines = (bundle / "requests.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run every point once on one stand-in deployment; return its tree."""
    root = tmp_path_factory.mktemp("timeline")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    (root / "server.py").write_text(_SERVER, encoding="utf-8")
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
    server = f"""
[servers.stand-in]
port = {port}
command = [{json.dumps(sys.executable)}, "server.py", "{port}", "{_MARK}"]
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

    # The stand-in for ssh, the boundary to another host: it logs the host
    # and runs the remote command here with the shell, as sshd runs it
    # there.
    tools = root / "bin"
    tools.mkdir()
    (tools / "ssh").write_text(
        "#!/bin/sh\n"
        'while [ "$1" = -o ]; do shift 2; done\n'
        f'echo "$1" >> {root / "ssh.log"}\n'
        "shift\n"
        'exec sh -c "$*"\n',
        encoding="utf-8",
    )
    (tools / "ssh").chmod(0o755)

    # The failed request and the closed connection fail their own points'
    # validation, so the run ends with status 2 after every point ran.
    argv = ["--config", str(profile), "run", "all", "--reuse-deployment"]
    argv += ["--keep-going", "--output-root", str(root / "results")]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
        assert _command(argv) == 2
    return root


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
    storage = int((runs / "storage-server.pid").read_text())
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
        ids = [_token(prompt, position) for position in range(_TOKENS)]
        assert record["status_code"] == 200
        assert record["finish_reason"] == "length"
        assert record["token_ids"] == ids
        assert [top[0] for top in record["top_logprobs"]] == [
            [token, -0.1] for token in ids
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
    point: str,
    failing: set[str],
) -> None:
    """A clean run passes; each fault fails the items it violates."""
    result = runs / f"{point}.md"
    status = _command(
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


def _check(
    runs: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    point: str,
    *options: str,
) -> tuple[int, dict, set[str]]:
    """Run a check of `point`'s bundle with the passing evidence.

    Returns its status, report and failed items; the report is empty when
    the command could not check.
    """
    status = _command(
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
        key for key, item in report.get("items", {}).items() if not item["pass"]
    }
    return status, report, failed


def test_a_cancel_disconnects_streams_in_flight_and_they_go_on(
    runs: Path, capsys: pytest.CaptureFixture[str]
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
    status, report, failed = _check(
        runs, capsys, "check-change", "cancel", "--control", control
    )
    assert (status, failed, report["cancelled"]) == (0, set(), 2)


def test_a_signal_reaches_a_process_on_another_host(runs: Path) -> None:
    """A signal with a host goes through ssh to that host.

    There the one process whose command line matches is found and
    signalled.
    """
    timeline = json.loads(
        (runs / "results" / "remote" / "timeline.json").read_text()
    )
    changes = {change["name"]: change for change in timeline["changes"]}
    storage = int((runs / "storage-server.pid").read_text())

    stop, resume = changes["stop"], changes["continue"]
    assert "error" not in stop and "error" not in resume
    assert stop["host"] == resume["host"] == "standin-host"
    assert stop["pid"] == resume["pid"] == storage
    assert f"--stand-in {_MARK}" in stop["cmdline"]
    assert resume["issued_s"] >= stop["ended_s"] + 0.05
    hosts = (runs / "ssh.log").read_text().split()
    assert hosts == ["standin-host", "standin-host"]


def test_check_change_compares_a_request_with_its_configurations_control(
    runs: Path, capsys: pytest.CaptureFixture[str]
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
        if record["token_ids"][0] != _token(record["example"]["prompt"], 0)
    ]
    assert on_b, "some request ended on configuration B"

    control = str(runs / "results" / "control")
    control_b = str(runs / "results" / "control_b")
    status, report, failed = _check(
        runs,
        capsys,
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

    status, _, failed = _check(
        runs,
        capsys,
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

    status, report, _ = _check(
        runs, capsys, "check-change", "moved", "--control", control
    )
    assert (status, report) == (1, {})


@pytest.mark.parametrize(
    ("point", "passed"), [("clean", True), ("uncached", False)]
)
def test_check_change_with_a_block_size_requires_the_prompts_cached(
    runs: Path, capsys: pytest.CaptureFixture[str], point: str, passed: bool
) -> None:
    """With a block size, the target must find the prompts cached.

    The first admission after the switch of each prompt served before it
    must report cached the whole blocks of its prompt less the last token
    (case E4's item); a target
    that found nothing fails that item alone.
    """
    control = str(runs / "results" / "control")
    status, report, failed = _check(
        runs,
        capsys,
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
    runs: Path, capsys: pytest.CaptureFixture[str]
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
    status, report, failed = _check(
        runs, capsys, "check-faults", "cancel", "--reference", alone
    )
    assert (status, failed, report["cancelled"]) == (0, set(), 2)
    assert (
        (runs / "check-faults-cancel.md")
        .read_text()
        .startswith("# Faults under load: cancel")
    )

    status, _, failed = _check(
        runs, capsys, "check-faults", "tie_outside", "--reference", alone
    )
    assert (status, failed) == (2, {"same_tokens_as_alone"})

    control = str(runs / "results" / "control")
    status, report, _ = _check(
        runs, capsys, "check-faults", "clean", "--reference", control
    )
    assert (status, report) == (1, {})


@pytest.mark.parametrize(
    ("metric", "replay"),
    [
        ("stand_in_check_failures_total", "replay-fail.json"),
        ("absent_check_failures_total", "replay-pass.json"),
    ],
)
def test_check_change_requires_the_invariants_evidence(
    runs: Path, capsys: pytest.CaptureFixture[str], metric: str, replay: str
) -> None:
    """A failed replay, or a check count no scrape reports, fails item 5."""
    status = _command(
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
