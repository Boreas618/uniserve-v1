"""Online profiling of a serving deployment through the admin API.

Qwen3 serves on one GPU (``UNISERVE_PROFILING_DEVICE``, default 0), with a
Storage Server of the test's own in a runtime directory the test creates
(``tempfile``'s directory, ``TMPDIR`` when set):

- The fence probes are always on: every decode step's span and its
  transformer layers' time reach the Prometheus histogram, and switching
  the probes off through ``PUT /admin/config`` is a change in place that
  stops them, until they are switched on again.
- A session with every collector records its window of decode steps into
  one file per worker: CUPTI's activity records and API calls, the fence
  records, PM samples, range metrics and PC samples; the reader states
  its contents and the Perfetto converter turns it into a trace. One
  session runs at a time.
- A session whose world is replaced ends with the world change, its file
  written.
- The quality-of-service guard turns collectors off, costliest first, and
  then ends a session whose step time stays past the slowdown it allows.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    worker_executable,
)
from uniserve_eval.profile import ProfileFile, convert

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

NAME = "qwen3-profiling"
OPTIONS = (
    "dtype=bfloat16",
    "kv_cache_config.max_tokens=32768",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
)
PROMPT = (
    "Count from 1 to 3000, separated by commas and spaces. "
    "Write only the numbers."
)


def _checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_QWEN3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_MODEL must name the model checkpoint")
    return checkpoint


def _stop_storage_server(socket: Path) -> None:
    """Stop the Storage Server serving `socket`, found by its command line."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in command and str(socket).encode() in command:
            os.kill(int(entry.name), signal.SIGTERM)


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A Qwen3 deployment serving with the default fence probes, which the
    module's tests share.
    """  # noqa: D205
    tmp_path = tmp_path_factory.mktemp("served")
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        _checkpoint(),
        "--served-model-name",
        NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
    ]
    for option in OPTIONS:
        command += ["--set", option]
    environment = {
        "CUDA_VISIBLE_DEVICES": os.environ.get(
            "UNISERVE_PROFILING_DEVICE", "0"
        ),
        "XDG_RUNTIME_DIR": str(runtime),
    }
    try:
        with server_process(
            command, base_url, tmp_path / "server.log", 900, environment
        ):
            yield base_url
    finally:
        _stop_storage_server(runtime / "uniserve" / "storage.sock")
        shutil.rmtree(runtime, ignore_errors=True)


def _chat(base_url: str, tokens: int = 32) -> None:
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": NAME,
            "messages": [{"role": "user", "content": "Say hello."}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_completion_tokens": tokens,
        },
        timeout=600,
    )
    response.raise_for_status()


def _stream(base_url: str, stop: threading.Event) -> None:
    """Stream counts until `stop` is set: decode steps throughout."""
    while not stop.is_set():
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": NAME,
                "messages": [{"role": "user", "content": PROMPT}],
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0,
                "max_completion_tokens": 2048,
                "stream": True,
            },
            timeout=1800,
        ) as response:
            for _ in response.iter_lines():
                if stop.is_set():
                    break


class _Load:
    """Requests streaming on threads of their own while the block runs."""

    def __init__(self, base_url: str, streams: int = 2) -> None:
        self.stop = threading.Event()
        self.threads = [
            threading.Thread(target=_stream, args=(base_url, self.stop))
            for _ in range(streams)
        ]

    def __enter__(self) -> _Load:
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=600)


def _fence_count(base_url: str, stage: str, component: str) -> int:
    """The steps the fence histogram counted for `stage` and `component`."""
    text = httpx.get(f"{base_url}/metrics", timeout=30).text
    pattern = re.compile(
        r"^uniserve:worker_fence_step_seconds_count\{([^}]*)\} (\S+)$",
        re.MULTILINE,
    )
    total = 0
    for labels, value in pattern.findall(text):
        if f'stage="{stage}"' in labels and component in labels:
            total += int(float(value))
    return total


def _profile(base_url: str) -> dict | None:
    response = httpx.get(f"{base_url}/admin/profile", timeout=30)
    response.raise_for_status()
    return response.json()["session"]


def _start(base_url: str, request: dict) -> dict:
    response = httpx.post(
        f"{base_url}/admin/profile", json=request, timeout=600
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def _ended(base_url: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        session = _profile(base_url)
        if session and session["state"] == "ended":
            return session
        time.sleep(1.0)
    raise TimeoutError(f"the session did not end within {timeout_s} s")


def _apply(base_url: str, *assignments: str) -> dict:
    response = httpx.put(
        f"{base_url}/admin/config",
        json={"set": list(assignments)},
        timeout=1800,
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def test_fence_probes_report_steps_and_switch_in_place(served: str):
    _chat(served)
    with _Load(served, streams=1):
        time.sleep(5)
    decode = _fence_count(served, "decode", 'component="step"')
    layers = _fence_count(served, "decode", "TransformerLayer")
    print(f"decode steps {decode}, layer steps {layers}")
    assert decode > 0 and layers > 0

    off = _apply(served, "profiling_config.fence_probes=false")
    assert off["change"] == "in_place", off
    # The steps whose probes ran before the switch are read with the next
    # results; none is counted after them.
    with _Load(served, streams=1):
        time.sleep(5)
    settled = _fence_count(served, "decode", 'component="step"')
    with _Load(served, streams=1):
        time.sleep(5)
    assert _fence_count(served, "decode", 'component="step"') == settled

    on = _apply(served, "profiling_config.fence_probes=true")
    assert on["change"] == "in_place", on
    with _Load(served, streams=1):
        time.sleep(5)
    assert _fence_count(served, "decode", 'component="step"') > settled


def test_a_session_records_every_collector_into_its_file(
    served: str, tmp_path: Path
):
    output = tmp_path / "profiles"
    request = {
        "collectors": [
            "timeline",
            "fence_timeline",
            "pm_sampling",
            "range_metrics",
            "pc_sampling",
        ],
        "stages": ["decode"],
        "start_step": 5,
        "steps": 40,
        "components": ["*.TransformerLayer"],
        "output_dir": str(output),
    }
    with _Load(served):
        session = _start(served, request)
        worker = session["workers"][0]
        assert worker["selected"], worker
        assert all(
            state == "prepared" for state in worker["collectors"].values()
        ), worker
        # One session at a time.
        refused = httpx.post(
            f"{served}/admin/profile", json=request, timeout=600
        )
        assert refused.status_code == 400, refused.text
        ended = _ended(served, 600)
    print(json.dumps(ended, indent=2))
    assert ended["end_reason"] == "window", ended
    assert ended["steps"] == 40, ended
    assert ended["bytes_written"] > 0, ended

    [path] = sorted(output.glob("*.sqlite"))
    with ProfileFile(path) as profile:
        summary = profile.summary()
        print(json.dumps(summary, indent=2))
        rows = summary["rows"]
        for table in (
            "graph",
            "api",
            "fence",
            "pm_sample",
            "range_metric",
            "pc_sample",
            "steps",
            "clocks",
        ):
            assert rows.get(table, 0) > 0, table
        steps = list(profile.rows("steps"))
        taken = [step for step in steps if step["selected"]]
        assert len(taken) == 40
        assert {step["stage"] for step in taken} == {"decode"}
        intervals = list(profile.fence_intervals())
        assert intervals
        assert {item["component"] for item in intervals} == {
            "models.qwen3.transformer.TransformerLayer"
        }
    trace = convert([path])
    names = {event.get("name") for event in trace["traceEvents"]}
    assert "models.qwen3.transformer.TransformerLayer" in names


def test_a_world_change_ends_the_session_of_the_world_it_replaces(
    served: str, tmp_path: Path
):
    with _Load(served, streams=1):
        _start(
            served,
            {
                "collectors": ["timeline"],
                "timeline": {"graphs": "nodes"},
                "output_dir": str(tmp_path / "profiles"),
            },
        )
        changed = _apply(
            served, "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8, 16]"
        )
        assert changed["change"] == "world", changed
        session = _profile(served)
        assert session["state"] == "ended", session
        assert session["end_reason"] == "world change", session
        assert session["files"][0]["file"]["rows"]["kernel"] > 0, session
        committed = httpx.post(
            f"{served}/admin/config/commit", json={}, timeout=600
        )
        assert committed.status_code == 200, committed.text


def test_the_guard_releases_collectors_then_ends_the_session(
    served: str, tmp_path: Path
):
    def actions() -> list[tuple[str, str | None]]:
        session = _profile(served) or {}
        return [
            (action["action"], action["collector"])
            for action in session.get("guard", [])
        ]

    with _Load(served):
        # The guard's baseline: the step time before the session.
        time.sleep(30)
        _start(
            served,
            {
                "collectors": ["timeline", "pc_sampling"],
                "timeline": {"graphs": "nodes"},
                "output_dir": str(tmp_path / "profiles"),
                "max_slowdown": 0.0,
            },
        )
        deadline = time.monotonic() + 300
        while len(actions()) < 2 and time.monotonic() < deadline:
            time.sleep(1.0)
        session = _profile(served)
        if session["state"] == "running":
            stopped = httpx.delete(f"{served}/admin/profile", timeout=600)
            assert stopped.status_code == 200, stopped.text
        ended = _ended(served, 60)
    print(json.dumps(ended, indent=2))
    taken = actions()
    # Costliest first; a slowdown left without a collector ends the
    # session, else the operator ended it.
    assert taken[:2] == [
        ("release", "pc_sampling"),
        ("release", "timeline"),
    ], taken
    assert ended["end_reason"] in ("quality of service", "operator"), ended
    assert (taken[2:] == [("end", None)]) == (
        ended["end_reason"] == "quality of service"
    ), taken
