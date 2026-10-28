"""Online profiling of a serving deployment through the admin API.

Qwen3 serves on one GPU (``UNISERVE_WORLD_DEVICE``, default 0), with a
Storage Server of the test's own in a runtime directory the test creates
(``tempfile``'s directory, ``TMPDIR`` when set):

- The fence probes are always on: every decode step's span and its
  transformer layers' time reach the Prometheus histogram, and switching
  the probes off through ``PUT /admin/config`` is a change in place that
  stops them, until they are switched on again.
- Sessions record their windows of decode steps into one file per
  worker, together running every collector: CUPTI's activity records and
  API calls with the fence records and PC sampling's counts (its records
  of graph kernels need ``profiling_config.graph_pc_sampling``), PM
  samples, and range metrics; the reader states each file's contents and
  the Perfetto
  converter turns it into a trace. One session runs at a time, and one
  naming two collectors of the device's performance monitors is refused.
- A range_metrics session books its range profiler pass's device storage
  with the worker's process item in the GPU's ledger until it ends.
- A session whose world is replaced ends with the world change, its file
  written.
- The quality-of-service guard turns collectors off, costliest first, and
  then ends a session whose step time stays past the slowdown it allows.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.profiling_helpers import NAME, RANGE_PASS_STORAGE, serve
from tests.python.e2e.profiling_helpers import Load as _Load
from tests.python.e2e.profiling_helpers import ended as _ended
from tests.python.e2e.profiling_helpers import profile as _profile
from tests.python.e2e.profiling_helpers import start as _start
from uniserve_eval.profile import ProfileFile, convert

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

OPTIONS = (
    "dtype=bfloat16",
    "kv_cache_config.max_tokens=32768",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
    'profiling_config.fence_components=["*.TransformerLayer"]',
)


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A Qwen3 deployment serving with fence probes on its transformer
    layers, which the module's tests share.
    """  # noqa: D205
    with serve(tmp_path_factory.mktemp("served"), OPTIONS) as base_url:
        yield base_url


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


# Sessions that together run every collector, each holding at most one of
# the collectors of the device's performance monitors, and the tables each
# leaves rows in besides its steps and clocks.
SESSIONS = {
    # Without profiling_config.graph_pc_sampling, PC sampling attributes only
    # kernels launched outside CUDA graphs; its takes count every sample
    # (test_graph_pc_sampling_http covers the graphs' kernels).
    "timeline": (
        ["timeline", "fence_timeline", "pc_sampling"],
        ("graph", "api", "fence", "pc_take"),
    ),
    "pm_sampling": (["pm_sampling"], ("pm_sample",)),
    "range_metrics": (["range_metrics"], ("range_metric",)),
}


@pytest.mark.parametrize("name", sorted(SESSIONS))
def test_a_session_records_its_collectors_into_its_file(
    served: str, tmp_path: Path, name: str
):
    collectors, tables = SESSIONS[name]
    output = tmp_path / "profiles"
    request = {
        "collectors": collectors,
        "stages": ["decode"],
        "start_step": 5,
        "steps": 40,
        "components": ["*.TransformerLayer"],
        "output_dir": str(output),
    }
    # Two collectors of the performance monitors are refused together.
    conflicting = {**request, "collectors": ["pm_sampling", "pc_sampling"]}
    refused = httpx.post(
        f"{served}/admin/profile", json=conflicting, timeout=600
    )
    assert refused.status_code == 400, refused.text
    assert "performance monitors" in refused.text, refused.text

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
        # Every collector ran to the session's end; none failed to start.
        states = {
            (row["name"], row["state"]) for row in profile.rows("collectors")
        }
        assert {state for _, state in states} == {"prepared", "ended"}, states
        assert {name for name, _ in states} == set(collectors), states
        rows = summary["rows"]
        for table in (*tables, "steps", "clocks"):
            assert rows.get(table, 0) > 0, table
        steps = list(profile.rows("steps"))
        taken = [step for step in steps if step["selected"]]
        assert len(taken) == 40
        assert {step["stage"] for step in taken} == {"decode"}
        if "fence_timeline" in collectors:
            intervals = list(profile.fence_intervals())
            assert intervals
            assert {item["component"] for item in intervals} == {
                "models.qwen3.transformer.TransformerLayer"
            }
    if "fence_timeline" in collectors:
        trace = convert([path])
        names = {event.get("name") for event in trace["traceEvents"]}
        assert "models.qwen3.transformer.TransformerLayer" in names


def _process_bytes(base_url: str) -> int:
    """The bytes the worker's process item books in its GPU's ledger."""
    status = httpx.get(f"{base_url}/admin/config", timeout=60).json()
    entries = [
        entry
        for node in status["ledgers"]
        for ledger in node["ledgers"]
        for entry in ledger["entries"]
        if entry["item"] == "process"
    ]
    assert len(entries) == 1, entries
    return entries[0]["bytes"]


def test_a_range_metrics_session_books_its_pass_while_it_runs(
    served: str, tmp_path: Path
):
    before = _process_bytes(served)
    _start(
        served,
        {
            "collectors": ["range_metrics"],
            "stages": ["decode"],
            "output_dir": str(tmp_path / "profiles"),
        },
    )
    assert _process_bytes(served) == before + RANGE_PASS_STORAGE
    stopped = httpx.delete(f"{served}/admin/profile", timeout=600)
    assert stopped.status_code == 200, stopped.text
    assert _ended(served, 60)["end_reason"] == "operator"
    assert _process_bytes(served) == before


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
