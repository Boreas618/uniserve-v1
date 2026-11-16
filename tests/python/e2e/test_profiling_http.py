"""Online profiling of a serving deployment through the admin API.

Qwen3 serves on the GPU the module's lease holds:

- The fence probes are always on: every decode step's span and its
  transformer layers' time reach the Prometheus histogram, and switching
  the probes off through ``PUT /admin/config`` is a change in place that
  stops them, until they are switched on again.
- Sessions record their windows of decode steps into one file per
  worker, together running every collector: CUPTI's activity records and
  API calls with the fence records and PC sampling's counts (its records
  of graph kernels need ``profiling_config.graph_pc_sampling``), PM
  samples, and range metrics, each in its table of the file's schema
  (``crates/engine/src/profiling/schema.sql``). One session runs at a time,
  and one naming two collectors of the device's performance monitors is
  refused.
- A range_metrics session reserves its range profiler pass's device storage,
  which its worker measures and reports, with the worker's process item in
  the GPU's quota until it ends.
- A session whose epoch is replaced ends with the epoch change, its file
  written.
- The quality-of-service guard turns collectors off, costliest first, and
  then ends a session whose step time stays past the slowdown it allows.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import pytest
from uniserve_testing import admin, chat, client, profiling
from uniserve_testing.deployments import Deployments
from uniserve_testing.profiling import Load
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    # The module serves one deployment for all of its cases.
    pytest.mark.gpus(1),
]

#: The module classes the deployment probes, and the sessions' fence records
#: keep: its transformer layers, as a pattern of their class names.
LAYERS = "*.TransformerLayer"

OPTIONS = (
    "--set",
    "kv_cache_config.max_tokens=32768",
    "--set",
    f'profiling_config.fence_modules=["{LAYERS}"]',
)

#: How long the workers take to report the forwards or fence steps of the
#: load's decode steps, in seconds (scaled): each report carries the steps
#: since the last, once a second.
REPORTED_S = 60.0


@pytest.fixture(scope="module")
def served(module_deployments: Deployments) -> Iterator[Server]:
    """A Qwen3 deployment serving with fence probes on its transformer
    layers, which the module's tests share.
    """  # noqa: D205
    with profiling.serve(module_deployments, *OPTIONS) as server:
        yield server


def _apply(server: Server, *assignments: str) -> dict[str, Any]:
    """The result of applying `assignments` (``PUT /admin/config``)."""
    response = admin.apply(server, *assignments)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def _forwards(server: Server) -> float:
    """The forwards the workers reported so far, which their reports carry
    beside the fence histogram's steps
    (``uniserve:worker_forward_mode_counts``).
    """  # noqa: D205
    return client.metric(server.base_url, "uniserve:worker_forward_mode_counts")


def _forwards_reported(server: Server) -> None:
    """Return once the workers reported forwards they ran from now on.

    The counts reach ``/metrics`` with the workers' reports, which no event
    of the node's log announces, so they are read every second.
    """
    forwards = _forwards(server)
    server.wait_until(
        lambda: _forwards(server) > forwards,
        REPORTED_S,
        "the workers reported no forward",
    )


def test_fence_probes_report_steps_and_switch_in_place(served: Server):
    def steps() -> int:
        return profiling.fence_steps(served, "decode", "step")

    def layer_steps() -> int:
        return profiling.fence_steps(served, "decode", LAYERS)

    chat.complete(
        served.base_url,
        chat.body(
            profiling.NAME, "Say hello.", 32, temperature=0, **chat.NO_THINKING
        ),
    )
    with Load(served, streams=1):
        # The counts reach /metrics with the workers' reports, which no event
        # of the node's log announces.
        served.wait_until(
            lambda: steps() > 0 and layer_steps() > 0,
            REPORTED_S,
            "the fence probes reported no decode step",
        )
    print(f"decode steps {steps()}, layer steps {layer_steps()}")

    off = _apply(served, "profiling_config.fence_probes=false")
    assert off["change"] == "in_place", off
    try:
        # The steps whose probes ran before the switch are read with the next
        # results; none is counted after them, while the workers report the
        # forwards they run.
        with Load(served, streams=1):
            _forwards_reported(served)
            settled = steps()
            _forwards_reported(served)
            assert steps() == settled
    finally:
        # The module's server goes on to the sessions of the tests after this
        # one with its probes as it was started, whatever this one found.
        on = _apply(served, "profiling_config.fence_probes=true")
    assert on["change"] == "in_place", on
    with Load(served, streams=1):
        served.wait_until(
            lambda: steps() > settled,
            REPORTED_S,
            "the fence probes counted no step once switched on",
        )


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
    served: Server, tmp_path: Path, name: str
):
    collectors, tables = SESSIONS[name]
    output = tmp_path / "profiles"
    request = {
        "collectors": collectors,
        "stages": ["decode"],
        "start_step": 5,
        "steps": 40,
        "modules": [LAYERS],
        "output_dir": str(output),
    }
    # Two collectors of the performance monitors are refused together.
    conflicting = {**request, "collectors": ["pm_sampling", "pc_sampling"]}
    refused = admin.request(served, "POST", profiling.PROFILE, body=conflicting)
    assert refused.status_code == 400, refused.text
    assert "performance monitors" in refused.text, refused.text

    with Load(served):
        session = profiling.start(served, request)
        worker = session["workers"][0]
        assert worker["selected"], worker
        assert all(
            state == "prepared" for state in worker["collectors"].values()
        ), worker
        # One session at a time.
        refused = admin.request(served, "POST", profiling.PROFILE, body=request)
        assert refused.status_code == 400, refused.text
        ended = profiling.ended(served)
    print(json.dumps(ended, indent=2))
    assert ended["end_reason"] == "window", ended
    assert ended["steps"] == 40, ended
    assert ended["bytes_written"] > 0, ended

    [path] = sorted(output.glob("*.sqlite"))
    (version,) = profiling.table(path, "schema_version")
    assert version["version"] >= 2, version
    # Every collector ran to the session's end; none failed to start.
    states = {
        (row["name"], row["state"])
        for row in profiling.table(path, "collectors")
    }
    assert {state for _, state in states} == {"prepared", "ended"}, states
    assert {name for name, _ in states} == set(collectors), states
    for name in (*tables, "steps", "clocks"):
        assert profiling.table(path, name), name
    steps = profiling.table(path, "steps")
    taken = [step for step in steps if step["selected"]]
    assert len(taken) == 40
    assert {step["stage"] for step in taken} == {"decode"}
    if "fence_timeline" in collectors:
        # The fence records the session kept are the calls of the module
        # classes its request named: each probe's entry (an even id below
        # the step markers' 0xFFFFE) names the module class it fences.
        probes = {
            row["probe"]: row["module"]
            for row in profiling.table(path, "fence_probe")
        }
        modules = {
            probes[row["probe"] - row["probe"] % 2]
            for row in profiling.table(path, "fence")
            if row["probe"] < 0xFFFFE
        }
        assert modules, "the session kept no fence record"
        assert all(fnmatchcase(module, LAYERS) for module in modules), modules


def _process_bytes(server: Server) -> int:
    """The bytes the worker's process item reserves in its GPU's quota."""
    entries = [
        entry
        for entry in admin.reservations(server)
        if entry["item"] == "process"
    ]
    assert len(entries) == 1, entries
    return entries[0]["bytes"]


def test_a_range_metrics_session_books_its_pass_while_it_runs(
    served: Server, tmp_path: Path
):
    before = _process_bytes(served)
    session = profiling.start(
        served,
        {
            "collectors": ["range_metrics"],
            "stages": ["decode"],
            "output_dir": str(tmp_path / "profiles"),
        },
    )
    # The pass's device storage, as the worker measured it when the session
    # started.
    [worker] = session["workers"]
    held = worker["device_storage"]
    assert held > 0, worker
    assert _process_bytes(served) == before + held
    stopped = profiling.stop(served)
    assert stopped["state"] == "ended", stopped
    assert stopped["end_reason"] == "operator", stopped
    assert _process_bytes(served) == before


def test_a_epoch_change_ends_the_session_of_the_epoch_it_replaces(
    served: Server, tmp_path: Path
):
    with Load(served, streams=1):
        profiling.start(
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
        assert changed["change"] == "epoch", changed
        session = profiling.session(served)
        assert session["state"] == "ended", session
        assert session["end_reason"] == "epoch change", session
        assert session["files"][0]["file"]["rows"]["kernel"] > 0, session
        committed = admin.operation(served, "config/commit")
        assert committed.status_code == 200, committed.text


#: The span before a guarded session whose average step time its guard
#: compares the session's with (``baseline_s``), in seconds: the load decodes
#: alone for this span before the session starts.
BASELINE_S = 30

#: The span of the session the guard averages its step time over, and the
#: least time between two of its actions (``guard_window_s``), in seconds.
GUARD_WINDOW_S = 10

#: How long the guard takes to act twice, in seconds (scaled): it first acts
#: once a window of the session's steps is measured and again at least a
#: window later, so two actions take up to three windows; the bound allows
#: twice that.
GUARD_ACTIONS_S = 6 * GUARD_WINDOW_S


def test_the_guard_releases_collectors_then_ends_the_session(
    served: Server, tmp_path: Path
):
    def actions() -> list[tuple[str, str | None]]:
        current = profiling.session(served) or {}
        return [
            (action["action"], action["collector"])
            for action in current.get("guard", [])
        ]

    with Load(served) as load:
        # The guard's baseline is the step time over the span its request
        # states before the session, in which the load alone decodes: a span
        # of time by definition, not a state to wait for.
        load.decoding()
        time.sleep(BASELINE_S)
        profiling.start(
            served,
            {
                "collectors": ["timeline", "pc_sampling"],
                "timeline": {"graphs": "nodes"},
                "output_dir": str(tmp_path / "profiles"),
                "max_slowdown": 0.0,
                "baseline_s": BASELINE_S,
                "guard_window_s": GUARD_WINDOW_S,
            },
        )
        # The guard's actions are in the session's state, which no event of
        # the node's log announces.
        served.wait_until(
            lambda: (
                len(actions()) >= 2
                or (profiling.session(served) or {}).get("state") == "ended"
            ),
            GUARD_ACTIONS_S,
            "the guard did not act twice",
        )
        current = profiling.session(served)
        ended = (
            profiling.stop(served)
            if current["state"] == "running"
            else profiling.ended(served)
        )
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
