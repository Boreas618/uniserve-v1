"""Reading a profiling session's file and converting it to a trace."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from uniserve_eval.profile import ProfileError, ProfileFile, convert
from uniserve_eval.profile.reader import STEP_BEGIN, STEP_END

pytestmark = pytest.mark.unit

# The tables of schema version 1 this test writes (`crates/worker/src/
# profiling/session/artifact.rs` states the whole schema).
_TABLES = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE strings (id INTEGER PRIMARY KEY, text TEXT NOT NULL);
CREATE TABLE collectors (name TEXT NOT NULL, state TEXT NOT NULL,
    time INTEGER NOT NULL, detail TEXT NOT NULL);
CREATE TABLE clocks (source TEXT NOT NULL, device INTEGER, gpu INTEGER,
    realtime INTEGER NOT NULL, cupti INTEGER, monotonic INTEGER,
    uncertainty INTEGER NOT NULL);
CREATE TABLE kernel (start INTEGER NOT NULL, end INTEGER NOT NULL,
    device INTEGER, context INTEGER, stream INTEGER, correlation INTEGER,
    grid_x INTEGER, grid_y INTEGER, grid_z INTEGER, block_x INTEGER,
    block_y INTEGER, block_z INTEGER, name INTEGER, graph INTEGER,
    graph_node INTEGER);
CREATE TABLE api (start INTEGER NOT NULL, end INTEGER NOT NULL,
    domain INTEGER, cbid INTEGER, name INTEGER, thread INTEGER,
    correlation INTEGER, result INTEGER);
CREATE TABLE fence_probe (probe INTEGER PRIMARY KEY, instance INTEGER,
    site TEXT, path TEXT, component TEXT);
CREATE TABLE fence (device INTEGER, time INTEGER, probe INTEGER,
    step INTEGER);
"""


def _file(path: Path, version: int = 1) -> Path:
    db = sqlite3.connect(path)
    db.executescript(_TABLES)
    db.execute("INSERT INTO schema_version VALUES (?)", (version,))
    db.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [("session", "3"), ("world", "1"), ("rank", "2"), ("device", "0")],
    )
    db.executemany(
        "INSERT INTO strings VALUES (?, ?)",
        [(1, "gemm"), (2, "cuLaunchKernel")],
    )
    db.execute("INSERT INTO collectors VALUES ('timeline', 'prepared', 0, '')")
    # The device's %globaltimer is 1000 ns behind CUPTI's clock: the less
    # certain sample says otherwise and is not taken.
    db.executemany(
        "INSERT INTO clocks VALUES ('globaltimer', 0, ?, 0, ?, NULL, ?)",
        [(5_000, 6_000, 40), (5_000, 9_000, 4_000)],
    )
    db.execute(
        "INSERT INTO kernel VALUES (11000, 12000, 0, 1, 7, 9, 1, 1, 1, "
        "128, 1, 1, 1, 0, 0)"
    )
    db.execute("INSERT INTO api VALUES (10500, 10600, 1, 307, 2, 42, 9, 0)")
    db.executemany(
        "INSERT INTO fence_probe VALUES (?, ?, ?, ?, ?)",
        [
            (0, 0, "entry", "layers.0", "models.m.TransformerLayer"),
            (1, 0, "exit", "layers.0", "models.m.TransformerLayer"),
            (2, 1, "entry", "layers.0.attention", "models.m.Attention"),
            (3, 1, "exit", "layers.0.attention", "models.m.Attention"),
        ],
    )
    # Step 5: its markers around the layer, which holds the attention.
    db.executemany(
        "INSERT INTO fence VALUES (0, ?, ?, 5)",
        [
            (9_900, STEP_BEGIN),
            (10_000, 0),
            (10_100, 2),
            (10_700, 3),
            (11_000, 1),
            (11_100, STEP_END),
        ],
    )
    db.commit()
    db.close()
    return path


def test_a_file_names_its_session_and_pairs_its_fence_records(tmp_path):
    with ProfileFile(_file(tmp_path / "rank2.sqlite")) as profile:
        assert (profile.session, profile.rank) == (3, 2)
        intervals = sorted(
            (item["component"], item["start"], item["end"])
            for item in profile.fence_intervals()
        )
        assert intervals == [
            ("models.m.Attention", 10_100, 10_700),
            ("models.m.TransformerLayer", 10_000, 11_000),
        ]
        # The least uncertain sample aligns the device's clock.
        assert profile.to_cupti(10_000, "globaltimer", 0) == 11_000
        summary = profile.summary()
        assert summary["rows"]["kernel"] == 1
        assert summary["collectors"][0]["name"] == "timeline"


def test_a_file_of_an_unknown_version_or_none_is_refused(tmp_path):
    with pytest.raises(ProfileError):
        ProfileFile(_file(tmp_path / "old.sqlite", version=0))
    other = tmp_path / "other.sqlite"
    sqlite3.connect(other).execute("CREATE TABLE t (x)").connection.close()
    with pytest.raises(ProfileError):
        ProfileFile(other)


def test_the_trace_lines_up_device_and_probe_records(tmp_path):
    trace = convert([_file(tmp_path / "rank2.sqlite")])
    json.dumps(trace)
    slices = {
        event["name"]: event
        for event in trace["traceEvents"]
        if event.get("ph") == "X"
    }
    assert set(slices) >= {
        "gemm",
        "cuLaunchKernel",
        "models.m.TransformerLayer",
        "models.m.Attention",
        "step 5",
    }
    # The layer's probes, moved onto CUPTI's clock, span exactly the
    # kernel's interval (the trace's microseconds).
    layer, kernel = slices["models.m.TransformerLayer"], slices["gemm"]
    assert (layer["ts"], layer["dur"]) == (kernel["ts"], kernel["dur"])
    assert all(event["pid"] == 2 for event in trace["traceEvents"])
