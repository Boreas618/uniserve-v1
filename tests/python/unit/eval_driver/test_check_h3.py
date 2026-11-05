"""The fixed MiniMax-H3 check set covers every served cell with frozen requests.

The design confirms MiniMax-H3 by people watching generated results: a fixed
set of requests covering every combination the service offers, each with a
fixed seed, whose MP4s land in a fixed directory under ``artifacts/``. This
test holds ``uniserve_eval/check_h3.toml`` and its frozen manifests to that
contract: each cell exactly once, one measured row at seed 0 and one warmup
row at seed 42 copied unchanged from the performance workloads, condition
media and their anchors kept, and a fixed per-cell MP4 directory. The same six
cells also run reversed, which is the candidate's obligation, and their MP4s
go to a sibling directory so neither order overwrites the other. The only
automatic checks of a run stay the video task's MP4 validity and the
protected performance metrics; this test adds no check to the driver.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from uniserve_eval.config import load_config
from uniserve_eval.datasets import load_examples

pytestmark = pytest.mark.unit

# The profile ships beside the evaluator and runs from the repository root.
_PROFILE = (
    Path(__file__).resolve().parents[4] / "uniserve_eval" / "check_h3.toml"
)
_WORKLOADS = _PROFILE.parent / "workloads" / "minimax_h3"
_ARTIFACT_ROOT = Path("artifacts/minimax_h3/check/results")


@dataclass(frozen=True)
class _Cell:
    """One served combination and the frozen request that checks it.

    ``workload`` names the performance workload whose seed-0 measured row and
    seed-42 warmup row the cell's manifests copy: W1 (t2va), W3 (fl2va) or
    W4 (ref2va). ``directory`` is the fixed per-cell MP4 directory under
    ``artifacts/minimax_h3/check``.
    """

    checkpoint: str
    task: str
    workload: str
    directory: str


# The cells the service offers: FastH3 DMD serves t2va and fl2va, FastH3
# OmniRef serves ref2va, and base MiniMax-H3 serves t2va and fl2va through its
# transformer partition and ref2va through its transformer_ref partition. The
# order is the forward suite's: the reference cell first, which is how today's
# implementation is measured (it shares one Storage Server key between the two
# base servers on one checkpoint, so the cell that runs second inherits the
# first's pages). The reversed suite is the candidate's obligation, and its
# MP4s go to a sibling directory so neither order overwrites the other.
_CELLS = {
    "check-base-ref2va": _Cell(
        "MiniMaxAI/MiniMax-H3",
        "ref2va",
        "w4",
        "base-ref2va",
    ),
    "check-fasth3-dmd-t2va": _Cell(
        "FastVideo/FastVideo-FastH3-8-Step-V2",
        "t2va",
        "w1",
        "fasth3-dmd-t2va",
    ),
    "check-fasth3-dmd-fl2va": _Cell(
        "FastVideo/FastVideo-FastH3-8-Step-V2",
        "fl2va",
        "w3",
        "fasth3-dmd-fl2va",
    ),
    "check-fasth3-omniref-ref2va": _Cell(
        "FastVideo/FastVideo-FastH3-Omni-8-Step-V1",
        "ref2va",
        "w4",
        "fasth3-omniref-ref2va",
    ),
    "check-base-t2va": _Cell(
        "MiniMaxAI/MiniMax-H3",
        "t2va",
        "w1",
        "base-t2va",
    ),
    "check-base-fl2va": _Cell(
        "MiniMaxAI/MiniMax-H3",
        "fl2va",
        "w3",
        "base-fl2va",
    ),
}

#: The reversed suite's run order: the forward order with the reference cell
#: moved from first to last, so the cell that inherited the first's pages runs
#: last and a pass says the ordering requirement is gone.
_REVERSED_ORDER = [name for name in _CELLS if name != "check-base-ref2va"] + [
    "check-base-ref2va"
]
#: The reversed suite's cells, each the same request as its forward cell.
_REVERSED = {f"{name}-reversed": _CELLS[name] for name in _REVERSED_ORDER}


@pytest.fixture(scope="module")
def config():
    """Load the check-set profile once for every test of it."""
    return load_config(_PROFILE)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read a manifest's non-blank JSON Lines rows in file order."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _only_row(path: Path) -> dict[str, Any]:
    """Return the single row of a frozen check manifest."""
    rows = _jsonl_rows(path)
    assert len(rows) == 1
    return rows[0]


def _row_with_seed(path: Path, seed: int) -> dict[str, Any]:
    """Return the one row of a performance workload carrying that seed."""
    matches = [row for row in _jsonl_rows(path) if row["seed"] == seed]
    assert len(matches) == 1
    return matches[0]


def test_the_profile_states_each_cell_without_precision_flags() -> None:
    """The shipped TOML fixes the protocol, whatever the environment holds."""
    raw = tomllib.loads(_PROFILE.read_text(encoding="utf-8"))
    assert set(raw["servers"]) == {
        "fasth3-dmd",
        "fasth3-omniref",
        "base-transformer",
        "base-reference",
    }
    # The launch's default dtype is bfloat16 and an empty quantization loads
    # the first precision preset, quality, so stating either would not change
    # what the check set confirms, so a check cell names neither. No worker
    # executable is pinned either, so a deployment launched with --executable
    # runs the new build end to end.
    for name, server in raw["servers"].items():
        assert "--worker-executable" not in server["command"], name
        for value in server["command"]:
            assert not value.startswith("dtype="), name
            assert not value.startswith("quantization."), name
    assert list(raw["suites"]["check-h3"]["points"]) == list(_CELLS)
    assert list(raw["suites"]["check-h3-reversed"]["points"]) == [
        f"{name}-reversed" for name in _REVERSED_ORDER
    ]
    assert set(raw["benchmarks"]) == set(_CELLS) | set(_REVERSED)
    assert len(raw["benchmarks"]) == len(_CELLS) + len(_REVERSED)
    for name, cell in _CELLS.items():
        point = raw["benchmarks"][name]
        assert point["server"] in raw["servers"]
        # The point requests the name its server serves.
        command = raw["servers"][point["server"]]["command"]
        served = command[command.index("--served-model-name") + 1]
        assert served == point["model"], name
        assert point["video"]["task"] == cell.task
        # W1 targets 16:9; W3 and W4 resolve their canvas from the condition
        # media they are sent.
        assert point["video"]["aspect_ratio"] == (
            "16:9" if cell.task == "t2va" else "auto"
        )
        assert point["video"]["media_dir"] == (
            f"artifacts/minimax_h3/check/{cell.directory}"
        )
        assert point["load"]["num_prompts"] == 1
        if cell.task == "t2va":
            assert "condition_root" not in point["video"]
        else:
            assert point["video"]["condition_root"] == (
                "${UNISERVE_MINIMAX_H3_INPUTS}"
            )
        # The point reads a frozen check manifest, not a performance one.
        manifests = (
            point["dataset_path"],
            point["load"]["warmup_manifest"],
        )
        for manifest in manifests:
            assert manifest.startswith(
                "uniserve_eval/workloads/minimax_h3/check-"
            )
            assert (_PROFILE.parents[1] / manifest).is_file()
    # A reversed cell is its forward cell: same server, model, frozen rows and
    # task, and a sibling MP4 directory so neither order overwrites the other.
    for name, cell in _REVERSED.items():
        point = raw["benchmarks"][name]
        forward_name = name.removesuffix("-reversed")
        assert forward_name in _CELLS
        forward = raw["benchmarks"][forward_name]
        for key in ("server", "model", "dataset", "dataset_path", "task"):
            assert point[key] == forward[key], name
        assert (
            point["load"]["warmup_manifest"]
            == forward["load"]["warmup_manifest"]
        )
        assert point["video"]["task"] == cell.task
        assert (
            point["video"]["aspect_ratio"] == forward["video"]["aspect_ratio"]
        )
        assert point["video"]["media_dir"] == (
            f"artifacts/minimax_h3/check-reversed/{cell.directory}"
        )


def test_the_suite_covers_every_cell_exactly_once(config) -> None:
    """The profile declares the expected cells, each once, in run order."""
    assert set(config.benchmarks) == set(_CELLS) | set(_REVERSED)
    assert len(config.benchmarks) == len(_CELLS) + len(_REVERSED)
    assert config.selected_points("check-h3") == tuple(
        config.benchmarks[name] for name in _CELLS
    )
    assert config.selected_points("check-h3-reversed") == tuple(
        config.benchmarks[f"{name}-reversed"] for name in _REVERSED_ORDER
    )
    for name, cell in {**_CELLS, **_REVERSED}.items():
        point = config.benchmarks[name]
        assert point.video.task == cell.task
        assert point.provenance == {
            "checkpoint": cell.checkpoint,
            "task": cell.task,
        }


@pytest.mark.parametrize("name", _CELLS, ids=tuple(_CELLS))
def test_each_cell_measures_the_seed_zero_row_of_its_workload(
    config, name: str
) -> None:
    """The measured row is the workload's seed-0 row, unchanged."""
    cell = _CELLS[name]
    point = config.benchmarks[name]
    rows, tokenizer = load_examples(
        replace(
            point,
            dataset_path=str(config.root / point.dataset_path),
        )
    )
    assert tokenizer is None
    assert len(rows) == 1
    measured = rows[0].as_dict()
    assert measured["seed"] == 0
    source = _row_with_seed(_WORKLOADS / f"{cell.workload}.jsonl", seed=0)
    assert measured == source
    assert measured["metadata"]["task"] == cell.task
    assert measured["metadata"]["partition"] == "measured"
    # The condition media and their anchors survive the copy.
    if cell.task == "fl2va":
        (keyframe,) = measured["conditions"]
        assert keyframe["role"] == "keyframe"
        assert keyframe["frame_index"] == 0
    elif cell.task == "ref2va":
        assert {row["role"] for row in measured["conditions"]} == {"reference"}
    else:
        assert "conditions" not in measured


@pytest.mark.parametrize("name", _CELLS, ids=tuple(_CELLS))
def test_each_cell_warms_up_with_the_seed_42_row_of_its_workload(
    config, name: str
) -> None:
    """The unmeasured warmup row is the workload's seed-42 row, unchanged."""
    cell = _CELLS[name]
    point = config.benchmarks[name]
    assert point.load.warmup_manifest is not None
    warmup = _only_row(config.root / point.load.warmup_manifest)
    assert warmup["seed"] == 42
    source = _row_with_seed(
        _WORKLOADS / f"{cell.workload}-warmup.jsonl", seed=42
    )
    assert warmup == source
    assert warmup["metadata"]["task"] == cell.task
    assert warmup["metadata"]["partition"] == "warmup"


@pytest.mark.parametrize("name", _CELLS, ids=tuple(_CELLS))
def test_each_cell_writes_its_mp4s_to_a_fixed_directory(
    config, name: str
) -> None:
    """Every cell names the fixed MP4 directory a person watches."""
    cell = _CELLS[name]
    media_dir = config.benchmarks[name].video.media_dir
    assert media_dir == f"artifacts/minimax_h3/check/{cell.directory}"
    # The point's bundle goes to the profile's fixed artifact root.
    assert config.artifact_root == config.root / _ARTIFACT_ROOT


@pytest.mark.parametrize("name", _CELLS, ids=tuple(_CELLS))
def test_each_cell_measures_one_request_at_a_time(config, name: str) -> None:
    """One measured request at seed 0 follows its one seed-42 warmup."""
    point = config.benchmarks[name]
    assert point.load.num_prompts == 1
    assert point.load.max_concurrency == 1
    assert point.load.request_rate == float("inf")
    assert (config.root / point.dataset_path).is_file()
    assert point.dataset_path.startswith(
        "uniserve_eval/workloads/minimax_h3/check-"
    )


def test_every_cell_has_its_own_media_directory(config) -> None:
    """No two cells write MP4s to the same fixed directory."""
    names = list(_CELLS) + list(_REVERSED)
    directories = [config.benchmarks[name].video.media_dir for name in names]
    assert len(set(directories)) == len(names)
    assert all(directory is not None for directory in directories)
