"""The fixed MiniMax-H3 check set covers every served cell with frozen requests.

The design confirms MiniMax-H3 by people watching generated results: a fixed
set of requests covering every combination the service offers, each with a
fixed seed, whose MP4s land in a fixed directory under ``artifacts/``. This
test holds the evaluator's shipped ``check_h3.toml`` and its frozen manifests
to that rule: each cell (a checkpoint and a task) exactly once, one measured
row at seed 0 and one warmup row at seed 42 copied unchanged from a
performance workload and its warmup, reference media and their anchors kept,
and a fixed MP4 directory of its own. The same cells also run reversed,
which is the candidate's obligation, and their MP4s go to a sibling
directory so neither order overwrites the other. The only automatic checks
of a run stay the video task's MP4 validity and the protected performance
metrics; this test adds no check to the driver.

The test reads the cells from the profile itself and asserts the properties
every cell must have, so a cell added to or removed from the profile is held
to the same rule.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import pytest

from uniserve_eval.config import EvaluationConfig, load_config
from uniserve_eval.datasets import load_examples
from uniserve_eval.types import BenchmarkPoint

pytestmark = pytest.mark.unit

#: The forward suite, in which every cell runs once, and its reversed twin.
_FORWARD = "check-h3"
_REVERSED = "check-h3-reversed"

#: The seeds of a cell's measured and warmup rows.
_MEASURED_SEED = 0
_WARMUP_SEED = 42


@pytest.fixture(scope="module")
def config(shipped_profile: Callable[[str], Path]) -> EvaluationConfig:
    """Load the check-set profile once for every test of it."""
    return load_config(shipped_profile("check_h3.toml"))


@pytest.fixture(scope="module")
def forward(config: EvaluationConfig) -> tuple[BenchmarkPoint, ...]:
    """The forward suite's cells, in run order."""
    return config.selected_points(_FORWARD)


def _jsonl_rows(path: Traversable) -> list[dict[str, Any]]:
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


def _performance_workloads() -> dict[str, list[dict[str, Any]]]:
    """Every MiniMax-H3 performance workload the evaluator ships, by name.

    The check set's own frozen manifests (``check-*``) are not among them.
    """
    directory = files("uniserve_eval") / "workloads" / "minimax_h3"
    workloads = {}
    for resource in directory.iterdir():
        name = resource.name.removesuffix(".jsonl")
        if resource.name.endswith(".jsonl") and not name.startswith("check-"):
            workloads[name] = _jsonl_rows(resource)
    return workloads


def test_the_suites_run_each_cell_once_and_reversed(
    config: EvaluationConfig, forward: tuple[BenchmarkPoint, ...]
) -> None:
    """Every point is a forward cell or its reversed twin, each once.

    The reversed suite runs the forward order with its first cell, the
    reference cell, moved to the end, so the cell that inherited the first's
    pages runs last and a pass says the ordering requirement is gone.
    """
    names = [point.name for point in forward]
    reversed_names = [f"{name}-reversed" for name in names[1:] + names[:1]]
    assert [p.name for p in config.selected_points(_REVERSED)] == (
        reversed_names
    )
    assert sorted(config.benchmarks) == sorted(names + reversed_names)

    # A cell is a checkpoint serving a task; no two points check one cell.
    cells = [
        (point.provenance["checkpoint"], point.provenance["task"])
        for point in forward
    ]
    assert len(set(cells)) == len(cells)


def test_each_cell_requests_its_servers_model_at_its_default_precision(
    config: EvaluationConfig, forward: tuple[BenchmarkPoint, ...]
) -> None:
    """A point names what its server serves and states no precision.

    The launch's default dtype is bfloat16 and an empty quantization loads
    the first precision preset, quality, so stating either would not change
    what the check set confirms, and a check cell names neither.
    """
    for point in forward:
        command = list(config.servers[point.server].command)
        served = command[command.index("--served-model-name") + 1]
        assert served == point.model, point.name
    for name, server in config.servers.items():
        for value in server.command:
            assert not value.startswith("dtype="), name
            assert not value.startswith("quantization."), name


def test_every_cell_states_one_output_format(
    forward: tuple[BenchmarkPoint, ...],
) -> None:
    """Every cell asks for the same canvas, frame rate and soundtrack."""
    formats = {
        (
            point.video.size,
            point.video.fps,
            point.video.audio_sample_rate,
            point.video.audio_channels,
        )
        for point in forward
    }
    assert len(formats) == 1


def test_each_cell_measures_and_warms_up_with_workload_rows(
    forward: tuple[BenchmarkPoint, ...],
) -> None:
    """The measured row is a workload's seed-0 row, the warmup its seed-42.

    Both are copied unchanged into the cell's frozen check manifests: the
    measured row from a performance workload, the warmup row from that
    workload's warmup manifest.
    """
    workloads = _performance_workloads()
    for point in forward:
        _assert_frozen_rows(point, workloads)


def _assert_frozen_rows(
    point: BenchmarkPoint, workloads: dict[str, list[dict[str, Any]]]
) -> None:
    """Hold one cell's frozen manifests to the workloads they copy."""
    task = point.provenance["task"]
    rows, tokenizer = load_examples(point)
    assert tokenizer is None
    assert len(rows) == 1
    measured = rows[0].as_dict()
    assert Path(point.dataset_path).name.startswith("check-")
    assert point.load.warmup_manifest is not None
    assert Path(point.load.warmup_manifest).name.startswith("check-")
    warmup = _only_row(Path(point.load.warmup_manifest))

    assert measured["seed"] == _MEASURED_SEED
    assert measured["metadata"]["task"] == task
    assert measured["metadata"]["partition"] == "measured"
    assert warmup["seed"] == _WARMUP_SEED
    assert warmup["metadata"]["task"] == task
    assert warmup["metadata"]["partition"] == "warmup"

    sources = [
        name
        for name, workload in workloads.items()
        if measured in workload and f"{name}-warmup" in workloads
    ]
    assert sources, f"no performance workload holds {point.name}'s row"
    assert any(warmup in workloads[f"{name}-warmup"] for name in sources)

    # The reference media and their anchors survive the copy: fl2va's one
    # first-frame keyframe, ref2va's global references without an anchor.
    if task == "fl2va":
        (keyframe,) = measured["input_references"]
        assert keyframe["type"] == "image"
        assert keyframe["frame_index"] == 0
    elif task == "ref2va":
        assert measured["input_references"]
        assert all(
            "frame_index" not in row for row in measured["input_references"]
        )
    else:
        assert "input_references" not in measured

    # A cell reads reference media only when its rows name some.
    assert (point.video.reference_root is None) == (
        "input_references" not in measured
    )


def test_each_cell_measures_one_request_at_a_time(
    forward: tuple[BenchmarkPoint, ...],
) -> None:
    """One measured request at seed 0 follows its one seed-42 warmup."""
    for point in forward:
        assert point.load.num_prompts == 1
        assert point.load.max_concurrency == 1
        assert point.load.request_rate == float("inf")


def test_every_cell_writes_its_mp4s_to_a_directory_of_its_own(
    config: EvaluationConfig, forward: tuple[BenchmarkPoint, ...]
) -> None:
    """No two points share an MP4 directory; a reversed cell's is a sibling.

    A reversed cell is its forward cell (the same server, model, frozen rows,
    task and video settings) writing to a directory of the same name beside
    the forward cell's, so neither order overwrites the other.
    """
    directories = [
        point.video.media_dir for point in config.benchmarks.values()
    ]
    assert all(directory is not None for directory in directories)
    assert len(set(directories)) == len(directories)

    for point in forward:
        twin = config.benchmarks[f"{point.name}-reversed"]
        for key in ("server", "model", "dataset", "dataset_path", "task"):
            assert getattr(twin, key) == getattr(point, key), point.name
        assert twin.load.warmup_manifest == point.load.warmup_manifest
        assert twin.provenance == point.provenance
        assert replace(twin.video, media_dir=None) == replace(
            point.video, media_dir=None
        )

        assert point.video.media_dir is not None
        assert twin.video.media_dir is not None
        directory, twin_directory = (
            Path(point.video.media_dir),
            Path(twin.video.media_dir),
        )
        assert twin_directory.name == directory.name
        assert twin_directory.parent.parent == directory.parent.parent
        assert twin_directory.parent != directory.parent
