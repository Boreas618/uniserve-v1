"""Compares two GSM8K result bundles under the quality confirmation rule.

The design confirms Qwen3-MoE quality by running ``uniserve-eval`` twice on
the same checkpoint, sample set and sampling settings -- once against the
reference implementation and once against the candidate -- and accepting the
candidate when the two accuracies' difference lies within the 95% confidence
interval of the difference of two binomial proportions at that sample size.
``compare_runs`` enforces the premises before computing the interval: both
bundles must be completed, valid GSM8K runs over the same selected rows with
the same sampling settings, served model and benchmark cell (the point's
provenance), and both must have scored every selected row. A bundle that
fails a premise is not comparable, so the command reports the mismatch
instead of a verdict.

The comparison itself is :func:`uniserve_eval.accuracy.difference_interval`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..accuracy import DifferenceInterval, Proportion, difference_interval

# The task every compared bundle must have run.
GSM8K_TASK = "gsm8k"


@dataclass(frozen=True)
class Bundle:
    """A completed result bundle and the records the comparison reads.

    Attributes:
        directory: The bundle's directory, as the caller named it.
        summary: The parsed ``summary.json``.
        run: The parsed ``run.json`` lifecycle record.
    """

    directory: Path
    summary: dict[str, Any]
    run: dict[str, Any]

    @property
    def launch(self) -> dict[str, Any]:
        """Return the deployment provenance recorded in the summary."""
        launch = self.summary.get("launch")
        return launch if isinstance(launch, dict) else {}


def read_bundle(directory: str | Path) -> Bundle:
    """Read one result bundle's summary and lifecycle records.

    Args:
        directory: The bundle directory that ``uniserve-eval run`` wrote.

    Returns:
        The parsed bundle.

    Raises:
        FileNotFoundError: If either record is missing.
        ValueError: If either record is not a JSON object.
    """
    path = Path(directory)
    summary = _read_object(path / "summary.json")
    run = _read_object(path / "run.json")
    return Bundle(directory=path, summary=summary, run=run)


def compare_runs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare a candidate bundle against a reference bundle.

    Args:
        reference_dir: The bundle of the run accepted as the baseline.
        candidate_dir: The bundle of the run under confirmation.
        confidence: Two-sided confidence level of the difference interval.

    Returns:
        A JSON-compatible report with both runs' accuracies and provenance,
        the difference, its standard error, the interval and the verdict.

    Raises:
        FileNotFoundError: If either bundle's records are missing.
        ValueError: If a bundle is not a completed, valid GSM8K run, or the
            two runs do not share the same sample set, sampling settings,
            served model or benchmark cell.
    """
    reference = read_bundle(reference_dir)
    candidate = read_bundle(candidate_dir)
    _require_comparable(reference, candidate)

    interval = difference_interval(
        _proportion(reference),
        _proportion(candidate),
        confidence=confidence,
    )
    return {
        "confidence": confidence,
        "method": (
            "two-proportion difference, Wald normal interval at the sample "
            "size of the runs"
        ),
        "reference": _describe(reference),
        "candidate": _describe(candidate),
        "difference": _describe_interval(interval),
        "verdict": "accepted" if interval.accepted else "rejected",
    }


def _describe(bundle: Bundle) -> dict[str, Any]:
    """Return one run's accuracy, workload identity and provenance."""
    workload = _mapping(bundle.summary.get("workload"), "workload")
    proportion = _proportion(bundle)
    launch = bundle.launch
    return {
        "directory": str(bundle.directory),
        "model": workload.get("model"),
        "provenance": workload.get("provenance"),
        "accuracy": proportion.value,
        "correct": proportion.correct,
        "total": proportion.total,
        "selected_rows": bundle.summary.get("selected_rows"),
        "command": launch.get("command"),
        "git_head": launch.get("git_head"),
        "dirty": launch.get("dirty"),
    }


def _describe_interval(interval: DifferenceInterval) -> dict[str, Any]:
    """Return the statistical result in JSON-compatible form."""
    return {
        "value": interval.difference,
        "standard_error": interval.standard_error,
        "half_width": interval.half_width,
        "interval": [interval.low, interval.high],
        "within_interval": interval.accepted,
    }


def _proportion(bundle: Bundle) -> Proportion:
    """Return the bundle's scored outcome counts.

    Raises:
        ValueError: If the summary does not carry integer ``correct`` and
            ``total`` metrics.
    """
    metrics = _mapping(bundle.summary.get("metrics"), "metrics")
    correct = metrics.get("correct")
    total = metrics.get("total")
    if isinstance(correct, bool) or not isinstance(correct, int):
        raise ValueError(f"{bundle.directory}: summary.metrics.correct missing")
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError(f"{bundle.directory}: summary.metrics.total missing")
    return Proportion(correct, total)


def _require_comparable(reference: Bundle, candidate: Bundle) -> None:
    """Reject runs that do not share the comparison's premises.

    Raises:
        ValueError: With the first premise that fails.
    """
    for role, bundle in (("reference", reference), ("candidate", candidate)):
        if bundle.run.get("status") != "completed":
            raise ValueError(
                f"{role} run {bundle.directory} is not completed: "
                f"{bundle.run.get('status')!r}"
            )
        if bundle.summary.get("task") != GSM8K_TASK:
            raise ValueError(
                f"{role} run {bundle.directory} is not a {GSM8K_TASK} run"
            )
        if bundle.summary.get("dataset") != GSM8K_TASK:
            raise ValueError(
                f"{role} run {bundle.directory} did not score {GSM8K_TASK}"
            )
        validation = _mapping(bundle.summary.get("validation"), "validation")
        if validation.get("valid") is not True:
            raise ValueError(
                f"{role} run {bundle.directory} is not a valid run"
            )

    reference_rows = reference.summary.get("selected_rows")
    candidate_rows = candidate.summary.get("selected_rows")
    if reference_rows != candidate_rows:
        raise ValueError(
            "the runs scored different sample sets: "
            f"{reference_rows} and {candidate_rows}"
        )
    if not isinstance(reference_rows, dict):
        raise ValueError("the runs recorded no selected sample set")
    count = reference_rows.get("count")
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("the selected sample set has no row count")

    reference_workload = _mapping(reference.summary.get("workload"), "workload")
    candidate_workload = _mapping(candidate.summary.get("workload"), "workload")
    for key in ("sampling", "provenance", "model"):
        if reference_workload.get(key) != candidate_workload.get(key):
            raise ValueError(
                f"the runs differ in workload.{key}: "
                f"{reference_workload.get(key)} and "
                f"{candidate_workload.get(key)}"
            )
    for role, workload in (
        ("reference", reference_workload),
        ("candidate", candidate_workload),
    ):
        load = _mapping(workload.get("load"), "load")
        if load.get("num_prompts") != count:
            raise ValueError(
                f"{role} run scores {load.get('num_prompts')} rows but "
                f"records {count} selected rows"
            )

    for role, bundle in (("reference", reference), ("candidate", candidate)):
        if _proportion(bundle).total != count:
            raise ValueError(
                f"{role} run scored {_proportion(bundle).total} rows but "
                f"records {count} selected rows"
            )

    reference_model = _model_argument(reference)
    candidate_model = _model_argument(candidate)
    if reference_model is not None and candidate_model is not None:
        if reference_model != candidate_model:
            raise ValueError(
                "the runs served different checkpoints: "
                f"{reference_model} and {candidate_model}"
            )


def _model_argument(bundle: Bundle) -> str | None:
    """Return the checkpoint path a bundle's daemon command served.

    The daemon command is ``uniserve serve [MODEL] [OPTIONS]``, so the model
    is the first positional value after the ``serve`` subcommand. A command
    recorded without one yields ``None``.
    """
    command = bundle.launch.get("command")
    if not isinstance(command, list) or not command:
        return None
    values = [str(item) for item in command]
    try:
        index = values.index("serve")
    except ValueError:
        return None
    for value in values[index + 1 :]:
        if value.startswith("-"):
            return None
        return value
    return None


def _read_object(path: Path) -> dict[str, Any]:
    """Read one JSON object of a bundle.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not hold a JSON object.
    """
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Return a required JSON object field.

    Raises:
        ValueError: If the value is not a mapping.
    """
    if not isinstance(value, dict):
        raise ValueError(f"the bundle has no {name} object")
    return value
