"""Tests the accuracy comparison rule and the GSM8K bundle comparison.

The expected values come from the protocol in the swarm map: the interval of
the difference of two binomial proportions at the runs' sample sizes, and the
premises a pair of GSM8K runs must share to be comparable.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import pytest

from uniserve_eval.accuracy import Proportion, difference_interval
from uniserve_eval.cli import main
from uniserve_eval.pipeline.compare import compare_runs, read_bundle

pytestmark = pytest.mark.unit


def _z(confidence: float) -> float:
    """Return the two-sided normal quantile of a confidence level."""
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def test_interval_is_the_wald_interval_of_the_difference() -> None:
    """The reported interval equals the formula at the runs' sizes."""
    reference = Proportion(1186, 1319)
    candidate = Proportion(1190, 1319)

    interval = difference_interval(reference, candidate)

    p1 = reference.value
    p2 = candidate.value
    expected_se = math.sqrt(p1 * (1.0 - p1) / 1319 + p2 * (1.0 - p2) / 1319)
    assert interval.difference == pytest.approx(p2 - p1)
    assert interval.standard_error == pytest.approx(expected_se)
    assert interval.half_width == pytest.approx(_z(0.95) * expected_se)
    assert interval.low == pytest.approx(p2 - p1 - _z(0.95) * expected_se)
    assert interval.high == pytest.approx(p2 - p1 + _z(0.95) * expected_se)
    assert interval.accepted


def test_a_difference_beyond_the_half_width_is_rejected() -> None:
    """An accuracy gap wider than the interval's half-width is rejected."""
    interval = difference_interval(
        Proportion(1100, 1319), Proportion(1250, 1319)
    )

    assert interval.difference == pytest.approx(150 / 1319)
    assert abs(interval.difference) > interval.half_width
    assert not interval.accepted


def test_degenerate_samples_only_accept_an_equal_accuracy() -> None:
    """A sample with no spread has a zero-width interval."""
    interval = difference_interval(Proportion(0, 10), Proportion(0, 10))

    assert interval.standard_error == 0.0
    assert interval.half_width == 0.0
    assert interval.accepted
    # Both samples are at a bound, so the interval is still zero-width and
    # only an equal accuracy is accepted.
    assert not difference_interval(
        Proportion(0, 10), Proportion(10, 10)
    ).accepted


def test_a_lower_confidence_level_narrows_the_interval() -> None:
    """The confidence level selects the normal quantile."""
    reference = Proportion(1186, 1319)
    candidate = Proportion(1190, 1319)

    wide = difference_interval(reference, candidate, confidence=0.95)
    narrow = difference_interval(reference, candidate, confidence=0.90)

    assert narrow.half_width < wide.half_width
    assert narrow.half_width == pytest.approx(_z(0.90) * narrow.standard_error)


def test_invalid_inputs_are_rejected() -> None:
    """Counts outside the sample and confidence outside (0, 1) fail."""
    with pytest.raises(ValueError):
        Proportion(11, 10)
    with pytest.raises(ValueError):
        Proportion(-1, 10)
    with pytest.raises(ValueError):
        Proportion(0, 0)
    with pytest.raises(ValueError):
        difference_interval(
            Proportion(1, 10), Proportion(2, 10), confidence=1.0
        )


class _BundleWriter:
    """Writes a minimal completed GSM8K result bundle for comparison tests."""

    def __init__(self, directory: Path) -> None:
        """Bind the writer to the bundle directory it will create."""
        self.directory = directory

    def write(
        self,
        *,
        correct: int = 1186,
        total: int = 1319,
        selected_count: int | None = None,
        row_sha256: str = "row-digest",
        sampling: dict[str, object] | None = None,
        provenance: dict[str, object] | None = None,
        model: str = "Qwen3-30B-A3B",
        command: list[str] | None = None,
        valid: bool = True,
        status: str = "completed",
        task: str = "gsm8k",
        dataset: str = "gsm8k",
        metrics: dict[str, object] | None = None,
    ) -> Path:
        """Write ``summary.json`` and ``run.json`` and return the directory."""
        self.directory.mkdir(parents=True, exist_ok=True)
        count = total if selected_count is None else selected_count
        summary = {
            "benchmark": "qwen3-moe-gsm8k-bf16",
            "task": task,
            "dataset": dataset,
            "metrics": (
                metrics
                if metrics is not None
                else {
                    "accuracy": correct / total,
                    "correct": correct,
                    "total": total,
                }
            ),
            "selected_rows": {"count": count, "sha256": row_sha256},
            "workload": {
                "model": model,
                "sampling": sampling or {"temperature": 0.0, "seed": 42},
                "provenance": provenance
                or {"checkpoint": "Qwen3-30B-A3B", "precision": "bfloat16"},
                "load": {"num_prompts": count},
            },
            "validation": {"valid": valid, "checks": {}},
            "launch": {
                "command": command
                or [
                    "/repo/.venv/bin/uniserve",
                    "serve",
                    "/models/Qwen3-30B-A3B",
                ]
            },
        }
        (self.directory / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        (self.directory / "run.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )
        return self.directory


def test_a_matching_candidate_is_accepted(tmp_path: Path) -> None:
    """Runs sharing every premise compare to a verdict with both accuracies."""
    reference = _BundleWriter(tmp_path / "reference").write(correct=1186)
    candidate = _BundleWriter(tmp_path / "candidate").write(correct=1190)

    report = compare_runs(reference, candidate)

    assert report["verdict"] == "accepted"
    assert report["reference"]["correct"] == 1186
    assert report["candidate"]["correct"] == 1190
    assert report["difference"]["within_interval"] is True
    assert report["difference"]["interval"][0] < 0
    assert report["difference"]["interval"][1] > 0
    # The report is written to the caller as JSON.
    json.dumps(report)


def test_an_accuracy_gap_beyond_the_interval_is_rejected(
    tmp_path: Path,
) -> None:
    """A wide accuracy gap yields a rejected verdict, not an error."""
    reference = _BundleWriter(tmp_path / "reference").write(correct=1000)
    candidate = _BundleWriter(tmp_path / "candidate").write(correct=1250)

    report = compare_runs(reference, candidate)

    assert report["verdict"] == "rejected"
    assert report["difference"]["within_interval"] is False


def test_different_sample_sets_are_not_comparable(tmp_path: Path) -> None:
    """The sample-set identity must match exactly."""
    reference = _BundleWriter(tmp_path / "reference").write()
    candidate = _BundleWriter(tmp_path / "candidate").write(
        row_sha256="other-digest"
    )

    with pytest.raises(ValueError, match="different sample sets"):
        compare_runs(reference, candidate)


def test_different_sampling_settings_are_not_comparable(
    tmp_path: Path,
) -> None:
    """The sampling settings must match exactly."""
    reference = _BundleWriter(tmp_path / "reference").write()
    candidate = _BundleWriter(tmp_path / "candidate").write(
        sampling={"temperature": 0.6, "seed": 42}
    )

    with pytest.raises(ValueError, match="workload.sampling"):
        compare_runs(reference, candidate)


def test_different_cells_are_not_comparable(tmp_path: Path) -> None:
    """BF16 and NVFP4 runs are distinct cells and must not be compared."""
    reference = _BundleWriter(tmp_path / "reference").write()
    candidate = _BundleWriter(tmp_path / "candidate").write(
        provenance={"checkpoint": "Qwen3-30B-A3B", "precision": "nvfp4"}
    )

    with pytest.raises(ValueError, match="workload.provenance"):
        compare_runs(reference, candidate)


def test_different_served_checkpoints_are_not_comparable(
    tmp_path: Path,
) -> None:
    """The daemon command's model argument must name the same checkpoint."""
    reference = _BundleWriter(tmp_path / "reference").write()
    candidate = _BundleWriter(tmp_path / "candidate").write(
        command=[
            "/repo/.venv/bin/uniserve",
            "serve",
            "/models/Qwen3-30B-A3B-NVFP4",
        ]
    )

    with pytest.raises(ValueError, match="different checkpoints"):
        compare_runs(reference, candidate)


def test_an_invalid_reference_run_is_not_comparable(tmp_path: Path) -> None:
    """A run that failed validation cannot serve as the baseline."""
    reference = _BundleWriter(tmp_path / "reference").write(valid=False)
    candidate = _BundleWriter(tmp_path / "candidate").write()

    with pytest.raises(ValueError, match="not a valid run"):
        compare_runs(reference, candidate)


def test_an_incomplete_run_is_not_comparable(tmp_path: Path) -> None:
    """Only a completed bundle carries a canonical accuracy."""
    reference = _BundleWriter(tmp_path / "reference").write(status="failed")
    candidate = _BundleWriter(tmp_path / "candidate").write()

    with pytest.raises(ValueError, match="not completed"):
        compare_runs(reference, candidate)


def test_a_run_must_score_every_selected_row(tmp_path: Path) -> None:
    """The binomial denominator is the selected sample size."""
    reference = _BundleWriter(tmp_path / "reference").write()
    candidate = _BundleWriter(tmp_path / "candidate").write(
        correct=950, selected_count=1319, total=1000
    )

    with pytest.raises(ValueError, match="scored 1000 rows"):
        compare_runs(reference, candidate)


def test_a_pair_at_another_sample_size_is_not_comparable(
    tmp_path: Path,
) -> None:
    """The acceptance interval is defined at the protocol's 1319 rows."""
    reference = _BundleWriter(tmp_path / "reference").write(
        correct=95, total=100, selected_count=100
    )
    candidate = _BundleWriter(tmp_path / "candidate").write(
        correct=96, total=100, selected_count=100
    )

    with pytest.raises(ValueError, match="1319"):
        compare_runs(reference, candidate)


def test_a_summary_without_counts_is_not_comparable(tmp_path: Path) -> None:
    """The comparison reads integer correct and total metrics."""
    reference = _BundleWriter(tmp_path / "reference").write()
    candidate = _BundleWriter(tmp_path / "candidate").write(
        metrics={"accuracy": 0.9}
    )

    with pytest.raises(ValueError, match="metrics.correct"):
        compare_runs(reference, candidate)


def test_read_bundle_reports_missing_records(tmp_path: Path) -> None:
    """A directory that is not a bundle is reported, not guessed."""
    with pytest.raises(FileNotFoundError):
        read_bundle(tmp_path / "absent")


def _run_compare(argv: list[str]) -> int:
    """Invoke the command-line entry point and return its exit status."""
    previous = sys.argv
    sys.argv = ["uniserve-eval", *argv]
    try:
        try:
            main()
        except SystemExit as exit_info:
            return int(exit_info.code or 0)
    finally:
        sys.argv = previous
    return 0


def test_command_exit_status_reports_the_verdict(tmp_path: Path) -> None:
    """Accepted exits 0, rejected 2 and an unusable bundle 1."""
    reference = _BundleWriter(tmp_path / "reference").write(correct=1186)
    accepted = _BundleWriter(tmp_path / "accepted").write(correct=1190)
    rejected = _BundleWriter(tmp_path / "rejected").write(correct=1000)

    assert _run_compare(["compare", str(reference), str(accepted)]) == 0
    assert _run_compare(["compare", str(reference), str(rejected)]) == 2
    assert (
        _run_compare(["compare", str(reference), str(tmp_path / "absent")]) == 1
    )
