"""The review manifest names, per cell, what a reviewer needs to judge an MP4.

The design confirms H3 quality by a person watching and listening, so the
manifest is the only thing that makes the review possible: without the prompt,
the seed, the inputs and the baseline MP4 beside the candidate's, a reviewer can
say only that a video plays.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uniserve_eval.h3_review_manifest import manifest, reference_media


def write_cell(root: Path, name: str, mp4: str) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / mp4).write_bytes(b"not really an mp4")


def profile(tmp_path: Path) -> Path:
    rows = {
        "check-t2va.jsonl": {
            "id": "w1-measured-seed0",
            "prompt": "a starship bridge",
            "seed": 0,
        },
        "check-ref2va.jsonl": {
            "id": "w1-measured-seed0",
            "prompt": "a man walks",
            "seed": 0,
            "input_references": [
                {"type": "image", "media": "media/ref.png"},
                {"type": "audio", "media": "media/ref.mp3"},
            ],
        },
    }
    for filename, row in rows.items():
        (tmp_path / filename).write_text(json.dumps(row) + "\n")

    # The tool resolves the profile's dataset paths against the repository
    # root, which is one level above the directory the profile lives in.
    (tmp_path / "uniserve_eval").mkdir()
    config = tmp_path / "uniserve_eval" / "profile.toml"
    config.write_text(
        """
[root]
path = "."
[servers.dmd]
checkpoint = "fasth3-dmd"
[servers.base-reference]
checkpoint = "base"
[benchmarks.check-t2va]
server = "dmd"
dataset_path = "check-t2va.jsonl"
[benchmarks.check-t2va.provenance]
task = "t2va"
[benchmarks.check-ref2va]
server = "base-reference"
dataset_path = "check-ref2va.jsonl"
[benchmarks.check-ref2va.provenance]
task = "ref2va"
[suites.check-h3]
points = ["check-t2va", "check-ref2va"]
"""
    )
    return config


def test_every_cell_has_a_task_a_checkpoint_a_seed_and_its_inputs(
    tmp_path: Path,
) -> None:
    config = profile(tmp_path)
    baseline = tmp_path / "run"
    write_cell(baseline, "check-t2va", "t2va.mp4")
    write_cell(baseline, "check-ref2va", "ref2va.mp4")

    text = manifest(config, baseline, None)

    assert "| `check-t2va` | t2va | `fasth3-dmd` | 0 |" in text
    assert "| `check-ref2va` | ref2va | `base` | 0 |" in text
    assert "**Baseline MP4**: `t2va.mp4`" in text
    assert "**Baseline MP4**: `ref2va.mp4`" in text
    assert "a starship bridge" in text
    assert "image: `media/ref.png`" in text
    assert "audio: `media/ref.mp3`" in text


def test_the_shipped_profile_names_the_three_checkpoints(
    tmp_path: Path,
) -> None:
    """The page names checkpoints, not the deployment that served them."""
    shipped = (
        Path(__file__).resolve().parents[4] / "uniserve_eval" / "check_h3.toml"
    )
    baseline = tmp_path / "run"
    for cell in (
        "check-base-ref2va",
        "check-fasth3-dmd-t2va",
        "check-fasth3-dmd-fl2va",
        "check-fasth3-omniref-ref2va",
        "check-base-t2va",
        "check-base-fl2va",
    ):
        write_cell(baseline, cell, f"{cell}.mp4")

    text = manifest(shipped, baseline, None)

    assert "`MiniMaxAI/MiniMax-H3`" in text
    assert "`FastVideo/FastVideo-FastH3-8-Step-V2`" in text
    assert "`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`" in text
    for server in ("base-reference", "base-transformer", "fasth3-dmd"):
        assert f"| `{server}` |" not in text


def test_the_candidate_column_is_empty_until_a_candidate_has_run(
    tmp_path: Path,
) -> None:
    config = profile(tmp_path)
    baseline = tmp_path / "run"
    write_cell(baseline, "check-t2va", "t2va.mp4")

    text = manifest(config, baseline, None)

    assert "Candidate run: not measured" in text
    assert "**Candidate MP4**: not measured" in text


def test_a_candidate_mp4_is_named_beside_the_baseline_it_is_compared_with(
    tmp_path: Path,
) -> None:
    config = profile(tmp_path)
    baseline = tmp_path / "run"
    candidate = tmp_path / "candidate"
    write_cell(baseline, "check-t2va", "baseline.mp4")
    write_cell(candidate, "check-t2va", "candidate.mp4")

    text = manifest(config, baseline, candidate)

    assert "**Baseline MP4**: `baseline.mp4`" in text
    assert "**Candidate MP4**: `candidate.mp4`" in text


def test_a_row_with_no_references_names_no_inputs() -> None:
    assert reference_media({"prompt": "x", "input_references": []}) == []
    assert reference_media({"prompt": "x"}) == []


def test_a_keyframe_names_the_frame_it_anchors() -> None:
    row = {
        "prompt": "x",
        "input_references": [
            {"type": "image", "media": "media/last.png", "frame_index": -1}
        ],
    }
    assert reference_media(row) == [
        "image: `media/last.png` (keyframe at frame -1)"
    ]


def test_a_candidate_alone_is_a_reviewable_run(tmp_path: Path) -> None:
    """The design's confirmation is the candidate's six videos.

    The earlier run's bundles need not survive on the host; a page that
    required them would make the review impossible exactly when it matters.
    """
    config = profile(tmp_path)
    candidate = tmp_path / "candidate"
    write_cell(candidate, "check-t2va", "t2va.mp4")

    text = manifest(config, None, candidate)

    assert "Baseline run: not on this host" in text
    assert "**Baseline MP4**: not on this host" in text
    assert "**Candidate MP4**: `t2va.mp4`" in text
    assert "Candidate run: `" in text


def test_a_page_without_a_run_is_refused(tmp_path: Path) -> None:
    """A page for no run would read as a review that found nothing."""
    config = profile(tmp_path)

    with pytest.raises(ValueError, match="needs a run"):
        manifest(config, None, None)
