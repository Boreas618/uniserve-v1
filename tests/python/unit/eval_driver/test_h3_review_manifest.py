"""The review manifest names, per cell, what a reviewer needs to judge an MP4.

The design confirms H3 quality by a person watching and listening, so the
manifest is the only thing that makes the review possible: without the prompt,
the seed, the inputs and the baseline MP4 beside the candidate's, a reviewer can
say only that a video plays.
"""

from __future__ import annotations

import json
from pathlib import Path

from uniserve_eval.h3_review_manifest import condition_media, manifest


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
            "conditions": [],
        },
        "check-ref2va.jsonl": {
            "id": "w1-measured-seed0",
            "prompt": "a man walks",
            "seed": 0,
            "conditions": [
                {
                    "type": "image",
                    "media": "media/ref.png",
                    "role": "reference",
                },
                {
                    "type": "audio",
                    "media": "media/ref.mp3",
                    "role": "reference",
                },
            ],
        },
    }
    for filename, row in rows.items():
        (tmp_path / filename).write_text(json.dumps(row) + "\n")

    config = tmp_path / "profile.toml"
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
[benchmarks.check-t2va.video]
task = "t2va"
[benchmarks.check-ref2va]
server = "base-reference"
dataset_path = "check-ref2va.jsonl"
[benchmarks.check-ref2va.video]
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
    assert "image: `media/ref.png` (reference)" in text
    assert "audio: `media/ref.mp3` (reference)" in text


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


def test_a_row_with_no_conditions_names_no_inputs() -> None:
    assert condition_media({"prompt": "x", "conditions": []}) == []
    assert condition_media({"prompt": "x"}) == []
