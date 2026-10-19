"""The model package's MiniMax-H3 rules follow the reference vectors.

``processing`` calls the request rules of ``uniserve-core`` through the rank
runtime library. Its canvas rule and the frame count of a duration match the
vectors in ``minimax_h3_plan.json``, generated from the diffusers MiniMax-H3
pipeline; the rules' own tests (``crates/core/src/video``) check every
planned request and presentation against the same vectors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uniserve.media import image
from uniserve.models.minimax_h3 import processing

pytestmark = pytest.mark.unit

_FIXTURE = json.loads(
    (
        Path(__file__).parents[2] / "fixtures" / "minimax_h3_plan.json"
    ).read_text()
)


@pytest.mark.parametrize("case", _FIXTURE["canvas_cases"])
def test_canvas(case):
    if case.get("error"):
        with pytest.raises(ValueError):
            processing.canvas(*case["aspect"])
        return
    width, height = case["canvas"]
    assert processing.canvas(*case["aspect"]) == image.Config(
        height=height, width=width
    )


@pytest.mark.parametrize("case", _FIXTURE["duration_cases"])
def test_duration(case):
    if case.get("error"):
        with pytest.raises(ValueError):
            processing.frame_count(case["seconds"])
        return
    assert processing.frame_count(case["seconds"]) == case["num_frames"]


def test_max_seconds_bounds_the_duration():
    assert processing.frame_count(8.0, max_seconds=8.0) == 192
    with pytest.raises(ValueError):
        processing.frame_count(10.0, max_seconds=8.0)
