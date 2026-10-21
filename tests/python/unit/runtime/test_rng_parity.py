"""Token selection on the rank's random draws matches the simulator's rule.

``tests/python/generated/sampling_rng_parity.json`` holds two kinds of cases.
Each RNG case records a uniform draw of ``uniserve_ipc::philox``, the
generator the rank's sampler draws through, and the token the inverse-CDF
rule selects from the shared dyadic ``probs`` at that draw. Each boundary
case pairs its own distribution with a draw at an edge of the rule (the
first positive-mass entry whose cumulative mass exceeds the draw, else the
last positive-mass entry). ``uniserve.sampling.sample_categorical`` selects
every case here, and the simulator's ``sample_categorical`` unit test
selects from the same cases, so both implementations share one selection
boundary.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from uniserve.sampling import sample_categorical

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "generated"
    / "sampling_rng_parity.json"
)


def _draw(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _select_token(probs: list[float], draw: float) -> int:
    # Fixture probabilities and draws are exact float32 values.
    selected = sample_categorical(
        torch.tensor([probs], dtype=torch.float32),
        torch.tensor([draw], dtype=torch.float32),
    )
    return int(selected.item())


def test_the_generator_draws_select_the_simulator_tokens() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    probs = fixture["probs"]
    assert fixture["cases"], "fixture carries at least one coordinate"
    selected = [
        _select_token(probs, _draw(case["uniform_bits"]))
        for case in fixture["cases"]
    ]
    assert selected == [case["selected_token"] for case in fixture["cases"]]


def test_inverse_cdf_boundaries_match_the_shared_fixture() -> None:
    # Covers a zero draw with masked leading entries, a draw on a cumulative
    # boundary, and float32 totals below the largest canonical draw.
    boundaries = json.loads(_FIXTURE.read_text())["boundaries"]
    assert boundaries, "fixture carries boundary cases"
    selected = {
        case["name"]: _select_token(case["probs"], _draw(case["uniform_bits"]))
        for case in boundaries
    }
    assert selected == {
        case["name"]: case["selected_token"] for case in boundaries
    }
