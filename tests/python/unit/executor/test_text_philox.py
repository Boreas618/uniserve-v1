"""Counter-based draws of the token sampler.

``uniserve.executor.text.philox``. The draws must equal ``ipc::philox``'s
bit for bit: the engine simulator draws the same values, and a seeded
request repeats across builds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uniserve.executor.text.philox import (
    DrawWindow,
    philox4x32_10,
    request_sampling_key,
    sampling_key,
    sampling_uniform,
)

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "generated"
    / "sampling_rng_parity.json"
)


def test_philox_matches_the_reference_known_answer_vectors():
    # Random123 Philox4x32-10 known-answer vectors establish the exact
    # round count, lane ordering, and key schedule.
    assert philox4x32_10([0, 0, 0, 0], [0, 0]).tolist() == [
        0x6627_E8D5,
        0xE169_C58D,
        0xBC57_AC4C,
        0x9B00_DBD8,
    ]
    assert philox4x32_10([0xFFFF_FFFF] * 4, [0xFFFF_FFFF] * 2).tolist() == [
        0x408F_276D,
        0x41C8_3B0E,
        0xA20B_C7C6,
        0x6D54_51FD,
    ]


def test_a_request_key_is_a_fixed_function_of_its_coordinates():
    for coordinates, key in [
        ((0, 0, 0, 0), 2_391_539_541_053_276_776),
        ((0, 7, 11, 3), 9_525_647_252_769_408_357),
        ((1, 4_000_000_000, 5, 900), 2_683_534_742_688_207_316),
        ((2**64 - 1, 7, 11, 3), 7_685_010_690_696_164_484),
    ]:
        assert sampling_key(*coordinates) == key


def test_a_seeded_request_stream_depends_on_its_seed_alone():
    seeded = request_sampling_key(7, 1, 2, 3)
    assert seeded == request_sampling_key(7, 4, 5, 6)
    assert seeded != request_sampling_key(8, 1, 2, 3)
    # Unseeded requests keep one stream per lineage.
    unseeded = request_sampling_key(None, 1, 2, 3)
    assert unseeded == request_sampling_key(None, 1, 2, 3)
    assert unseeded != request_sampling_key(None, 1, 5, 3)
    assert unseeded != request_sampling_key(None, 1, 2, 4)


def test_draws_match_the_shared_generator_fixture():
    cases = json.loads(_FIXTURE.read_text())["cases"]
    keys = np.array([case["key"] for case in cases], dtype=np.uint64)
    positions = np.array(
        [case["semantic_token_index"] for case in cases], dtype=np.uint64
    )
    for case in cases:
        # The fixture's target-sampling cases fold the key this runtime
        # draws with.
        if case["draw_layout"] == 0:
            assert case["key"] == sampling_key(
                case["session_seed"],
                case["engine_id"],
                case["request_id"],
                case["request_epoch"],
            )
        draw = sampling_uniform(
            case["key"],
            case["semantic_token_index"],
            case["processor_stage"],
            case["draw_index"],
        )
        assert int(np.float32(draw).view(np.uint32)) == case["uniform_bits"]
        index = case["semantic_token_index"]
        key = case["key"]
        counter = [
            index & 0xFFFF_FFFF,
            index >> 32,
            case["processor_stage"],
            case["draw_index"],
        ]
        words = philox4x32_10(counter, [key & 0xFFFF_FFFF, key >> 32])
        assert words.tolist() == case["words"]

    # Vectorized draws equal the case-by-case draws.
    stage_zero = [
        index
        for index, case in enumerate(cases)
        if case["processor_stage"] == 0 and case["draw_index"] == 0
    ]
    draws = sampling_uniform(keys[stage_zero], positions[stage_zero])
    assert draws.view(np.uint32).tolist() == [
        cases[index]["uniform_bits"] for index in stage_zero
    ]


def test_a_uniform_draw_stays_in_the_unit_interval():
    key = sampling_key(0x1234_5678_9ABC_DEF0, 7, 11, 3)
    draws = sampling_uniform(key, np.arange(4096, dtype=np.uint64))
    assert draws.dtype == np.float32
    assert np.all((draws >= 0.0) & (draws < 1.0))


def test_windowed_draws_equal_direct_draws():
    window = DrawWindow(slots=4, width=8)
    keys = np.array([0, 11, 12, 13, 14], dtype=np.uint64)

    def check(slots: list[int], positions: list[int], keys_of=keys) -> None:
        slot = np.asarray(slots)
        drawn = window.draws(slot, keys_of[slot], positions)
        direct = sampling_uniform(keys_of[slot], positions)
        assert drawn.view(np.uint32).tolist() == (
            direct.view(np.uint32).tolist()
        )

    # Decode steps advance one position per slot, crossing window ends.
    for step in range(20):
        check([1, 2, 3, 4], [step, 100 + step, 3 * step, 7])
    # A slot sampled at positions farther apart than one window.
    check([2, 2, 1], [0, 50, 5])
    # A slot taken by another request draws that request's stream.
    other = keys.copy()
    other[3] = 99
    check([3, 4], [10, 11], keys_of=other)
    check([3, 4], [10, 11])
