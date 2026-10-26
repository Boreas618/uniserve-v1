"""Request constants and slot state of a text lane.

``uniserve.executor.text.state``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from uniserve.executor.text.philox import request_sampling_key
from uniserve.executor.text.sampling import (
    NEUTRAL,
    SamplingParams,
    sampling_paths,
)
from uniserve.executor.text.state import NewRequest, RequestTable, SlotState
from uniserve.sampling.select import CATEGORICAL, GREEDY, TOP_K_PATH

pytestmark = pytest.mark.unit


def _state() -> tuple[SlotState, torch.Tensor]:
    # Three request slots over a five-token vocabulary.
    lengths = torch.zeros(4, dtype=torch.int32)
    return SlotState.allocate(3, 5, torch.bfloat16, lengths), lengths


def test_reset_restores_a_slot_without_touching_its_neighbors():
    state, lengths = _state()
    assert state.future_input_tokens.view(-1).tolist() == [1, 1, 1, 1]

    # Dirty slots 1 and 2, then reset slot 2 only.
    for slot in [1, 2]:
        state.future_input_tokens[slot] = 77
        state.logical_lengths[slot] = 9
        lengths[slot] = 9
        state.penalty_counts[slot, 0] = 3
    state.set_predicates([1, 2], True)
    state.reset([2])
    assert state.future_input_tokens.view(-1).tolist() == [1, 77, 1, 1]
    assert state.logical_lengths.tolist() == [0, 9, 0, 0]
    assert lengths.tolist() == [0, 9, 0, 0]
    assert state.predicates.tolist() == [0, 1, 0, 0]
    counts = state.penalty_counts.view(-1)
    assert counts[5] == 3
    assert counts[10] == 0
    with pytest.raises(ValueError):
        state.reset([4])


def test_a_request_installs_its_logit_biases_in_its_slot():
    state, _ = _state()
    state.install_bias(2, [(1, 0.5), (4, -2.0)])
    biases = state.logit_bias.view(-1)
    assert biases[10:15].tolist() == [0.0, 0.5, 0.0, 0.0, -2.0]
    assert not biases[:10].any() and not biases[15:].any()
    # Token 5 is outside the vocabulary of five tokens.
    state.install_bias(1, [(5, 1.0)])
    assert not state.logit_bias[1].any()
    with pytest.raises(ValueError):
        state.install_bias(4, [(0, 1.0)])


def test_a_resumed_request_counts_its_generated_tokens():
    state, _ = _state()
    state.count_generated(2, [4, 1, 4])
    counts = state.penalty_counts.view(-1)
    assert counts[10:15].tolist() == [0, 1, 0, 0, 2]
    assert not counts[:10].any() and not counts[15:].any()
    with pytest.raises(ValueError):
        state.count_generated(1, [5])


def test_a_slot_selects_its_requests_adapter():
    state, _ = _state()
    state.select_adapter(3, 7)
    assert state.adapters.tolist() == [0, 0, 0, 7]
    state.reset([3])
    assert state.adapters.tolist() == [0, 0, 0, 0]


def _request(slot: int, **sampling: object) -> NewRequest:
    return NewRequest.from_json(
        {
            "request_key": {
                "engine_id": 1,
                "request_id": 100 + slot,
                "request_epoch": 2,
            },
            "request_pool_idx": slot,
            "generation": {
                "sampling": sampling,
                "negative_token_ids": [],
                "finish_token_ids": [7, 2],
                "initial_position": 0,
                "generated_token_ids": [],
            },
            "image": None,
            "diffusion": None,
            "video": None,
            "prompt_token_ids": [],
            "input_images": 0,
        }
    )


def test_a_started_request_resolves_its_sampler_from_its_slot():
    table = RequestTable(slots=4, vocab=1000)
    table.start(_request(1))
    table.start(_request(2, temperature=0.8, top_k=40, seed=11))
    table.start(_request(3, temperature=0.8, typical_p=0.5))
    table.start(_request(4, allowed_token_ids=[3, 4]))
    slots = np.array([1, 2, 3, 4])
    assert sampling_paths(table, slots).tolist() == [
        GREEDY,
        TOP_K_PATH,
        CATEGORICAL,
        CATEGORICAL,
    ]
    assert table.finish_ids[1].tolist() == [7, 2, -1, -1, -1, -1, -1, -1]
    assert table.key[2] == request_sampling_key(11, 0, 0, 0)
    assert table.key[1] == request_sampling_key(None, 1, 101, 2)
    assert table.params[2].tolist() == pytest.approx(
        [0.8, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 40.0]
    )

    # A call must name the request its slot holds.
    table.check(slots, np.array([101, 102, 103, 104]), np.full(4, 2))
    with pytest.raises(ValueError):
        table.check(slots, np.array([101, 102, 103, 999]), np.full(4, 2))

    # A released slot holds neutral values again and admits no call.
    table.release([2])
    assert table.params[2].tolist() == NEUTRAL.tolist()
    assert sampling_paths(table, np.array([2])).tolist() == [GREEDY]
    with pytest.raises(ValueError):
        table.check(np.array([2]), np.array([102]), np.array([2]))
    with pytest.raises(ValueError):
        table.start(_request(5))


def test_sampling_parameters_compare_as_float32():
    # 0.1 is not a float32 value; the wire value is, and the request's
    # nucleus cutoff is the float32 one.
    params = SamplingParams.from_json({"top_p": 0.1, "temperature": 1.0})
    assert params.top_p == float(np.float32(0.1))
    assert params.parameters(1000)[1] == np.float32(0.1)
