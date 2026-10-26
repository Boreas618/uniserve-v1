"""A prepared batch's device predicates: where they lie and what they keep."""

from __future__ import annotations

import pytest
import torch

from uniserve.executor.predicates import kept, split

pytestmark = pytest.mark.unit


def test_a_batchs_predicates_follow_its_other_tensors_in_call_order():
    reads = [torch.zeros(2), torch.ones(3)]
    first, second = torch.ones(1, dtype=torch.uint8), torch.zeros(1)
    calls = [
        {"index": 0, "predicate": 2},
        {"index": 1},
        {"index": 2, "predicate": 3},
    ]
    tensors, predicates = split(calls, [*reads, first, second])
    assert tensors == reads
    assert predicates == {0: first, 2: second}

    # A batch without predicates keeps every tensor.
    assert split([{"index": 0}], reads) == (reads, {})


def test_predicates_before_a_batchs_other_tensors_are_refused():
    calls = [{"index": 0, "predicate": 0}]
    with pytest.raises(ValueError):
        split(calls, [torch.zeros(1, dtype=torch.uint8), torch.zeros(2)])


def test_a_false_predicate_keeps_what_the_state_held():
    value, previous = torch.tensor([1.0, 2.0]), torch.tensor([7.0, 8.0])
    true = torch.ones(1, dtype=torch.uint8)
    false = torch.zeros(1, dtype=torch.uint8)
    assert torch.equal(kept(true, value, previous), value)
    assert torch.equal(kept(false, value, previous), previous)
    # An ungated call always writes what it computed.
    assert kept(None, value, previous) is value
