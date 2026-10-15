"""Expert steps: capacity, reached layers and tail joins of a forward.

Every rank of an expert group calls the same expert layers in every step.
A one-member transport stands in for the other ranks (the communication
boundary): its dispatch receives this rank's own rows, so the step's
numerics equal the layers' standalone evaluation, and its dispatches show
which layers exchanged and with how many rows.
"""

import pytest
import torch
from torch import nn

from uniserve.backends.experts import ExpertTransport
from uniserve.bindings.context import ExecutionContext
from uniserve.distributed import Communicator
from uniserve.nn.moe import FusedMoE

pytestmark = pytest.mark.unit

TOKENS, HIDDEN, EXPERTS, TOP_K = 5, 16, 4, 2


class OneMemberTransport(ExpertTransport):
    """An expert group of this rank alone: routed rows arrive here."""

    transport = "alltoall"

    def __init__(self, max_tokens: int) -> None:
        super().__init__(
            Communicator(),
            None,
            max_tokens=max_tokens,
            attention_ranks=0,
            barrier_first=False,
            yield_turn=None,
        )
        self.dispatched: list[int] = []

    def dispatch(self, hidden, topk_ids, topk_weights, *, invalid_expert):
        if hidden.shape[0] > self.capacity:
            raise ValueError("a rank's tokens exceed the step capacity")
        self.dispatched.append(hidden.shape[0])
        return hidden, topk_ids, topk_weights

    def combine(self, partial, tokens):
        return partial[:tokens]


def layers() -> nn.ModuleList:
    torch.manual_seed(0)
    modules = nn.ModuleList(
        FusedMoE(EXPERTS, HIDDEN, 8, top_k=TOP_K, activation="silu")
        for _ in range(3)
    )
    for module in modules:
        for parameter in module.parameters():
            nn.init.normal_(parameter, std=0.1)
    return modules


def routes():
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(TOKENS, HIDDEN, generator=generator)
    ids = torch.randint(0, EXPERTS, (TOKENS, TOP_K), generator=generator)
    weights = torch.rand(TOKENS, TOP_K, generator=generator)
    return hidden, ids.to(torch.int32), weights


def bound(modules, transport):
    context = ExecutionContext(modules, experts=transport)
    context.prepare(_size(transport.max_tokens))
    return context


def _size(tokens):
    from uniserve.models import TextSize

    return TextSize(tokens, 1)


def test_a_forward_reaching_two_layers_joins_the_third_without_tokens():
    modules = layers()
    hidden, ids, weights = routes()
    expected = modules[1](modules[0](hidden, ids, weights), ids, weights)

    transport = OneMemberTransport(max_tokens=8)
    context = bound(modules, transport)
    with context.activate(), context.expert_step(8):
        output = modules[1](modules[0](hidden, ids, weights), ids, weights)

    torch.testing.assert_close(output, expected)
    assert transport.dispatched == [TOKENS, TOKENS, 0]
    assert transport.capacity == 0


def test_a_forward_may_not_skip_a_layer_before_one_it_reaches():
    modules = layers()
    hidden, ids, weights = routes()
    context = bound(modules, OneMemberTransport(max_tokens=8))
    with pytest.raises(RuntimeError, match="skipped an expert layer"):
        with context.activate(), context.expert_step(8):
            modules[1](hidden, ids, weights)


def test_tokens_and_capacity_stay_within_their_bounds():
    modules = layers()
    hidden, ids, weights = routes()
    context = bound(modules, OneMemberTransport(max_tokens=8))
    with pytest.raises(ValueError, match="capacity"):
        with context.activate(), context.expert_step(16):
            pass
    with pytest.raises(ValueError, match="step capacity"):
        with context.activate(), context.expert_step(4):
            modules[0](hidden, ids, weights)


def test_expert_parallel_layers_run_only_inside_an_expert_step():
    modules = layers()
    hidden, ids, weights = routes()
    context = bound(modules, OneMemberTransport(max_tokens=8))
    with pytest.raises(RuntimeError, match="inside an expert step"):
        with context.activate():
            modules[0](hidden, ids, weights)
