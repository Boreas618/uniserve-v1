"""Expert step agreement over an expert union's host group
(``uniserve.executor.experts.agreement``).
"""  # noqa: D205

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import pytest

from uniserve.executor.experts import ExpertAgreement, HostGroup, Selection

pytestmark = pytest.mark.unit

T = TypeVar("T")


def selection(rank: int, fused: bool) -> Selection:
    # Members 0 and 2 form one source group, 1 sources alone, and 3 and 4
    # hold experts only.
    return Selection(5, rank, [[0, 2], [1], [0, 2], [], []], 192, fused)


def test_source_groups_wait_together_and_ready_kinds_take_turns():
    for rank in range(5):
        exchange = selection(rank, fused=False)
        records = [
            (129, False, 2),
            (7, False, 0),
            (0, False, 2),
            (0, False, 0),
            (0, False, 0),
        ]
        # The group of members 0 and 2 waits for member 2's tokens.
        assert exchange.select(records) == 8
        assert exchange.active == (rank == 1)

        # Both groups are ready: kind 2 follows the last selected kind 0,
        # and the next selection wraps around to kind 0.
        records[2] = (64, False, 2)
        assert exchange.select(records) == 192
        assert exchange.active == (rank in (0, 2))
        assert exchange.select(records) == 8
        assert exchange.active == (rank == 1)

        # A group whose members have different kinds ready waits.
        records[2] = (64, False, 1)
        assert exchange.select(records) == 8
        assert exchange.active == (rank == 1)

        records[2] = (64, False, 2)
        records[0] = (193, False, 2)
        with pytest.raises(ValueError, match="193 tokens"):
            exchange.select(records)


def test_release_waits_for_every_rank_and_no_remaining_tokens():
    exchange = selection(1, fused=True)
    records = [(0, True, 0)] * 5
    records[4] = (0, False, 0)
    assert exchange.select(records) == 0
    assert not exchange.released

    records[4] = (0, True, 0)
    records[1] = (1, True, 0)
    assert exchange.select(records) == 192
    assert not exchange.released

    records[1] = (0, True, 0)
    assert exchange.select(records) == 0
    assert exchange.released


def test_capacities_are_powers_of_two_below_the_maximum():
    exchange = selection(0, fused=False)
    assert exchange.capacities == (1, 2, 4, 8, 16, 32, 64, 128, 192)
    assert exchange.capacity_for(0) == 1
    assert exchange.capacity_for(65) == 128
    assert exchange.capacity_for(192) == 192
    with pytest.raises(ValueError, match="193 tokens"):
        exchange.capacity_for(193)

    assert selection(0, fused=True).capacities == (192,)


@pytest.mark.parametrize(
    ("size", "rank", "memberships", "max_tokens"),
    [
        (1, 0, [[0]], 8),
        (2, 2, [[0], [1]], 8),
        (2, 0, [[0], [1]], 0),
        (2, 0, [[0, 2], [1]], 8),
    ],
    ids=["one-rank", "rank-outside", "no-token", "source-outside"],
)
def test_an_exchange_rejects_an_invalid_union(
    size, rank, memberships, max_tokens
):
    with pytest.raises(ValueError):
        Selection(size, rank, memberships, max_tokens, fused=False)


class _Mesh:
    """An in-process host group: every collective exchanges one value per
    member through shared slots, between two barriers.
    """  # noqa: D205

    def __init__(self, size: int) -> None:
        self.size = size
        self.slots: list[object] = [None] * size
        self.barrier = threading.Barrier(size, timeout=30)

    def exchange(self, rank: int, value: object) -> list[object]:
        self.slots[rank] = value
        self.barrier.wait()
        values = list(self.slots)
        # No member writes the next collective's value before every member
        # read this one's.
        self.barrier.wait()
        return values

    def member(self, rank: int) -> HostGroup:
        def all_gather(value: bytes) -> list[bytes]:
            return [bytes(item) for item in self.exchange(rank, bytes(value))]

        def broadcast(value: bytes | None, root: int) -> bytes:
            # Only the root supplies its value, as the worker's group
            # requires.
            assert (value is not None) == (rank == root)
            return bytes(self.exchange(rank, value)[root])

        def all_gather_fixed(record: bytes) -> bytes:
            records = self.exchange(rank, bytes(record))
            assert len({len(item) for item in records}) == 1
            return b"".join(records)

        return HostGroup(
            rank, self.size, all_gather, broadcast, all_gather_fixed
        )


def union(size: int, body: Callable[[int, HostGroup], T]) -> list[T]:
    """Run ``body`` on every member of an in-process union of ``size``."""
    mesh = _Mesh(size)

    def member(rank: int) -> T:
        try:
            return body(rank, mesh.member(rank))
        except BaseException:
            # Peers waiting in a collective fail instead of hanging.
            mesh.barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=size) as pool:
        futures = [pool.submit(member, rank) for rank in range(size)]
        results = []
        errors = []
        for future in futures:
            try:
                results.append(future.result(timeout=60))
            except BaseException as error:  # noqa: BLE001 - reported below
                errors.append(error)

    # The member that failed first, not a peer its abort interrupted.
    for error in errors:
        if not isinstance(error, threading.BrokenBarrierError):
            raise error
    if errors:
        raise errors[0]

    return results


def test_colocated_ranks_agree_on_the_selected_step_and_release_together():
    def body(rank: int, host: HostGroup):
        agreement = ExpertAgreement.configure(host, 0, [rank], 64, fused=False)
        # Rank 0 sends 5 tokens of kind 0, rank 2 9 tokens of kind 1; kinds
        # take turns, the idle rank joins both.
        tokens = [5, 0, 9][rank]
        kind = [0, 0, 1][rank]
        first = agreement.agree(tokens, kind, False)
        second = agreement.agree(tokens, kind, False)
        leaving = agreement.agree(0, 0, True)
        assert agreement.warmup(16) == 16
        return first, second, leaving

    for rank, (first, second, leaving) in enumerate(union(3, body)):
        assert (first.capacity, first.active) == (8, rank == 0)
        assert (second.capacity, second.active) == (16, rank == 2)
        assert leaving.capacity == 0
        assert leaving.released


@pytest.mark.parametrize("together", [True, False])
def test_steps_run_at_a_capacity_their_sources_prepared(together: bool):
    # Ranks 0 and 1 source tokens and rank 2 holds the experts; the sources
    # form one group (`together`) or one group each.
    def body(rank: int, host: HostGroup):
        if rank == 2:
            sources = []
        elif together:
            sources = [0, 1]
        else:
            sources = [rank]
        agreement = ExpertAgreement.configure(host, 2, sources, 48, fused=False)
        # A group's members send equal tokens; separate groups do not.
        if rank == 2:
            tokens = 0
        elif together:
            tokens = 5
        else:
            tokens = [5, 20][rank]
        prepared = agreement.capacities_for(tokens, 1)
        agreed = agreement.agree(tokens, 0, False)
        return prepared, agreed.capacity

    results = union(3, body)
    expected = [(8,), (8,)] if together else [(8, 16, 32, 48), (32, 48)]
    for (prepared, agreed), capacities in zip(
        results[:2], expected, strict=True
    ):
        assert prepared == capacities
        assert agreed in prepared

    # The expert rank runs the same step.
    assert len({agreed for _, agreed in results}) == 1


def test_each_microbatch_prepares_the_capacity_of_its_largest_peer():
    # One source group of two ranks, each splitting its step into two
    # microbatches of 5 and 20 tokens; rank 2 holds the experts.
    def body(rank: int, host: HostGroup):
        sources = [0, 1] if rank < 2 else []
        agreement = ExpertAgreement.configure(host, 2, sources, 48, fused=False)
        # A rank sends the most tokens of its microbatches.
        tokens = 20 if rank < 2 else 0
        agreed = agreement.agree(tokens, 0, False).capacity
        prepared = [agreement.capacities_for(tokens, 2) for tokens in (5, 20)]
        alone = agreement.capacities_for(5, 1)
        return agreed, prepared, alone

    for agreed, prepared, alone in union(3, body)[:2]:
        assert agreed == 32
        assert prepared == [(8, 16, 32, 48), (32, 48)]
        assert all(agreed in capacities for capacities in prepared)
        assert alone == (8,)


def test_expert_ranks_borrow_the_source_bound_and_follow_announcements():
    def body(rank: int, host: HostGroup):
        # Two source ranks, one expert rank whose own bound is ignored.
        sources = [rank] if rank < 2 else []
        agreement = ExpertAgreement.configure(
            host, 2, sources, [48, 48, 1][rank], fused=False
        )
        announced = agreement.warmup(32 if rank == 0 else 0)
        return agreement.max_tokens, agreement.capacities, announced

    for max_tokens, capacities, announced in union(3, body):
        assert (max_tokens, announced) == (48, 32)
        assert capacities == (1, 2, 4, 8, 16, 32, 48)


def test_attention_ranks_leave_an_expert_rank():
    def never(*_):
        raise AssertionError("no collective runs")

    host = HostGroup(0, 2, never, never, never)
    with pytest.raises(ValueError, match="expert rank"):
        ExpertAgreement.configure(host, 2, [0], 8, fused=False)
