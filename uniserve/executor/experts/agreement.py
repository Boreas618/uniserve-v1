"""Step agreement of an expert union.

Every rank of an expert union executes the same sequence of expert layer
calls: a rank with tokens runs its forward, every other rank joins each
layer with no tokens of its own. Before each step the ranks agree over the
union's host group on which source group runs and on the step's per-rank
token capacity (:meth:`ExpertAgreement.agree`); :class:`Selection` makes
that choice from the gathered records without communication.

During startup warmup the source ranks announce each eager expert step
instead (:meth:`ExpertAgreement.warmup`): dedicated expert ranks run no
forward of their own and execute the announced capacities until the
sources announce zero.

Wire formats, all little-endian: a source membership is one ``u64`` per
member index, the sources' token bound and a warmup announcement are one
``u64``, and an agreement record is ``(tokens, leaving, kind)`` as three
``i64`` words.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["Agreement", "ExpertAgreement", "HostGroup", "Selection"]

# One agreement record: (tokens, leaving, kind).
_RECORD = struct.Struct("<qqq")

# One unsigned word: a member index, a token bound or an announcement.
_WORD = struct.Struct("<Q")


class Selection:
    """Coordinate a serialized sequence of expert calls over one rank group.

    Sources executing one forward together must all have the same call kind
    ready. Other sources and expert-only ranks participate without local
    rows. The caller supplies the gathered records; selection performs no
    communication.

    Args:
        size: The union's member count.
        rank: This rank's member index.
        memberships: Per member (in member order), the members that execute
            that member's forwards together: its worker's ranks for a
            source, nothing for an expert-only rank.
        max_tokens: The most tokens one rank sends per step.
        fused: The transport keeps one symmetric layout of the maximum
            capacity (MegaMoE); other transports exchange at power-of-two
            capacities plus the maximum.

    Raises:
        ValueError: Fewer than two members, ``rank`` outside the union, no
            token per step, or a membership naming a member outside the
            union.
    """

    def __init__(
        self,
        size: int,
        rank: int,
        memberships: Sequence[Sequence[int]],
        max_tokens: int,
        fused: bool,
    ) -> None:
        if size < 2 or not 0 <= rank < size or max_tokens <= 0:
            raise ValueError(
                "an expert exchange spans two or more ranks and one token"
            )

        # Every source declares the group of members that execute its
        # forwards together; expert-only ranks declare none. A group is
        # recorded once, by its first member.
        sources: list[tuple[int, ...]] = []
        for index, members in enumerate(memberships):
            peers = tuple(members)
            if any(not 0 <= member < size for member in peers):
                raise ValueError(
                    "an expert source names a rank outside the union"
                )

            if peers and peers[0] == index:
                sources.append(peers)

        if fused:
            capacities: tuple[int, ...] = (max_tokens,)
        else:
            # Powers of two strictly below the maximum, then the maximum:
            # 192 gives 1, 2, ..., 128, 192.
            powers = (max_tokens - 1).bit_length()
            capacities = (
                *(1 << power for power in range(powers)),
                max_tokens,
            )

        self._rank = rank
        self._sources = tuple(sources)
        self._max_tokens = max_tokens
        self._capacities = capacities

        # The kind of the last selected step: ready kinds take turns in
        # increasing cyclic order starting after it.
        self._kind = -1
        self._active = False
        self._released = False

    @property
    def capacities(self) -> tuple[int, ...]:
        """Step capacities, ascending."""
        return self._capacities

    @property
    def max_tokens(self) -> int:
        """The most tokens one rank sends per step."""
        return self._max_tokens

    @property
    def source_groups(self) -> int:
        """The number of source groups, each running its forwards together."""
        return len(self._sources)

    @property
    def active(self) -> bool:
        """The last selected step runs this rank's forward."""
        return self._active

    @property
    def released(self) -> bool:
        """At the last selection every rank was leaving without tokens."""
        return self._released

    def capacity_for(self, tokens: int) -> int:
        """The smallest capacity holding ``tokens`` (at least one).

        Raises:
            ValueError: ``tokens`` exceeds the maximum.
        """
        for capacity in self._capacities:
            if capacity >= tokens:
                return capacity

        raise ValueError(
            f"{tokens} tokens exceed the expert exchange's "
            f"{self._max_tokens} tokens per rank"
        )

    def select(self, records: Sequence[tuple[int, bool, int]]) -> int:
        """Select the next step from every member's record.

        A source group is ready when each of its members sends tokens of
        the kind its first member has ready. Among the ready groups the
        smallest kind above the last selected one runs, wrapping around;
        every ready group of that kind runs in the same step, which this
        rank's forward joins when the rank belongs to one of them.

        Args:
            records: Every member's ``(tokens, leaving, kind)``, in member
                order, with ``tokens >= 0``.

        Returns:
            The step's per-rank capacity: the smallest holding the most
            tokens any selected member sends; zero when no group is ready.

        Raises:
            ValueError: The selected step sends more than the maximum.
        """
        self._active = False
        self._released = all(
            tokens == 0 and leaving for tokens, leaving, _ in records
        )

        # (kind, most tokens, this rank takes part) of the selection.
        selected: tuple[int, int, bool] | None = None
        for members in self._sources:
            kind = records[members[0]][2]
            most = 0
            ready = True
            for member in members:
                tokens, _, tag = records[member]
                if tokens == 0 or tag != kind:
                    ready = False
                    break

                most = max(most, tokens)

            if not ready:
                continue

            active = self._rank in members
            if selected is None:
                selected = (kind, most, active)
            elif selected[0] == kind:
                selected = (kind, max(selected[1], most), selected[2] or active)
            elif (kind <= self._kind, kind) < (
                selected[0] <= self._kind,
                selected[0],
            ):
                # Kinds above the last selected one order before the kinds
                # at or below it; within each, ascending.
                selected = (kind, most, active)

        if selected is None:
            return 0

        kind, most, active = selected
        self._kind = kind
        self._active = active
        return self.capacity_for(most)


@dataclass(frozen=True, slots=True)
class Agreement:
    """The outcome of one agreement.

    Attributes:
        capacity: Per-rank token capacity of the selected step; zero when no
            source group is ready.
        active: This rank's pending forward was selected. Otherwise it joins
            the step with no tokens and keeps its forward for a later
            agreement.
        released: Every rank is leaving without tokens: the union may shut
            down.
    """

    capacity: int
    active: bool
    released: bool


@dataclass(frozen=True, slots=True)
class HostGroup:
    """Host collectives over the expert union's members.

    The worker supplies these functions over the union's TCP host group.
    Every member calls the same collectives in the same order, and never
    concurrently with another host collective of the union (a transport's
    bootstrap shares the connections). Each call blocks until every member
    reached it, and must release the interpreter lock while it waits (the
    worker's TVM-FFI functions with ``release_gil``), so the process's
    other Python threads keep running.

    Attributes:
        rank: This member's index.
        size: The union's member count.
        all_gather: ``(bytes) -> [bytes]``: every member's bytes, in member
            order.
        broadcast: ``(bytes | None, root) -> bytes``: the root member's
            bytes; only the root supplies a value.
        all_gather_fixed: ``(bytes) -> bytes``: every member's record of
            one width (at most 56 bytes), concatenated in member order.
    """

    rank: int
    size: int
    all_gather: Callable[[bytes], Sequence[bytes]]
    broadcast: Callable[[bytes | None, int], bytes]
    all_gather_fixed: Callable[[bytes], bytes]


class ExpertAgreement:
    """The host side of an expert union: step agreement over its host group.

    Construct with :meth:`configure`, collectively over the union. One
    thread makes every agreement of a rank (its own steps' and its idle
    joins'), because each advances the shared ready-kind rotation; the
    capacities are fixed at construction and readable from any thread.

    Args:
        host: The union's host collectives.
        selection: Step selection over the union's members.
        attention_ranks: Leading source-only members; zero when every rank
            holds its experts.
    """

    def __init__(
        self, host: HostGroup, selection: Selection, attention_ranks: int
    ) -> None:
        self._host = host
        self._selection = selection
        self._attention_ranks = attention_ranks
        self._record_bytes = _RECORD.size * host.size

        # One source group sends the tokens of every step, so each step runs
        # at the smallest capacity holding that group's tokens.
        self._one_source = selection.source_groups == 1

    @classmethod
    def configure(
        cls,
        host: HostGroup,
        attention_ranks: int,
        sources: Sequence[int],
        max_tokens: int,
        fused: bool,
    ) -> ExpertAgreement:
        """Form the union's step protocol. Collective over the union.

        Args:
            host: The union's host collectives.
            attention_ranks: Leading source-only members; zero when every
                rank holds its experts.
            sources: The union members that execute each of this rank's
                forwards together; empty on an expert-only rank.
            max_tokens: The most tokens one of this rank's forwards sends
                per step, graph padding included. With dedicated expert
                ranks the first member's bound applies to every rank, so
                expert ranks size their exchanges for the sources' capacity.
            fused: The transport keeps one symmetric layout of the maximum
                capacity (MegaMoE).

        Raises:
            ValueError: The attention ranks leave no expert rank, or the
                memberships are invalid (:class:`Selection`).
            RuntimeError: A member's record arrived malformed.
        """
        if attention_ranks >= host.size:
            raise ValueError(
                "attention ranks must leave at least one expert rank"
            )

        membership = b"".join(_WORD.pack(member) for member in sources)
        memberships = []
        for value in host.all_gather(membership):
            if len(value) % _WORD.size:
                raise RuntimeError(
                    "expert source membership record is malformed"
                )

            memberships.append(
                tuple(member for (member,) in _WORD.iter_unpack(value))
            )

        if attention_ranks:
            bound = _WORD.pack(max_tokens) if host.rank == 0 else None
            max_tokens = _word(
                host.broadcast(bound, 0),
                "expert capacity broadcast is malformed",
            )

        selection = Selection(
            host.size, host.rank, memberships, max_tokens, fused
        )
        return cls(host, selection, attention_ranks)

    @property
    def rank(self) -> int:
        """This rank's member index."""
        return self._host.rank

    @property
    def size(self) -> int:
        """The union's member count."""
        return self._host.size

    @property
    def attention_ranks(self) -> int:
        """Leading source-only members; zero for colocated experts."""
        return self._attention_ranks

    @property
    def max_tokens(self) -> int:
        """Tokens each rank may send per step."""
        return self._selection.max_tokens

    @property
    def capacities(self) -> tuple[int, ...]:
        """Step capacities, ascending."""
        return self._selection.capacities

    def capacity_for(self, tokens: int) -> int:
        """The smallest step capacity holding ``tokens``.

        Raises:
            ValueError: ``tokens`` exceeds :attr:`max_tokens`.
        """
        return self._selection.capacity_for(tokens)

    def capacities_for(self, tokens: int, microbatches: int) -> tuple[int, ...]:
        """The capacities a forward sending ``tokens`` tokens can run at.

        The forward is one of its step's ``microbatches``. An agreement
        selects the smallest capacity holding the most tokens any ready
        source group sends. With one source group and one microbatch that
        is the smallest capacity holding ``tokens``. With several groups,
        or another microbatch of the step sending more, any capacity
        holding them is reachable. Warmup captures the forward's graphs at
        these capacities, ascending; empty when ``tokens`` exceeds
        :attr:`max_tokens`.
        """
        holding = tuple(
            capacity
            for capacity in self._selection.capacities
            if capacity >= max(tokens, 1)
        )
        if self._one_source and microbatches == 1:
            return holding[:1]

        return holding

    def agree(self, tokens: int, kind: int, leaving: bool) -> Agreement:
        """Agree with the union on the next step.

        One host round trip; no device work. Only source groups whose
        members all have one kind ready may send tokens; ready kinds take
        turns in cyclic order. A rank shutting down agrees with ``leaving``
        until :attr:`Agreement.released`.

        Args:
            tokens: The tokens this rank's forward sends (its live rows, or
                its graph's padded extent when it replays); zero when it
                only joins.
            kind: The call kind this rank has ready: a text forward's group
                index, or ``64 + i`` for diffusion LM component ``i``.
            leaving: This rank is shutting down.

        Raises:
            ValueError: The selected step exceeds :attr:`max_tokens`.
            RuntimeError: The gathered records are malformed.
        """
        local = _RECORD.pack(tokens, 1 if leaving else 0, kind)
        gathered = self._host.all_gather_fixed(local)
        if len(gathered) != self._record_bytes:
            raise RuntimeError("expert agreement records are malformed")

        records = [
            (max(count, 0), flag != 0, tag)
            for count, flag, tag in _RECORD.iter_unpack(gathered)
        ]
        capacity = self._selection.select(records)
        return Agreement(
            capacity=capacity,
            active=capacity != 0 and self._selection.active,
            released=self._selection.released,
        )

    def warmup(self, capacity: int) -> int:
        """Announce or receive one eager warmup step's capacity.

        Sources announce each eager expert step of their warmup with its
        ``capacity``, and zero to end warmup. With dedicated expert ranks
        the first source broadcasts and every rank returns its value, so
        expert ranks execute the same layer sequence. Colocated ranks run
        identical warmup plans and need no announcement: the value returns
        unchanged.

        Raises:
            RuntimeError: The announcement arrived malformed.
        """
        if not self._attention_ranks:
            return capacity

        value = _WORD.pack(capacity) if self._host.rank == 0 else None
        return _word(
            self._host.broadcast(value, 0),
            "expert warmup announcement is malformed",
        )


def _word(value: bytes, malformed: str) -> int:
    """The single unsigned word ``value`` carries."""
    if len(value) != _WORD.size:
        raise RuntimeError(malformed)

    return _WORD.unpack(value)[0]
