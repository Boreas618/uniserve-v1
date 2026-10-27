"""Mathematical distribution of one logical tensor on a device mesh.

A :class:`Distribution` names, for every axis of a device mesh, how that
axis divides the tensor. PyTorch's ``Shard(dim)`` and ``Replicate()`` cover
an even contiguous split and a full copy; three further kinds of division
cover what serving needs besides them:

- :class:`Cyclic`: position ``i`` of a dimension lies on member
  ``i mod n`` of the axis, as decode context parallelism distributes token
  positions.
- :class:`BlockShard`: a dimension made of named sub-blocks, each split
  evenly on its own, as the up and gate rows of stacked expert weights, or
  the query, key and value channels of a convolution window.
- :class:`Strided`: entry ``j`` of a compressed index stands for position
  ``j * step + offset`` of a domain distributed position-mod-N, so it lies
  with that position.

:meth:`Distribution.segments` resolves what one mesh coordinate holds as a
list of :class:`Segment` per dimension: index progressions of the logical
tensor (:class:`Span`) and where each starts along the local tensor. The
local tensor holds the Cartesian product of its dimensions' segments, in
segment order along each dimension. The distribution describes indices
only; what the local tensor's bytes are is its encoding, which the tensor
itself (or a quantized tensor's fields) states.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from torch.distributed.tensor import Placement, Replicate, Shard

from .mesh import DeviceMesh


@dataclass(frozen=True, slots=True)
class Span:
    """The indices ``start + step * k`` for ``k`` in ``range(count)``.

    An arithmetic progression of one dimension's indices: a contiguous
    range has step one, a position-mod-N share the step N.
    """

    start: int
    count: int
    step: int = 1

    def __post_init__(self) -> None:
        if self.start < 0 or self.count < 0 or self.step < 1:
            raise ValueError(
                "a span starts at a nonnegative index with a nonnegative "
                "count and a positive step"
            )

    def index(self, k: int) -> int:
        """The ``k``-th index of the span."""
        return self.start + self.step * k

    def indices(self) -> range:
        return range(self.start, self.start + self.step * self.count, self.step)


@dataclass(frozen=True, slots=True)
class Segment:
    """Logical indices ``span`` of one dimension at local index ``local``.

    The local tensor holds index ``span.index(k)`` at local position
    ``local + k`` along the dimension.
    """

    span: Span
    local: int


class Cyclic(Placement):
    """Position ``i`` of dimension ``dim`` lies on member ``i mod n``.

    Each member's local tensor holds its positions in increasing order:
    member ``c`` of ``n`` holds ``c, c + n, c + 2n, ...``.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cyclic) and other.dim == self.dim

    def __hash__(self) -> int:
        return hash(("cyclic", self.dim))

    def __repr__(self) -> str:
        return f"Cyclic(dim={self.dim})"


class BlockShard(Placement):
    """Dimension ``dim`` is named sub-blocks, each split evenly on its own.

    ``blocks`` names each sub-block and its extent, in the order the
    dimension stores them; they cover the dimension the axis receives.
    Member ``c`` of ``n`` holds the ``c``-th of ``n`` equal shares of every
    block, its shares stored in block order.
    """

    def __init__(self, dim: int, blocks: tuple[tuple[str, int], ...]) -> None:
        if not blocks or any(size < 1 for _, size in blocks):
            raise ValueError("block sharding names positive sub-blocks")
        self.dim = dim
        self.blocks = tuple((str(name), int(size)) for name, size in blocks)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BlockShard)
            and other.dim == self.dim
            and other.blocks == self.blocks
        )

    def __hash__(self) -> int:
        return hash(("blocks", self.dim, self.blocks))

    def __repr__(self) -> str:
        return f"BlockShard(dim={self.dim}, blocks={self.blocks})"


class Strided(Placement):
    """Entry ``j`` of ``dim`` stands for position ``j * step + offset``.

    The positions are distributed position-mod-N (:class:`Cyclic`), so
    entry ``j`` lies on member ``(j * step + offset) mod n``: a compressed
    index with one entry per ``step`` positions lies with the position each
    entry stands for. A member no entry's position falls on holds none.
    """

    def __init__(self, dim: int, step: int, offset: int = 0) -> None:
        if step < 1 or offset < 0:
            raise ValueError("a strided index has a positive step")
        self.dim = dim
        self.step = step
        self.offset = offset

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Strided) and (
            other.dim,
            other.step,
            other.offset,
        ) == (self.dim, self.step, self.offset)

    def __hash__(self) -> int:
        return hash(("strided", self.dim, self.step, self.offset))

    def __repr__(self) -> str:
        return (
            f"Strided(dim={self.dim}, step={self.step}, offset={self.offset})"
        )


def _even(length: int, members: int, what: str) -> int:
    if length % members:
        raise ValueError(
            f"{what} of {length} indices does not split evenly over "
            f"{members} members"
        )
    return length // members


def _narrow(segments: list[Segment], first: int, length: int) -> list[Segment]:
    """The segments' indices at local positions ``[first, first + length)``,
    renumbered from local position zero.
    """  # noqa: D205
    narrowed = []
    for segment in segments:
        low = max(first, segment.local)
        high = min(first + length, segment.local + segment.span.count)
        if low >= high:
            continue
        skipped = low - segment.local
        narrowed.append(
            Segment(
                Span(
                    segment.span.index(skipped), high - low, segment.span.step
                ),
                low - first,
            )
        )
    return narrowed


def _congruence(
    factor: int, target: int, modulus: int
) -> tuple[int, int] | None:
    """The solutions ``k`` of ``factor * k = target (mod modulus)`` as
    ``(first, period)``, ``None`` when there are none.
    """  # noqa: D205
    divisor = gcd(factor, modulus)
    if target % divisor:
        return None
    period = modulus // divisor
    if period == 1:
        return 0, 1
    inverse = pow(factor // divisor, -1, period)
    return (target // divisor) * inverse % period, period


def _split(
    placement: Placement,
    segments: list[Segment],
    members: int,
    coordinate: int,
) -> list[Segment]:
    """What member ``coordinate`` of ``members`` keeps of one dimension's
    ``segments`` under ``placement``.
    """  # noqa: D205
    length = sum(segment.span.count for segment in segments)
    if isinstance(placement, Shard):
        share = _even(length, members, "a shard")
        return _narrow(segments, coordinate * share, share)
    if isinstance(placement, Cyclic):
        kept = []
        position = 0
        for segment in segments:
            # Local positions = coordinate (mod members) inside the segment.
            first = (coordinate - segment.local) % members
            if first < segment.span.count:
                count = (segment.span.count - first + members - 1) // members
                kept.append(
                    Segment(
                        Span(
                            segment.span.index(first),
                            count,
                            segment.span.step * members,
                        ),
                        position,
                    )
                )
                position += count
        return kept
    if isinstance(placement, BlockShard):
        if sum(size for _, size in placement.blocks) != length:
            raise ValueError(
                "named sub-blocks must cover the dimension they divide"
            )
        kept = []
        position = 0
        start = 0
        for _, size in placement.blocks:
            share = _even(size, members, "a sub-block")
            for segment in _narrow(segments, start + coordinate * share, share):
                kept.append(Segment(segment.span, position + segment.local))
            position += share
            start += size
        return kept
    if isinstance(placement, Strided):
        kept = []
        position = 0
        for segment in segments:
            span = segment.span
            # Entry span.index(k) lies on member (index * step + offset)
            # mod members: solve span.step * step * k = coordinate - offset
            # - span.start * step (mod members).
            solution = _congruence(
                span.step * placement.step % members,
                (coordinate - placement.offset - span.start * placement.step)
                % members,
                members,
            )
            if solution is None:
                continue
            first, period = solution
            if first >= span.count:
                continue
            count = (span.count - first + period - 1) // period
            kept.append(
                Segment(
                    Span(span.index(first), count, span.step * period),
                    position,
                )
            )
            position += count
        return kept
    raise TypeError(f"{placement!r} does not divide a tensor dimension")


@dataclass(frozen=True, slots=True)
class Distribution:
    """Describe tensor placements without owning storage or communication."""

    mesh: DeviceMesh
    placements: tuple[Placement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.placements, tuple) or len(
            self.placements
        ) != len(self.mesh.axes):
            raise ValueError("placements must provide one entry per mesh axis")
        if any(
            not isinstance(placement, Placement)
            for placement in self.placements
        ):
            raise TypeError(
                "tensor placements must use PyTorch Placement objects"
            )

    def shard_axes(self, dim: int) -> tuple[str, ...]:
        """Return the mesh axes partitioning the specified tensor dimension."""
        return tuple(
            axis
            for axis, placement in zip(
                self.mesh.axes, self.placements, strict=True
            )
            if isinstance(placement, Shard) and placement.dim == dim
        )

    def segments(
        self, shape: tuple[int, ...], coordinate: tuple[int, ...] | None = None
    ) -> tuple[tuple[Segment, ...], ...]:
        """What the mesh coordinate holds of a tensor of ``shape``.

        The axes divide the tensor in mesh order, each the local tensor the
        axes before it left, as PyTorch nests placements. ``coordinate``
        defaults to the mesh's own rank's.

        Returns:
            One tuple of segments per dimension: the logical indices the
            local tensor holds along it, with their local positions.

        Raises:
            ValueError: An axis does not divide its dimension evenly.
        """
        if coordinate is None:
            coordinate = self.mesh.coordinate(self.mesh.rank)
        held = [[Segment(Span(0, extent), 0)] for extent in shape]
        for placement, members, position in zip(
            self.placements, self.mesh.shape, coordinate, strict=True
        ):
            if isinstance(placement, Replicate) or members == 1:
                continue
            dim = getattr(placement, "dim", None)
            if dim is None or not 0 <= dim < len(shape):
                raise ValueError(f"{placement!r} names no tensor dimension")
            held[dim] = _split(placement, held[dim], members, position)
        return tuple(tuple(segments) for segments in held)
