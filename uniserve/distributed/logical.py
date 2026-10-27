"""The logical tensors a rank's parameters are parts of.

A parameter that a parallel configuration partitions is one local part of
a logical tensor: the whole matrix the checkpoint names. Which part a rank
holds follows from the tensor's distribution and the rank's mesh
coordinate (:meth:`uniserve.distributed.Distribution.segments`). A module
that partitions a parameter declares, through ``logical_tensors()``, for
each of its parameters:

- the logical tensor's shape and what each dimension indexes;
- the partition unit of each dimension: how many consecutive indices any
  partition keeps together (a head's channels, one expert), or zero for a
  dimension no partition divides;
- the segments of each dimension the rank holds.

A parameter its module does not declare is whole: the local tensor is the
logical tensor. The runtime reads these declarations to key what a rank
holds by the region it is of the logical tensor, and to compute the copies
that redistribute it under another parallel configuration; the model
library itself allocates and moves nothing.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product

from torch import nn

from .distribution import Distribution, Segment, Span


def _clip(segments: tuple[Segment, ...], extent: int) -> tuple[Segment, ...]:
    """The segments' indices below ``extent``; the local positions of the
    indices past it are padding.
    """  # noqa: D205
    clipped = []
    for segment in segments:
        span = segment.span
        if span.start >= extent:
            continue
        count = min(
            span.count, (extent - span.start + span.step - 1) // span.step
        )
        clipped.append(
            Segment(Span(span.start, count, span.step), segment.local)
        )
    return tuple(clipped)


@dataclass(frozen=True, slots=True)
class Logical:
    """One local tensor's part of a logical tensor.

    ``shape`` is the logical tensor's shape and ``dims`` names what each of
    its dimensions indexes. ``units[d]`` is the number of consecutive
    indices of dimension ``d`` that every partition keeps together, zero
    when no partition divides it. ``segments[d]`` are the logical indices
    of dimension ``d`` the local tensor holds and their local positions;
    the local tensor holds the Cartesian product of its dimensions'
    segments. Local positions no segment covers are padding, which holds
    zeros.
    """

    shape: tuple[int, ...]
    dims: tuple[str, ...]
    units: tuple[int, ...]
    segments: tuple[tuple[Segment, ...], ...]

    def __post_init__(self) -> None:
        rank = len(self.shape)
        if len(self.dims) != rank or len(self.units) != rank:
            raise ValueError("a logical tensor names and units every dimension")
        if len(self.segments) != rank:
            raise ValueError(
                "a logical part has segments along every dimension"
            )
        if any(unit < 0 for unit in self.units):
            raise ValueError("partition units are nonnegative")

    @classmethod
    def whole(
        cls,
        shape: tuple[int, ...],
        dims: tuple[str, ...] | None = None,
        units: tuple[int, ...] | None = None,
    ) -> Logical:
        """The whole logical tensor of ``shape``, held at its own positions."""
        return cls(
            tuple(shape),
            tuple(dims or (f"dim{index}" for index in range(len(shape)))),
            tuple(units or (0,) * len(shape)),
            tuple((Segment(Span(0, extent), 0),) for extent in shape),
        )

    @classmethod
    def distributed(
        cls,
        distribution: Distribution,
        shape: tuple[int, ...],
        dims: tuple[str, ...],
        units: tuple[int, ...],
        *,
        domain: tuple[int, ...] | None = None,
    ) -> Logical:
        """The part of a logical tensor of ``shape`` that ``distribution``
        gives its mesh's own rank.

        ``domain`` is the padded extent the distribution divides when it is
        larger than the logical tensor (a vocabulary padded to a multiple of
        its partition): the local positions of indices past ``shape`` are
        padding.
        """  # noqa: D205
        divided = distribution.segments(tuple(domain or shape))
        return cls(
            tuple(shape),
            tuple(dims),
            tuple(units),
            tuple(
                _clip(segments, extent)
                for segments, extent in zip(divided, shape, strict=True)
            ),
        )

    def blocks(self) -> Iterator[tuple[tuple[Span, ...], tuple[int, ...]]]:
        """Every box the local tensor holds: its logical spans and its
        first local position, one per dimension.
        """  # noqa: D205
        for combination in product(*self.segments):
            yield (
                tuple(segment.span for segment in combination),
                tuple(segment.local for segment in combination),
            )

    def describe(self) -> dict:
        """A JSON-encodable form, as the runtime reads it
        (``storage::manifest::Logical``).
        """  # noqa: D205
        return {
            "shape": list(self.shape),
            "dims": list(self.dims),
            "units": list(self.units),
            "segments": [
                [
                    {
                        "span": {
                            "start": segment.span.start,
                            "count": segment.span.count,
                            "step": segment.span.step,
                        },
                        "local": segment.local,
                    }
                    for segment in segments
                ]
                for segments in self.segments
            ],
        }


def logical_tensors(module: nn.Module) -> dict[int, Logical]:
    """The declared logical tensor of every parameter and buffer under
    ``module``, by tensor object identity.

    Each submodule that defines ``logical_tensors()`` declares its own
    parameters, by their names relative to it (``"weight"``, or
    ``"up_gate.weight"`` for a child's); the first declaration of a tensor
    wins. Tensors no module declares are left out: they are whole.
    """  # noqa: D205
    declared: dict[int, Logical] = {}
    for child in module.modules():
        declare = getattr(child, "logical_tensors", None)
        if declare is None:
            continue
        parts: Mapping[str, Logical] = declare()
        for name, logical in parts.items():
            tensor = _tensor(child, name)
            if tensor is not None:
                declared.setdefault(id(tensor), logical)
    return declared


def _tensor(module: nn.Module, name: str):
    """The parameter or buffer ``name`` (a dotted path) of ``module``."""
    owner_path, _, leaf = name.rpartition(".")
    owner = module.get_submodule(owner_path) if owner_path else module
    if leaf in owner._parameters:
        return owner._parameters[leaf]
    return owner._buffers.get(leaf)
