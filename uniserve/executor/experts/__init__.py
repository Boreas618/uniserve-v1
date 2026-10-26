"""This rank's side of an expert union's step protocol.

Ranks of an expert union execute every expert step together: the ranks
whose forward the step runs send their tokens, every other rank joins each
expert layer with no tokens of its own. The protocol's parts:

- :mod:`.agreement`: agreement on each step's source group and per-rank
  token capacity over the union's host group, the warmup announcements
  dedicated expert ranks follow, and the capacities a forward's graphs are
  captured at (:class:`ExpertAgreement`, :class:`Selection`);
- :mod:`.microbatches`: the microbatch threads and the host turn their
  calls pass at expert dispatch points (:class:`MicrobatchThreads`,
  :class:`Rotation`);
- :mod:`.joins`: the captured joins of each step capacity
  (:class:`JoinGraphs`);
- :mod:`.union`: this rank's member, which owns the agreement, the
  exchanges, the microbatch threads and the joins and runs every expert
  step and idle join for the executor (:class:`ExpertUnion`).
"""

from uniserve.executor.experts.agreement import (
    Agreement,
    ExpertAgreement,
    HostGroup,
    Selection,
)
from uniserve.executor.experts.joins import JoinGraph, JoinGraphs, capture_order
from uniserve.executor.experts.microbatches import (
    MicrobatchAborted,
    MicrobatchThreads,
    Rotation,
)
from uniserve.executor.experts.union import ExpertUnion

__all__ = [
    "Agreement",
    "ExpertAgreement",
    "ExpertUnion",
    "HostGroup",
    "JoinGraph",
    "JoinGraphs",
    "MicrobatchAborted",
    "MicrobatchThreads",
    "Rotation",
    "Selection",
    "capture_order",
]
