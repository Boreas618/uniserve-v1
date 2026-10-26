"""Token transports of distributed routed experts.

An expert transport moves each rank's routed tokens to the ranks holding
their experts (``dispatch``), and the experts' weighted partial sums back to
the tokens' ranks (``combine``). Every rank of the expert group calls the
same layers in the same order in every step, including ranks without tokens
of their own; a step's per-rank token ``capacity`` is fixed by the caller
before the first call (:class:`uniserve.bindings.moe.ExpertStep`).

Transports:

* ``alltoall``: FlashInfer's NVLink all-to-all (:mod:`.alltoall`).
* ``deepep``: DeepEP elastic dispatch and combine (:mod:`.deepep`), for
  ordinary or source/expert (disaggregated) placement.
* ``megamoe``: the fused MegaMoE kernel, which exchanges, runs the experts
  and combines in one launch per layer (``backends.moe.megamoe``); with
  dedicated expert ranks, split MegaMoE over DeepGEMM (:mod:`.split`).

A transport bootstraps its own device communicators (NVLink workspace
handles, NCCL and NVSHMEM unique ids) with host collectives supplied by the
caller: the serving runtime's host group (:class:`HostBootstrap`) or, for
direct library use, a ``torch.distributed`` group (:class:`TorchBootstrap`).
Both expose FlashInfer's ``CommBackend`` protocol.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping
from typing import Any

import torch

from uniserve.distributed import Communicator
from uniserve.nn.moe import Activation

__all__ = [
    "ExpertTransport",
    "HostBootstrap",
    "TorchBootstrap",
    "create",
]


class HostBootstrap:
    """Host collectives over the serving runtime's host group.

    The worker constructs this over its host functions. Objects travel
    pickled. Every member calls the same collectives in the same order.

    Args:
        rank: This member's index in the group.
        size: The group's member count.
        all_gather: ``(bytes) -> [bytes]``, every member's bytes in member
            order.
        broadcast: ``(bytes | None, root) -> bytes``, the root's bytes.
        barrier: ``()`` returns once every member has entered it.
        split: ``(color, key)`` returns the arguments of the
            ``HostBootstrap`` over the members of equal ``color``, in this
            order.
        all_gather_fixed: ``(bytes) -> bytes``, every member's record of
            one width back to back in member order; the worker's expert
            agreement reads it, and a split passes it on.
    """

    def __init__(
        self,
        rank: int,
        size: int,
        all_gather: Callable[..., Any],
        broadcast: Callable[..., Any],
        barrier: Callable[..., Any],
        split: Callable[..., Any],
        all_gather_fixed: Callable[..., Any],
    ) -> None:
        self._rank = int(rank)
        self._size = int(size)
        self._all_gather = all_gather
        self._broadcast = broadcast
        self._barrier = barrier
        self._split = split
        self._all_gather_fixed = all_gather_fixed

    def Get_rank(self) -> int:  # noqa: N802 - FlashInfer's CommBackend
        return self._rank

    def Get_size(self) -> int:  # noqa: N802 - FlashInfer's CommBackend
        return self._size

    def allgather(self, data: Any) -> tuple[Any, ...]:
        """Every member's object, in member order."""
        values = self._all_gather(pickle.dumps(data))
        return tuple(pickle.loads(bytes(value)) for value in values)

    def bcast(self, data: Any, root: int) -> Any:
        """``root``'s object on every member; others' ``data`` is ignored."""
        payload = pickle.dumps(data) if self._rank == root else None
        return pickle.loads(bytes(self._broadcast(payload, int(root))))

    def barrier(self) -> None:
        self._barrier()

    def Split(self, color: int, key: int) -> HostBootstrap:  # noqa: N802
        """Members of equal ``color`` ordered by ``(key, member)``."""
        return HostBootstrap(*self._split(int(color), int(key)))


class TorchBootstrap:
    """Host collectives over a ``torch.distributed`` group, for library use.

    The group should use a CPU-capable backend (Gloo); objects travel through
    ``all_gather_object`` and ``broadcast_object_list``.
    """

    def __init__(self, group) -> None:
        import torch.distributed as dist

        self._dist = dist
        self.group = group

    def Get_rank(self) -> int:  # noqa: N802 - FlashInfer's CommBackend
        return self._dist.get_rank(self.group)

    def Get_size(self) -> int:  # noqa: N802 - FlashInfer's CommBackend
        return self._dist.get_world_size(self.group)

    def allgather(self, data: Any) -> tuple[Any, ...]:
        values: list[Any] = [None] * self.Get_size()
        self._dist.all_gather_object(values, data, group=self.group)
        return tuple(values)

    def bcast(self, data: Any, root: int) -> Any:
        values = [data]
        self._dist.broadcast_object_list(
            values,
            src=self._dist.get_global_rank(self.group, root),
            group=self.group,
        )
        return values[0]

    def barrier(self) -> None:
        self._dist.barrier(group=self.group)

    def Split(self, color: int, key: int) -> TorchBootstrap:  # noqa: N802
        members = self.allgather((color, key, self._dist.get_rank()))
        groups = {}
        for value in sorted({member[0] for member in members}):
            ordered = sorted((k, r) for c, k, r in members if c == value)
            # Every member creates every group, in color order, as
            # ``new_group`` requires.
            groups[value] = self._dist.new_group([r for _, r in ordered])
        return TorchBootstrap(groups[color])


class ExpertTransport:
    """The contract every token transport implements.

    Attributes:
        transport: The transport name (``alltoall``, ``deepep``,
            ``megamoe``).
        group: The expert group's communicator; its rank and size order the
            union's members.
        max_tokens: Tokens one rank sends per step at most.
        attention_ranks: Leading source-only members; zero when every member
            holds experts.
        source_only: This member sends tokens but holds no experts.
        capacity: The open step's per-rank capacity; zero outside a step.
        ready: The first call's host barrier has passed (or none is needed).
        fused: The fused MegaMoE buffer, for the ``megamoe`` transport.
    """

    transport: str
    fused: Any = None

    def __init__(
        self,
        group: Communicator,
        bootstrap,
        *,
        max_tokens: int,
        attention_ranks: int,
        barrier_first: bool,
        yield_turn: Callable[[], None] | None,
    ) -> None:
        self.group = group
        self.bootstrap = bootstrap
        self.max_tokens = max_tokens
        self.attention_ranks = attention_ranks
        self.source_only = 0 <= group.rank < attention_ranks
        self.capacity = 0
        self.ready = not barrier_first
        self._yield_turn = yield_turn

    def barrier(self) -> None:
        """Wait for every member of the group (host collective)."""
        self.bootstrap.barrier()

    def yield_turn(self) -> None:
        """Hand the host turn to the next microbatch after a dispatch.

        Outside microbatch execution this does nothing, so numerical layers
        make the same call in ordinary forwards.
        """
        if self._yield_turn is not None:
            self._yield_turn()

    def dispatch(self, hidden, topk_ids, topk_weights, *, invalid_expert):
        """Send each token to the ranks holding its experts.

        Returns this rank's received ``(hidden, topk_ids, topk_weights)``
        rows; ids of experts this rank does not hold, and of unused rows,
        read ``invalid_expert``. The views stay valid until ``combine``.
        """
        raise NotImplementedError

    def combine(self, partial: torch.Tensor, tokens: int) -> torch.Tensor:
        """Return the summed expert outputs of this rank's ``tokens`` rows.

        ``partial`` holds this rank's weighted expert sums in dispatch's
        received row order.
        """
        raise NotImplementedError

    def bind_microbatches(self, transports) -> None:
        """Bind a shared plan of all microbatches.

        Only transports with one native launch for every microbatch need
        one; the default does nothing.
        """

    def suspend(self) -> bool:
        """Release the device workspace while a standby world parks.

        Collective over the group, while no step runs. A transport whose
        workspace keeps its addresses across a release, so that the graphs
        captured over it stay valid, releases it and returns ``True``; the
        default keeps the workspace and returns ``False``.
        """
        return False

    def restore(self) -> None:
        """Map the workspace :meth:`suspend` released back at its addresses.

        Collective over the group, before any step runs; the default does
        nothing.
        """

    def release(self) -> bool:
        """Free a workspace that cannot keep its addresses, while parked.

        Collective over the group, while no step runs, for a workspace
        :meth:`suspend` kept. A transport whose workspace the captured
        graphs address, and which cannot be mapped back at its addresses,
        frees it and returns ``True``: the graphs that use it are captured
        again after :meth:`rebuild`. The default keeps it and returns
        ``False``.
        """
        return False

    def rebuild(self) -> None:
        """Create the workspace :meth:`release` freed again, collectively.

        The new workspace has the released one's layout at new addresses;
        the graphs that use it are captured again afterwards. The default
        does nothing.
        """

    def close(self) -> None:
        """Release owned resources after every reader retired.

        Collective for transports with device communicators.
        """


def create(
    transport: str,
    group: Communicator,
    bootstrap,
    *,
    max_tokens: int,
    top_k: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation: Activation,
    attention_ranks: int,
    device: torch.device,
    yield_turn: Callable[[], None] | None = None,
    encoding: str | None = None,
    symmetric: Mapping[str, Any] | None = None,
) -> ExpertTransport:
    """Construct one microbatch's transport. Collective over ``group``.

    ``yield_turn`` hands the host turn to the next microbatch after each
    dispatch (``None`` without microbatches). ``encoding`` is the experts'
    activation encoding and ``symmetric`` the peer-mapped storage of split
    MegaMoE (``storage``, an int8 tensor, and ``pointers``, every member's
    address in member order), allocated by the caller with
    ``split.symmetric_bytes`` bytes.

    Raises:
        ValueError: The transport cannot serve this placement.
    """
    options: dict[str, Any] = {
        "max_tokens": max_tokens,
        "attention_ranks": attention_ranks,
        "yield_turn": yield_turn,
    }
    if attention_ranks and transport not in {"deepep", "megamoe"}:
        raise ValueError("disaggregated experts require deepep or megamoe")
    if transport == "alltoall":
        from .alltoall import AllToAll

        return AllToAll(
            group,
            bootstrap,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            device=device,
            **options,
        )
    if transport == "deepep":
        from .deepep import DeepEP

        return DeepEP(
            group,
            bootstrap,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            **options,
        )
    if transport == "megamoe":
        from .megamoe import FusedExperts

        return FusedExperts(
            group,
            bootstrap,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
            encoding=encoding,
            symmetric=symmetric,
            **options,
        )
    raise ValueError(f"unknown expert transport {transport!r}")
