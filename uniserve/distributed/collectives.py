"""Backend collectives behind the logical model communicators.

Numerical code calls :class:`uniserve.distributed.Communicator`, which
orders logical members. This module resolves each call to the process group
the active execution scope binds for that communicator (the serving
executor's NCCL group created for the binding's stream) or, without a scope,
to the communicator's own initialized process group, and owns asynchronous
completion. Each function takes the borrowed process group itself: a
captured graph records the collective launches, not dispatcher calls, so no
group name has to survive a graph boundary.

Collectives on one process group run in issue order on that group's NCCL
stream, so a synchronous collective issued after a started one is ordered
behind it without further bookkeeping; a started collective's consumer joins
it with :meth:`Transfer.wait` on its own stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from typing import Any

import torch
import torch.distributed as dist

from uniserve.profiling import profile_range, ranges_active

# The process group each communicator's collectives run on during the running
# numerical invocation, by the logical communicator's process-group name; an
# execution context installs its ``collectives`` here for each activation.
# ``None`` selects each communicator's own initialized process group on the
# current stream. A mapping selects explicit stream bindings and must cover
# every used multi-member communicator, including when it is empty.
stream_bindings: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "stream_groups", default=None
)

_NO_RANGE = nullcontext()


def _launch_range(kind: str, group_name: str) -> AbstractContextManager[None]:
    """Profile one collective launch; the span is named only when emitted.

    The range covers the launch rather than the transfer, which is what a
    profile correlates a device kernel back to.
    """
    if not ranges_active():
        return _NO_RANGE
    return profile_range(
        f"uniserve.collective kind={kind} group={group_name} "
        f"rank={dist.get_rank()}"
    )


def _bound(group):
    """The process group a collective on ``group`` runs on.

    An unscoped numerical caller uses ``group`` itself. Once a caller selects
    an explicit stream scope, every used group must belong to it: falling
    back would enqueue communication on another binding's stream.

    Raises:
        RuntimeError: The active scope binds no group for ``group``.
    """
    bindings = stream_bindings.get()
    if bindings is None:
        return group
    try:
        return bindings[group.group_name]
    except KeyError:
        raise RuntimeError(
            f"stream scope has no binding for communicator {group.group_name!r}"
        ) from None


class Transfer:
    """A started collective whose consumer joins its completion.

    Outputs are readable on the consumer's stream after :meth:`wait`, which
    enqueues the dependency without blocking the host for CUDA transfers.
    """

    def __init__(self, work: Any, device: torch.device) -> None:
        self._work, self._device = work, device

    def wait(self) -> None:
        """Join the transfer; later calls do nothing."""
        work, self._work = self._work, None
        if work is None:
            return
        if self._device.type == "cuda":
            work.block_current_stream()
        else:
            work.wait()


def start_all_gather(
    output: torch.Tensor, input: torch.Tensor, group
) -> Transfer:
    """Publish a gather into backend-ordered ``output`` and return it."""
    with _launch_range("all_gather_start", group.group_name):
        work = dist.all_gather_into_tensor(
            output, input, group=_bound(group), async_op=True
        )
    return Transfer(work, input.device)


def start_all_to_all(
    output: torch.Tensor, input: torch.Tensor, group
) -> Transfer:
    """Publish an equal-split exchange in backend member order."""
    work = dist.all_to_all_single(
        output, input, group=_bound(group), async_op=True
    )
    return Transfer(work, input.device)


class _Works:
    """Join several process-group works as one transfer."""

    def __init__(self, works) -> None:
        self.works = works

    def block_current_stream(self) -> None:
        for work in self.works:
            work.block_current_stream()

    def wait(self) -> None:
        for work in self.works:
            work.wait()


def start_exchange(
    transfers: list[tuple[int, torch.Tensor | None, torch.Tensor | None]],
    group,
    device: torch.device,
) -> Transfer:
    """Send and receive per-peer tensors as one launch and return it.

    Each ``(peer, send, receive)`` names a global rank of ``group`` with the
    contiguous tensor sent to it and the one received from it, either of
    which may be ``None``. The transfers run as one batch of point-to-point
    calls on the bound group.
    """
    with _launch_range("exchange_start", group.group_name):
        bound = _bound(group)
        operations = [
            dist.P2POp(call, value, peer, bound)
            for peer, send, receive in transfers
            for call, value in ((dist.isend, send), (dist.irecv, receive))
            if value is not None and value.numel()
        ]
        work = _Works(dist.batch_isend_irecv(operations) if operations else [])
    return Transfer(work, device)


def all_gather(output: torch.Tensor, input: torch.Tensor, group) -> None:
    """Gather equal contributions into ``output`` in backend rank order.

    ``output`` is contiguous storage of any shape holding one ``input``-sized
    contribution per member, in backend rank order.
    """
    with _launch_range("all_gather", group.group_name):
        # The gather runs in place: this rank's block of ``output`` receives
        # its contribution first. ``output`` is never a registered NCCL
        # window, so NCCL gathers with its kernel algorithms. Both operands
        # are passed flat: Gloo accepts a concatenation of the contributions
        # along the leading axis but not a stack of them.
        flat = output.view(-1)
        local = flat.narrow(0, group.rank() * input.numel(), input.numel())
        local.view_as(input).copy_(input)
        start_all_gather(flat, local, group).wait()


def all_to_all(
    output: torch.Tensor,
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group,
) -> None:
    """Exchange leading-axis partitions in backend rank order."""
    with _launch_range("all_to_all", group.group_name):
        # Empty split lists select the native equal-count collective. Explicit
        # lists select variable-count send/recv, including when counts match.
        equal_counts = (
            input.numel() == output.numel()
            and len(set(output_splits + input_splits)) == 1
        )
        work = dist.all_to_all_single(
            output,
            input,
            output_split_sizes=None if equal_counts else output_splits,
            input_split_sizes=None if equal_counts else input_splits,
            group=_bound(group),
            async_op=True,
        )
        Transfer(work, input.device).wait()


def send_recv(
    value: torch.Tensor,
    output: torch.Tensor,
    dst: int,
    src: int,
    group,
) -> None:
    """Send ``value`` bytes to ``dst`` and receive ``output`` from ``src``."""
    with _launch_range("send_recv", group.group_name):
        bound = _bound(group)
        calls = [
            dist.P2POp(
                dist.isend, value.reshape(-1).view(torch.uint8), dst, bound
            ),
            dist.P2POp(
                dist.irecv, output.reshape(-1).view(torch.uint8), src, bound
            ),
        ]
        for work in dist.batch_isend_irecv(calls):
            Transfer(work, value.device).wait()


_REDUCTIONS = {
    "sum": dist.ReduceOp.SUM,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}


def all_reduce(value: torch.Tensor, op: str, group) -> None:
    """Reduce ``value`` in place over ``group``."""
    dist.all_reduce(value, op=_REDUCTIONS[op], group=_bound(group))


def gather(
    gather_list: list[torch.Tensor] | None,
    value: torch.Tensor,
    dst: int,
    group,
) -> None:
    """Gather ``value`` into ``gather_list`` on global rank ``dst``."""
    work = dist.gather(
        value,
        gather_list=gather_list,
        dst=dst,
        group=_bound(group),
        async_op=True,
    )
    Transfer(work, value.device).wait()


def broadcast(value: torch.Tensor, src: int, group) -> None:
    """Broadcast ``value`` in place from global rank ``src``."""
    dist.broadcast(value, src=src, group=_bound(group))


def reduce_scatter(output: torch.Tensor, value: torch.Tensor, group) -> None:
    """Sum backend-ordered partitions of ``value`` into ``output``."""
    dist.reduce_scatter_tensor(output, value, group=_bound(group))


def send(payload: torch.Tensor, dst: int, group) -> None:
    """Send contiguous bytes to global rank ``dst``."""
    dist.send(payload, dst=dst, group=_bound(group))


def recv(payload: torch.Tensor, src: int, group) -> None:
    """Receive contiguous bytes from global rank ``src``."""
    dist.recv(payload, src=src, group=_bound(group))


def group_collectives(groups, stream_groups: Mapping[str, Any]) -> dict:
    """The stream binding of ``groups``: each communicator's process group.

    The result is the ``collectives`` an execution context installs into
    :data:`stream_bindings` for each activation.

    Args:
        groups: The logical communicators a module calls.
        stream_groups: One stream binding's process group of each
            multi-member communicator, by communicator name (its mesh axes
            joined by ``.``). The serving executor creates them in the
            worker's planned order; each runs its collectives on its own
            NCCL stream behind the binding's current stream.

    Raises:
        ValueError: A multi-member communicator has no stream group: within
            an activation, a collective never falls back to another
            binding's group.
    """
    bound = {}
    for group in groups:
        if group.size == 1:
            continue
        stream_group = stream_groups.get(group.name)
        if stream_group is None:
            raise ValueError(f"communicator {group.name!r} has no stream group")
        bound[group._require().group_name] = stream_group
    return bound
