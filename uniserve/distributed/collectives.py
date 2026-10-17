"""Backend collectives behind the logical model communicators.

Numerical code calls :class:`uniserve.distributed.Communicator`,
which orders logical members. This module resolves each call to the
stream-bound collectives selected by the active execution scope (the
serving worker's NCCL communicators on the binding's stream) or, without
one, to the initialized process group of direct library use, and owns
asynchronous completion. Torch custom ops keep the process-group name
traceable through captured graphs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from typing import Any, Protocol

import torch
import torch.distributed as dist

from uniserve.profiling import profile_range, ranges_active


class StreamCollectives(Protocol):
    """Ordered collectives for one computation binding.

    Ordered collectives and deferred transfers for one computation binding.
    """

    def all_reduce(self, value: torch.Tensor, op: str = "sum") -> None: ...
    def all_gather(self, output: torch.Tensor, value: torch.Tensor) -> None: ...
    def start_all_gather(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> Any: ...
    def start_all_to_all(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> Any: ...
    def all_to_all(
        self,
        output: torch.Tensor,
        value: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> None: ...
    def gather(
        self, outputs: list[torch.Tensor] | None, value: torch.Tensor, root: int
    ) -> None: ...
    def broadcast(self, value: torch.Tensor, root: int) -> None: ...
    def reduce_scatter(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> None: ...

    def send(self, value: torch.Tensor, peer: int) -> None: ...
    def recv(self, value: torch.Tensor, peer: int) -> None: ...
    def send_recv(
        self, output: torch.Tensor, value: torch.Tensor, dst: int, src: int
    ) -> None: ...

    def close(self) -> None: ...


# The communication resources of the running numerical invocation, by
# communicator name; an execution context installs its ``collectives`` here
# for each activation. ``None`` selects initialized process-group
# communication on an ordinary device stream. A mapping selects explicit
# stream bindings and must cover every used communicator, including when it
# is empty.
stream_bindings: ContextVar[Mapping[str, StreamCollectives] | None] = (
    ContextVar("stream_collectives", default=None)
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


def stream_collectives(group_name: str) -> StreamCollectives | None:
    """Resolve a stream provider.

    Resolve a stream provider, rejecting incomplete explicit execution
    bindings.

    An unscoped numerical caller may use the initialized process group. Once a
    caller selects an explicit stream scope, every used group must belong to it;
    falling back would enqueue communication on a different execution stream.
    """
    bindings = stream_bindings.get()
    if bindings is None:
        return None
    try:
        return bindings[group_name]
    except KeyError:
        raise RuntimeError(
            f"stream scope has no binding for communicator {group_name!r}"
        ) from None


def process_group(name: str):
    """Resolve a process group from its stable backend name.

    A name is traceable through custom ops; process-group objects cannot
    cross a torch.library schema or a captured graph boundary.
    """
    return dist.distributed_c10d._resolve_process_group(
        dist.distributed_c10d.GroupName(name)
    )


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
        bound = stream_collectives(group.group_name)
        if bound is not None:
            work = bound.start_all_gather(output, input)
        else:
            work = dist.all_gather_into_tensor(
                output, input, group=group, async_op=True
            )
    return Transfer(work, input.device)


def start_all_to_all(
    output: torch.Tensor, input: torch.Tensor, group
) -> Transfer:
    """Publish an equal-split exchange in backend member order."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        work = bound.start_all_to_all(output, input)
    else:
        work = dist.all_to_all_single(output, input, group=group, async_op=True)
    return Transfer(work, input.device)


@torch.library.custom_op(
    "uniserve::all_gather_into_tensor", mutates_args=("output",)
)
def all_gather_into_tensor(
    output: torch.Tensor, input: torch.Tensor, group_name: str
) -> None:
    """Gather equal contributions into ``output`` in backend rank order."""
    with _launch_range("all_gather", group_name):
        group = process_group(group_name)
        # The gather runs in place: this rank's block of ``output`` receives
        # its contribution first. ``output`` is never a registered NCCL
        # window, so NCCL gathers with its kernel algorithms.
        local = output.view(-1).narrow(
            0, group.rank() * input.numel(), input.numel()
        )
        local = local.view_as(input)
        local.copy_(input)
        bound = stream_collectives(group_name)
        if bound is not None:
            # Immediate consumers need no fork onto the transport stream.
            bound.all_gather(output, local)
        else:
            start_all_gather(output, local, group).wait()


@all_gather_into_tensor.register_fake
def _all_gather_into_tensor_fake(output, input, group_name):
    pass


@torch.library.custom_op(
    "uniserve::all_to_all_single_into", mutates_args=("output",)
)
def all_to_all_single_into(
    output: torch.Tensor,
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group_name: str,
) -> None:
    """Exchange leading-axis partitions in backend rank order."""
    with _launch_range("all_to_all", group_name):
        bound = stream_collectives(group_name)
        if bound is not None:
            bound.all_to_all(output, input, output_splits, input_splits)
            return
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
            group=process_group(group_name),
            async_op=True,
        )
        Transfer(work, input.device).wait()


@all_to_all_single_into.register_fake
def _all_to_all_single_into_fake(
    output, input, output_splits, input_splits, group_name
):
    pass


@torch.library.custom_op("uniserve::all_reduce_max", mutates_args=("value",))
def all_reduce_max(value: torch.Tensor, group_name: str) -> None:
    """Reduce ``value`` in place to its elementwise maximum."""
    with _launch_range("all_reduce_max", group_name):
        bound = stream_collectives(group_name)
        if bound is not None:
            bound.all_reduce(value, "max")
            return
        work = dist.all_reduce(
            value,
            op=dist.ReduceOp.MAX,
            group=process_group(group_name),
            async_op=True,
        )
        Transfer(work, value.device).wait()


@all_reduce_max.register_fake
def _all_reduce_max_fake(value, group_name):
    pass


@torch.library.custom_op("uniserve::group_send_recv", mutates_args=("output",))
def send_recv(
    value: torch.Tensor,
    output: torch.Tensor,
    dst: int,
    src: int,
    group_name: str,
) -> None:
    """Send ``value`` bytes to ``dst`` and receive ``output`` from ``src``."""
    group = process_group(group_name)
    with _launch_range("send_recv", group_name):
        bound = stream_collectives(group_name)
        if bound is not None:
            bound.send_recv(output, value, dst, src)
            return
        calls = [
            dist.P2POp(
                dist.isend, value.reshape(-1).view(torch.uint8), dst, group
            ),
            dist.P2POp(
                dist.irecv, output.reshape(-1).view(torch.uint8), src, group
            ),
        ]
        for work in dist.batch_isend_irecv(calls):
            Transfer(work, value.device).wait()


@send_recv.register_fake
def _send_recv_fake(value, output, dst, src, group_name):
    pass


_REDUCTIONS = {
    "sum": dist.ReduceOp.SUM,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}


def all_reduce(value: torch.Tensor, op: str, group) -> None:
    """Reduce ``value`` in place over ``group``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.all_reduce(value, op)
    elif op == "max":
        all_reduce_max(value, group.group_name)
    else:
        dist.all_reduce(value, op=_REDUCTIONS[op], group=group)


def all_gather(storage: torch.Tensor, value: torch.Tensor, group) -> None:
    """Gather ``value`` into the leading backend-ordered axis of ``storage``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.all_gather(storage, value)
    else:
        dist.all_gather(list(storage.unbind(0)), value, group=group)


def gather(
    gather_list: list[torch.Tensor] | None,
    value: torch.Tensor,
    dst: int,
    group,
) -> None:
    """Gather ``value`` into ``gather_list`` on global rank ``dst``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.gather(gather_list, value, dst)
        return
    work = dist.gather(
        value, gather_list=gather_list, dst=dst, group=group, async_op=True
    )
    Transfer(work, value.device).wait()


def broadcast(value: torch.Tensor, src: int, group) -> None:
    """Broadcast ``value`` in place from global rank ``src``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.broadcast(value, src)
    else:
        dist.broadcast(value, src=src, group=group)


def reduce_scatter(output: torch.Tensor, value: torch.Tensor, group) -> None:
    """Sum backend-ordered partitions of ``value`` into ``output``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.reduce_scatter(output, value)
    else:
        dist.reduce_scatter_tensor(output, value, group=group)


def send(payload: torch.Tensor, dst: int, group) -> None:
    """Send contiguous bytes to global rank ``dst``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.send(payload, dst)
    else:
        dist.send(payload, dst=dst, group=group)


def recv(payload: torch.Tensor, src: int, group) -> None:
    """Receive contiguous bytes from global rank ``src``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.recv(payload, src)
    else:
        dist.recv(payload, src=src, group=group)


class _Transfer:
    """A started native collective; its consumer joins it on its stream."""

    def __init__(self, wait: Callable[..., Any], token: int) -> None:
        self._wait = wait
        # Cleared once the consumer has joined the collective.
        self._token: int | None = token

    def block_current_stream(self) -> None:
        token, self._token = self._token, None
        if token is not None:
            self._wait(token, torch.cuda.current_stream().cuda_stream)


class NativeCollectives:
    """One group's collectives on the worker's NCCL communicator.

    The worker binds one communicator per stream and group and constructs
    this object over its functions; every call enqueues on that stream
    (asynchronous ones on its transfer fork, joined by the consumer).
    Members are ordered by ascending world rank, the backend order
    ``Communicator`` assumes; roots and peers are world ranks.

    Args:
        name: The communicator's name: its mesh axes joined by ``.``.
        all_reduce: ``(value, op)`` with ``op`` 0 for sum, 2 for max and 3
            for min.
        all_gather, reduce_scatter: ``(output, value)``.
        start_all_gather, start_all_to_all: ``(output, value)``, returning
            the token ``wait`` joins.
        wait: ``(token, stream)`` joins a started collective on the CUDA
            stream handle ``stream``.
        all_to_all: ``(output, value, output_splits, input_splits)``.
        gather: ``(outputs, value, root)``.
        broadcast: ``(value, root)``.
        send, recv: ``(value, peer)``.
        send_recv: ``(output, value, destination, source)``.
    """

    _REDUCTIONS = {"sum": 0, "max": 2, "min": 3}

    def __init__(
        self,
        name: str,
        all_reduce: Callable[..., Any],
        all_gather: Callable[..., Any],
        start_all_gather: Callable[..., Any],
        start_all_to_all: Callable[..., Any],
        wait: Callable[..., Any],
        all_to_all: Callable[..., Any],
        gather: Callable[..., Any],
        broadcast: Callable[..., Any],
        reduce_scatter: Callable[..., Any],
        send: Callable[..., Any],
        recv: Callable[..., Any],
        send_recv: Callable[..., Any],
    ) -> None:
        self.name = str(name)
        self._all_reduce = all_reduce
        self._all_gather = all_gather
        self._start_all_gather = start_all_gather
        self._start_all_to_all = start_all_to_all
        self._wait = wait
        self._all_to_all = all_to_all
        self._gather = gather
        self._broadcast = broadcast
        self._reduce_scatter = reduce_scatter
        self._send = send
        self._recv = recv
        self._send_recv = send_recv

    def all_reduce(self, value: torch.Tensor, op: str = "sum") -> None:
        self._all_reduce(value, self._REDUCTIONS[op])

    def all_gather(self, output: torch.Tensor, value: torch.Tensor) -> None:
        self._all_gather(output, value)

    def start_all_gather(self, output, value) -> _Transfer:
        return _Transfer(self._wait, self._start_all_gather(output, value))

    def start_all_to_all(self, output, value) -> _Transfer:
        return _Transfer(self._wait, self._start_all_to_all(output, value))

    def all_to_all(self, output, value, output_splits, input_splits) -> None:
        self._all_to_all(output, value, list(output_splits), list(input_splits))

    def gather(self, outputs, value, root: int) -> None:
        self._gather(list(outputs or ()), value, int(root))

    def broadcast(self, value: torch.Tensor, root: int) -> None:
        self._broadcast(value, int(root))

    def reduce_scatter(self, output, value) -> None:
        self._reduce_scatter(output, value)

    def send(self, value: torch.Tensor, peer: int) -> None:
        self._send(value, int(peer))

    def recv(self, value: torch.Tensor, peer: int) -> None:
        self._recv(value, int(peer))

    def send_recv(self, output, value, dst: int, src: int) -> None:
        self._send_recv(output, value, int(dst), int(src))

    def close(self) -> None:
        """The worker destroys the communicator."""


def group_collectives(groups, collectives: Sequence[NativeCollectives]) -> dict:
    """The stream collectives of ``groups`` by group name.

    The result is the ``collectives`` an execution context installs into
    :data:`stream_bindings` for each activation.

    ``collectives`` holds one stream's collectives of each communicator.
    Every multi-member communicator of ``groups`` must have one: within an
    activation, a collective never falls back to another stream.

    Raises:
        ValueError: A communicator has no stream collectives.
    """
    named = {native.name: native for native in collectives}
    bound = {}
    for group in groups:
        if group.size == 1:
            continue
        native = named.get(group.name)
        if native is None:
            raise ValueError(
                f"communicator {group.name!r} has no stream collectives"
            )
        bound[group._require().group_name] = native
    return bound
