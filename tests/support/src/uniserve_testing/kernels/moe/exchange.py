"""The expert exchange's module interface, as its checks drive it.

An ``expert_exchange`` module (``crates/ops/src/moe/exchange.rs``) is a native
library of lifecycle entries (``create``, ``step``, ``suspend``, ``restore``,
``release``, ``info``, ``footprint``) and one ``launch``. Its call structures
are mirrored here field for field, each held to the operator crate's generated
header by :func:`uniserve_testing.kernels.packages.assert_abi_layout`.

A check drives one module with one rank per device, each on a thread of the
check's process, which is how a main process holds its ranks: the ranks' host
collectives are a barrier and an all-gather between the threads
(:class:`Channel`), handed to the module as its ``UniserveGroup``
(:func:`group_of`).
"""

import ctypes
import threading

import torch

#: The module's entries, ``uniserve_expert_exchange_<name>``.
ENTRIES = ("create", "step", "suspend", "restore", "release", "info", "launch")


class Lifecycle(ctypes.Structure):
    """``uniserve_exchange_lifecycle``: the transport's lifecycle call."""

    _fields_ = [
        ("transport", ctypes.c_void_p),
        ("group", ctypes.c_void_p),
        ("ranks", ctypes.c_int64),
        ("rank", ctypes.c_int64),
        ("experts", ctypes.c_int64),
        ("hidden", ctypes.c_int64),
        ("max_tokens", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("dtype", ctypes.c_int64),
        ("device", ctypes.c_int64),
        ("intermediate", ctypes.c_int64),
        ("capacity", ctypes.c_int64),
        ("attention_ranks", ctypes.c_int64),
        ("weights", ctypes.c_int64),
        ("shared", ctypes.c_int64),
        ("limit", ctypes.c_float),
        ("error", ctypes.c_char * 512),
    ]


class Info(ctypes.Structure):
    """``uniserve_exchange_info``: the transport's views, as ``info`` answers
    them.
    """  # noqa: D205

    _fields_ = [
        ("transport", ctypes.c_void_p),
        ("states", ctypes.c_void_p),
        ("ids", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("combined", ctypes.c_void_p),
        ("live", ctypes.c_void_p),
        ("packed", ctypes.c_void_p),
        ("packed_scales", ctypes.c_void_p),
        ("ranks", ctypes.c_int64),
        ("rank", ctypes.c_int64),
        ("capacity", ctypes.c_int64),
        ("hidden", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("dtype", ctypes.c_int64),
        ("device", ctypes.c_int64),
        ("fused", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


class Call(ctypes.Structure):
    """``uniserve_exchange_call``: one launch."""

    _fields_ = [
        ("transport", ctypes.c_void_p),
        ("states", ctypes.c_void_p),
        ("ids", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("live", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("layer", ctypes.c_void_p),
        ("mode", ctypes.c_int64),
        ("tokens", ctypes.c_int64),
        ("capacity", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("hidden", ctypes.c_int64),
        ("experts", ctypes.c_int64),
        ("max_tokens", ctypes.c_int64),
        ("ranks", ctypes.c_int64),
        ("dtype", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


class FusedLayer(ctypes.Structure):
    """``uniserve_fused_layer``: what a fused launch reads of one layer."""

    _fields_ = [
        ("up_gate", ctypes.c_void_p),
        ("up_gate_scales", ctypes.c_void_p),
        ("down", ctypes.c_void_p),
        ("down_scales", ctypes.c_void_p),
        ("up_gate_alpha", ctypes.c_void_p),
        ("down_alpha", ctypes.c_void_p),
        ("gated_scale", ctypes.c_void_p),
        ("up_gate_scale_bytes", ctypes.c_int64),
        ("down_scale_bytes", ctypes.c_int64),
        ("shared_up_gate", ctypes.c_void_p),
        ("shared_up_gate_scales", ctypes.c_void_p),
        ("shared_down", ctypes.c_void_p),
        ("shared_down_scales", ctypes.c_void_p),
    ]


GATHER = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
)
BARRIER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)


class Group(ctypes.Structure):
    """``UniserveGroup``: the host collectives a rank hands the transport."""

    _fields_ = [
        ("handle", ctypes.c_void_p),
        ("communicator", ctypes.c_void_p),
        ("ranks", ctypes.c_int64),
        ("rank", ctypes.c_int64),
        ("device", ctypes.c_int64),
        ("all_gather", GATHER),
        ("barrier", BARRIER),
        ("error", ctypes.c_char * 512),
    ]


#: The mirrors and the generated header structures they are held to.
MIRRORS = {
    "uniserve_exchange_lifecycle": Lifecycle,
    "uniserve_exchange_info": Info,
    "uniserve_exchange_call": Call,
    "uniserve_fused_layer": FusedLayer,
    "UniserveGroup": Group,
}


class Channel:
    """The host collectives of a group whose ranks are threads of this
    process.
    """  # noqa: D205

    def __init__(self, ranks):
        """A meeting point for ``ranks`` threads."""
        self.meeting = threading.Barrier(ranks)
        self.slots = [b""] * ranks

    def gather(self, rank, mine, peers, count):
        """Every rank's ``count`` bytes at ``mine``, in rank order at
        ``peers``.
        """  # noqa: D205
        self.slots[rank] = ctypes.string_at(mine, count)
        self.meeting.wait()
        for index, value in enumerate(self.slots):
            ctypes.memmove(peers + index * count, value, count)
        self.meeting.wait()
        return 0

    def barrier(self):
        """Wait for every rank."""
        self.meeting.wait()
        return 0


def group_of(channel, ranks, rank, device):
    """Rank ``rank``'s ``UniserveGroup`` over ``channel``, with the callbacks
    it holds; the caller keeps the answer alive while the module may call it.
    """  # noqa: D205
    gather = GATHER(
        lambda _, mine, peers, count: channel.gather(rank, mine, peers, count)
    )
    meet = BARRIER(lambda _: channel.barrier())
    group = Group(
        ranks=ranks, rank=rank, device=device, all_gather=gather, barrier=meet
    )
    return group, (gather, meet)


def message(call):
    """The error a call's entry wrote."""
    return bytes(call.error).split(b"\0", 1)[0].decode(errors="replace")


def tensor_at(address, shape, dtype, device):
    """A tensor over ``address``, a device pointer this rank maps."""
    size = (
        torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    )

    class Holder:
        __cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (address, False),
            "version": 2,
        }

    return torch.as_tensor(Holder(), device=device).view(dtype).view(*shape)


def entered(library, name, call, stream=None):
    """Call ``uniserve_expert_exchange_<name>``, refusing a nonzero status."""
    status = getattr(library, f"uniserve_expert_exchange_{name}")(
        ctypes.byref(call), None if stream is None else ctypes.c_void_p(stream)
    )
    assert status == 0, f"{name}: {message(call)}"


def load(path):
    """The module at ``path`` with its entries' signatures stated."""
    library = ctypes.CDLL(str(path))
    for name in ENTRIES:
        function = getattr(library, f"uniserve_expert_exchange_{name}")
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_int32
    return library


def run_ranks(channel, ranks, body):
    """Run ``body(rank)`` for every rank on a thread of its own; the failures,
    each naming its rank. A failing rank aborts ``channel``'s meetings, since
    the other ranks may be waiting on it.
    """  # noqa: D205
    failures = []

    def guarded(rank):
        try:
            body(rank)
        except BaseException as error:  # noqa: BLE001 - any failure is the result
            failures.append(f"rank {rank}: {error!r}")
            channel.meeting.abort()

    threads = [
        threading.Thread(target=guarded, args=(rank,)) for rank in range(ranks)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures
