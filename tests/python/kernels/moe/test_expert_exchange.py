"""The alltoall exchange's whole lifecycle over two real devices.

The module of the descriptor below (resolved at the session's Build Server for
the session's device; one object serves both directions and every extent) is
driven by two ranks, one per device, each on a thread of the test process,
which is how a main process holds its ranks; their host collectives are a
barrier and an all-gather between the threads. Each rank routes random rows
through a dispatch, stands in for its experts by writing each received row's
weighted sum of a per-expert factor into the transport's ``combined`` view,
and compares the combine with the same sum computed in torch, within 2% of the
largest value plus `1e-3`. It also meets what a served step meets: a rank with
no tokens, a step whose capacity is below the transport's, a launch whose live
count is below its extent (whose rows past the count must come back zero), and
a suspend and restore that must keep the views' addresses, since a captured
graph holds them.

It needs two CUDA devices. The call structures' mirrors are held to the
operator crate's generated header, which needs none.
"""

import ctypes
import threading

import pytest
import torch
from uniserve_testing.kernels.packages import (
    abi_enumerator,
    assert_abi_layout,
)

pytestmark = pytest.mark.unit

#: `ops::abi::code`'s number of BF16.
BF16 = abi_enumerator("UNISERVE_DTYPE_BF16")

#: The ranks, one per device, and the exchange's extents: the hidden width,
#: the routes per token, the experts and the transport's rows per rank.
RANKS, HIDDEN, TOP, EXPERTS, CAPACITY = 2, 2048, 8, 128, 64
KEY = (
    f"mode_dispatch_hidden_{HIDDEN}_intermediate_768_top_{TOP}_experts_"
    f"{EXPERTS}_max_tokens_{CAPACITY}_ranks_{RANKS}_attention_0_dtype_bf16"
)

#: The module's entries, `uniserve_expert_exchange_<name>`.
ENTRIES = ("create", "step", "suspend", "restore", "release", "info", "launch")

#: The directions of a launch.
DISPATCH = abi_enumerator("UNISERVE_EXCHANGE_DISPATCH")
COMBINE = abi_enumerator("UNISERVE_EXCHANGE_COMBINE")

#: The trials, each the rows a rank sends and the live count among them: the
#: transport's capacity, a step below it with a live count below its extent,
#: and a step of four rows.
TRIALS = ((CAPACITY, CAPACITY), (CAPACITY // 2, CAPACITY // 4), (4, 3))

#: The combine's allowance: 2% of the largest expected value, plus `1e-3`.
RELATIVE, ABSOLUTE = 0.02, 1e-3


class Lifecycle(ctypes.Structure):
    """The transport's lifecycle call."""

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
        ("error", ctypes.c_char * 512),
    ]


class Info(ctypes.Structure):
    """The transport's views, as its `info` entry answers them."""

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
    """`ops::moe::exchange`'s launch call, field for field."""

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
        ("top", ctypes.c_int64),
        ("hidden", ctypes.c_int64),
        ("experts", ctypes.c_int64),
        ("max_tokens", ctypes.c_int64),
        ("ranks", ctypes.c_int64),
        ("dtype", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
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
    """The host collectives a rank hands the transport."""

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


@pytest.mark.parametrize(
    ("mirror", "structure"),
    [
        pytest.param(Lifecycle, "uniserve_exchange_lifecycle", id="lifecycle"),
        pytest.param(Info, "uniserve_exchange_info", id="info"),
        pytest.param(Call, "uniserve_exchange_call", id="launch"),
        pytest.param(Group, "UniserveGroup", id="group"),
    ],
)
def test_a_call_mirror_is_its_generated_header_structure(mirror, structure):
    """The generated header asserts the size and offsets the mirror has."""
    assert_abi_layout(mirror, structure)


class Channel:
    """The host collectives of a group whose ranks are threads of this
    process.
    """  # noqa: D205

    def __init__(self, ranks):
        """A meeting point for `ranks` threads."""
        self.meeting = threading.Barrier(ranks)
        self.slots = [b""] * ranks

    def gather(self, rank, mine, peers, count):
        """Every rank's `count` bytes at `mine`, in rank order at `peers`."""
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


def message(call):
    """The error a call's entry wrote."""
    return bytes(call.error).split(b"\0", 1)[0].decode(errors="replace")


def tensor_at(address, shape, dtype, device):
    """A tensor over `address`, a device pointer this rank maps."""
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
    """Call `uniserve_expert_exchange_<name>`, refusing a nonzero status."""
    status = getattr(library, f"uniserve_expert_exchange_{name}")(
        ctypes.byref(call), None if stream is None else ctypes.c_void_p(stream)
    )
    assert status == 0, f"{name}: {message(call)}"


def run_rank(library, channel, rank, device):
    """One rank's lifecycle: create, the trials, suspend, restore, release."""
    torch.cuda.set_device(device)
    # The primary context is current on this thread.
    torch.zeros(1, device=device)
    stream = torch.cuda.Stream(device=device)
    gather = GATHER(
        lambda _, mine, peers, count: channel.gather(rank, mine, peers, count)
    )
    meet = BARRIER(lambda _: channel.barrier())
    group = Group(
        ranks=RANKS, rank=rank, device=device, all_gather=gather, barrier=meet
    )
    life = Lifecycle(
        group=ctypes.cast(ctypes.pointer(group), ctypes.c_void_p),
        ranks=RANKS,
        rank=rank,
        experts=EXPERTS,
        hidden=HIDDEN,
        max_tokens=CAPACITY,
        top=TOP,
        dtype=BF16,
        device=device,
        attention_ranks=0,
    )
    entered(library, "create", life)
    info = Info(transport=life.transport)
    entered(library, "info", info)
    rows = RANKS * CAPACITY
    states = tensor_at(info.states, (rows, HIDDEN), torch.bfloat16, device)
    ids = tensor_at(info.ids, (rows, TOP), torch.int32, device)
    weights = tensor_at(info.weights, (rows, TOP), torch.float32, device)
    combined = tensor_at(info.combined, (rows, HIDDEN), torch.bfloat16, device)
    live_view = tensor_at(info.live, (1,), torch.int32, device)
    local = EXPERTS // RANKS
    factor = (
        torch.arange(1, EXPERTS + 1, device=device, dtype=torch.float32)
        / EXPERTS
    )

    for trial, (extent, live) in enumerate(TRIALS):
        # Ranks send different counts, and the last rank nothing on the first
        # trial: the agreed step is the largest count, and a rank with no
        # tokens still joins.
        silent = trial == 0 and rank == RANKS - 1
        tokens = 0 if silent else extent
        count = 0 if silent else max(live - rank, 1)
        generator = torch.Generator().manual_seed(1000 * trial + rank)
        rows_sent = max(tokens, 1)
        hidden_rows = torch.randn(rows_sent, HIDDEN, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        scores = torch.rand(rows_sent, EXPERTS, generator=generator)
        route = torch.topk(scores, TOP, dim=1).indices.to(
            device=device, dtype=torch.int32
        )
        route_weights = torch.rand(rows_sent, TOP, generator=generator).to(
            device
        )
        live_word = torch.tensor([count], dtype=torch.int32, device=device)
        out = torch.full(
            (rows_sent, HIDDEN),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        with torch.cuda.stream(stream):
            stream.wait_stream(torch.cuda.current_stream(device))
            life.capacity = max(live, 1)
            entered(library, "step", life, stream.cuda_stream)
            call = Call(
                transport=life.transport,
                states=hidden_rows.data_ptr(),
                ids=route.data_ptr(),
                weights=route_weights.data_ptr(),
                live=live_word.data_ptr(),
                out=out.data_ptr(),
                mode=DISPATCH,
                tokens=tokens,
                top=TOP,
                hidden=HIDDEN,
                experts=EXPERTS,
                max_tokens=CAPACITY,
                ranks=RANKS,
                dtype=BF16,
            )
            entered(library, "launch", call, stream.cuda_stream)
            stream.synchronize()
            # Stand in for the experts: each received row sums weight · factor
            # over the routes to this rank's experts, times the row.
            received_rows = int(live_view[0].item())
            received = states[:received_rows].float()
            routes = ids[:received_rows].long()
            mine = (routes >= rank * local) & (routes < (rank + 1) * local)
            taken = torch.where(
                mine,
                weights[:received_rows]
                * factor[routes.clamp(min=0, max=EXPERTS - 1)],
                0.0,
            )
            combined[:received_rows] = (
                received * taken.sum(dim=1, keepdim=True)
            ).to(torch.bfloat16)
            stream.synchronize()
            call.mode = COMBINE
            entered(library, "launch", call, stream.cuda_stream)
            stream.synchronize()
        if tokens:
            expected = hidden_rows[:count].float() * (
                route_weights[:count] * factor[route[:count].long()]
            ).sum(dim=1, keepdim=True)
            error = (out[:count].float() - expected).abs().max().item()
            bound = RELATIVE * expected.abs().max().item() + ABSOLUTE
            assert error <= bound, (
                f"trial {trial}: the combine differs by {error}, bound {bound}"
            )
            assert bool((out[count:tokens] == 0).all()), (
                f"trial {trial}: a row past the live count {count} of "
                f"{tokens} is not zero"
            )

    before = (info.states, info.combined)
    # A backend that cannot suspend refuses the call, and keeps its storage.
    suspended = library.uniserve_expert_exchange_suspend(
        ctypes.byref(life), None
    )
    if suspended == 0:
        entered(library, "restore", life)
        after = Info(transport=life.transport)
        entered(library, "info", after)
        assert (after.states, after.combined) == before, (
            "restore moved the views: a captured graph addresses the old ones"
        )
    entered(library, "release", life)


@pytest.mark.gpu
def test_two_ranks_dispatch_combine_suspend_and_restore(kernel_module):
    if torch.cuda.device_count() < RANKS:
        pytest.skip(
            f"needs {RANKS} CUDA devices, this host shows "
            f"{torch.cuda.device_count()}"
        )
    library = ctypes.CDLL(
        str(kernel_module("expert_exchange", "alltoall", KEY) / "module.so")
    )
    for name in ENTRIES:
        function = getattr(library, f"uniserve_expert_exchange_{name}")
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_int32
    channel = Channel(RANKS)
    failures = []

    def rank_body(rank):
        try:
            run_rank(library, channel, rank, rank)
        except BaseException as error:  # noqa: BLE001 - any failure is the result
            failures.append(f"rank {rank}: {error!r}")
            # The other ranks may be waiting on this one.
            channel.meeting.abort()

    threads = [
        threading.Thread(target=rank_body, args=(rank,))
        for rank in range(RANKS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, "; ".join(failures)
