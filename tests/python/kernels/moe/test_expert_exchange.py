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

The same lifecycle also runs with each rank in a process of its own, their
host collectives over a socket between the processes, as the members of an
expert union on two hosts meet: the transport then maps the other process's
window by its fabric handle, since a file descriptor names a window in its own
process alone.

It needs two CUDA devices. The call structures' mirrors are held to the
operator crate's generated header, which needs none.
"""

import ctypes
import socket
import subprocess
import sys

import pytest
import torch
from uniserve_testing.kernels.moe.exchange import (
    MIRRORS,
    Call,
    Channel,
    Info,
    Lifecycle,
    entered,
    group_of,
    load,
    run_ranks,
    tensor_at,
)
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
    f"{EXPERTS}_max_tokens_{CAPACITY}_ranks_{RANKS}_attention_0_dtype_bf16_"
    "weights_none_shared_0_limit_0x0"
)

#: The directions of a launch.
DISPATCH = abi_enumerator("UNISERVE_EXCHANGE_DISPATCH")
COMBINE = abi_enumerator("UNISERVE_EXCHANGE_COMBINE")

#: The trials, each the rows a rank sends and the live count among them: the
#: transport's capacity, a step below it with a live count below its extent,
#: and a step of four rows.
TRIALS = ((CAPACITY, CAPACITY), (CAPACITY // 2, CAPACITY // 4), (4, 3))

#: The combine's allowance: 2% of the largest expected value, plus `1e-3`.
RELATIVE, ABSOLUTE = 0.02, 1e-3

#: How long a rank's process may take, in seconds: a lifecycle takes a few
#: seconds, and a rank whose peer never arrives waits on its device unbounded.
PROCESS_S = 300.0


@pytest.mark.parametrize("structure", sorted(MIRRORS))
def test_a_call_mirror_is_its_generated_header_structure(structure):
    """The generated header asserts the size and offsets the mirror has."""
    assert_abi_layout(MIRRORS[structure], structure)


class PeerChannel:
    """The host collectives of a group of two ranks, each in a process of its
    own, over the socket `peer` between the processes: what an expert union's
    host group carries between the engine processes of two hosts.
    """  # noqa: D205

    def __init__(self, peer):
        """The collectives over `peer`, this process's end of the socket."""
        self.peer = peer

    def _read(self, size):
        """`size` bytes from the other process."""
        data = bytearray()
        while len(data) < size:
            chunk = self.peer.recv(size - len(data))
            if not chunk:
                raise ConnectionError("the other rank's process ended")
            data += chunk
        return bytes(data)

    def _exchange(self, value):
        """Send `value` and return the other rank's: both send first, and a
        value this small fits the socket's buffer.
        """  # noqa: D205
        self.peer.sendall(len(value).to_bytes(8, "little") + value)
        return self._read(int.from_bytes(self._read(8), "little"))

    def gather(self, rank, mine, peers, count):
        """Both ranks' `count` bytes at `mine`, in rank order at `peers`."""
        value = ctypes.string_at(mine, count)
        theirs = self._exchange(value)
        ordered = (value, theirs) if rank == 0 else (theirs, value)
        for index, item in enumerate(ordered):
            ctypes.memmove(peers + index * count, item, count)
        return 0

    def barrier(self):
        """Wait for the other rank."""
        self._exchange(b"")
        return 0


def run_rank(library, channel, rank, device):
    """One rank's lifecycle: create, the trials, suspend, restore, release."""
    torch.cuda.set_device(device)
    # The primary context is current on this thread.
    torch.zeros(1, device=device)
    stream = torch.cuda.Stream(device=device)
    group, callbacks = group_of(channel, RANKS, rank, device)
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
                capacity=max(tokens, 1),
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
    del callbacks


def needs_devices():
    """Skip a case on a host of fewer than `RANKS` devices."""
    if torch.cuda.device_count() < RANKS:
        pytest.skip(
            f"needs {RANKS} CUDA devices, this host shows "
            f"{torch.cuda.device_count()}"
        )


@pytest.mark.gpu
def test_two_ranks_dispatch_combine_suspend_and_restore(kernel_module):
    needs_devices()
    library = load(
        kernel_module("expert_exchange", "alltoall", KEY) / "module.so"
    )
    channel = Channel(RANKS)
    failures = run_ranks(
        channel, RANKS, lambda rank: run_rank(library, channel, rank, rank)
    )
    assert not failures, "; ".join(failures)


@pytest.mark.gpu
def test_ranks_of_two_processes_dispatch_and_combine_over_fabric_handles(
    kernel_module,
):
    """Each rank runs the lifecycle in a process of its own (this file run as
    a script), their host collectives over a socket pair between the
    processes: every trial's combine matches, which needs each rank to map
    the other process's window.
    """  # noqa: D205
    needs_devices()
    module = kernel_module("expert_exchange", "alltoall", KEY) / "module.so"
    ends = socket.socketpair()
    processes = []
    for rank, end in enumerate(ends):
        argv = [sys.executable, __file__, str(module), str(rank)]
        processes.append(
            subprocess.Popen(
                [*argv, str(end.fileno())],
                pass_fds=(end.fileno(),),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    for end in ends:
        end.close()

    outputs = []
    for process in processes:
        try:
            outputs.append(process.communicate(timeout=PROCESS_S)[0])
        except subprocess.TimeoutExpired:
            for other in processes:
                other.kill()
            outputs.append(process.communicate()[0])
    failed = [
        f"rank {rank}: {output}"
        for rank, (process, output) in enumerate(zip(processes, outputs))
        if process.returncode != 0
    ]
    assert not failed, "\n".join(failed)


if __name__ == "__main__":
    # One rank of the two-process case: the module, the rank (its device too)
    # and the descriptor of this process's end of the socket to the other.
    path, rank, descriptor = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    peer = socket.socket(fileno=descriptor)
    run_rank(load(path), PeerChannel(peer), rank, rank)
