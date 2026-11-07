#!/usr/bin/env python3
"""Drive the alltoall backend's whole lifecycle over real devices.

    python exchange_check.py <module_dir> [--attention N] <device> <device> [<device> ...]

``<module_dir>`` is what ``resolve`` hands a runner: the directory holding
``module.so``. Each named device is one rank, on a thread of this process, which
is how a main process holds its ranks. The check routes random rows through a
dispatch, has every rank stand in for its experts by writing each received row's
weighted sum of a per-expert factor into ``combined``, and compares the combine
with the same sum computed on the host. It also exercises what a served step
meets: a rank with no tokens, a step whose capacity is below the transport's, a
launch whose live count is below its extent, and a suspend and restore that must
keep the addresses. With ``--attention N`` the first N ranks hold no experts: they
send rows and take sums back, as a dedicated deployment's attention ranks do, and
the others hold every expert. Run it with the Program Server's environment
(``crates/program-server/python/.venv/bin/python``), which has torch.
"""

import argparse
import ctypes
import pathlib
import sys
import threading

import torch

BF16 = 4


class Lifecycle(ctypes.Structure):
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


class Channel:
    """The host collectives of a group whose ranks are threads of this process."""

    def __init__(self, ranks):
        self.ranks = ranks
        self.meeting = threading.Barrier(ranks)
        self.slots = [b""] * ranks

    def gather(self, rank, mine, peers, count):
        self.slots[rank] = ctypes.string_at(mine, count)
        self.meeting.wait()
        for index, value in enumerate(self.slots):
            ctypes.memmove(peers + index * count, value, count)
        self.meeting.wait()
        return 0

    def barrier(self):
        self.meeting.wait()
        return 0


def message(call):
    return bytes(call.error).split(b"\0", 1)[0].decode(errors="replace")


def tensor_at(address, shape, dtype, device):
    """A tensor over `address`, a device pointer this rank maps."""
    count = 1
    for extent in shape:
        count *= extent
    size = count * torch.empty((), dtype=dtype).element_size()

    class Holder:
        __cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (address, False),
            "version": 2,
        }

    return torch.as_tensor(Holder(), device=device).view(dtype).view(*shape)


def rank_body(library, channel, rank, device, options, failures, ready):
    try:
        run_rank(library, channel, rank, device, options)
    except BaseException as error:  # noqa: BLE001 - a failure of any kind is the result
        failures.append(f"rank {rank}: {error!r}")
        ready.abort()
        raise


def run_rank(library, channel, rank, device, options):
    ranks, hidden, top, experts, capacity, attention = options
    torch.cuda.set_device(device)
    torch.zeros(1, device=device)  # the primary context is current on this thread
    stream = torch.cuda.Stream(device=device)
    gather = GATHER(lambda _, mine, peers, count: channel.gather(rank, mine, peers, count))
    meet = BARRIER(lambda _: channel.barrier())
    group = Group(
        handle=None,
        communicator=None,
        ranks=ranks,
        rank=rank,
        device=device,
        all_gather=gather,
        barrier=meet,
    )
    life = Lifecycle(
        group=ctypes.cast(ctypes.pointer(group), ctypes.c_void_p),
        ranks=ranks,
        rank=rank,
        experts=experts,
        hidden=hidden,
        max_tokens=capacity,
        top=top,
        dtype=BF16,
        device=device,
        attention_ranks=attention,
    )
    if library.uniserve_expert_exchange_create(ctypes.byref(life), None) != 0:
        raise RuntimeError(f"create: {message(life)}")
    info = Info(transport=life.transport)
    if library.uniserve_expert_exchange_info(ctypes.byref(info), None) != 0:
        raise RuntimeError("info")
    rows = ranks * capacity
    states = tensor_at(info.states, (rows, hidden), torch.bfloat16, device)
    ids = tensor_at(info.ids, (rows, top), torch.int32, device)
    weights = tensor_at(info.weights, (rows, top), torch.float32, device)
    combined = tensor_at(info.combined, (rows, hidden), torch.bfloat16, device)
    live_view = tensor_at(info.live, (1,), torch.int32, device)
    local = experts // (ranks - attention)
    owner = rank - attention
    factor = torch.arange(1, experts + 1, device=device, dtype=torch.float32) / experts
    # Rank 0 sends the most, the last rank nothing: a rank with no tokens still joins.
    for trial, (extent, live) in enumerate(
        ((capacity, capacity), (capacity // 2, capacity // 4), (4, 3))
    ):
        # Ranks send different counts, and the last rank nothing on the first trial: the
        # agreed step is the largest count, and a rank with no tokens still joins.
        silent = (trial == 0 and rank == ranks - 1) or (attention and rank >= attention)
        tokens = 0 if silent else extent
        count = 0 if silent else max(live - rank, 1)
        step = max(live, 1)
        generator = torch.Generator(device="cpu").manual_seed(1000 * trial + rank)
        hidden_rows = torch.randn(max(tokens, 1), hidden, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        scores = torch.rand(max(tokens, 1), experts, generator=generator)
        route = torch.topk(scores, top, dim=1).indices.to(device=device, dtype=torch.int32)
        route_weights = torch.rand(max(tokens, 1), top, generator=generator).to(device)
        live_word = torch.tensor([count], dtype=torch.int32, device=device)
        out = torch.full((max(tokens, 1), hidden), float("nan"), dtype=torch.bfloat16, device=device)
        with torch.cuda.stream(stream):
            stream.wait_stream(torch.cuda.current_stream(device))
            life.capacity = step
            if library.uniserve_expert_exchange_step(ctypes.byref(life), ctypes.c_void_p(stream.cuda_stream)) != 0:
                raise RuntimeError(f"step: {message(life)}")
            call = Call(
                transport=life.transport,
                states=hidden_rows.data_ptr(),
                ids=route.data_ptr(),
                weights=route_weights.data_ptr(),
                live=live_word.data_ptr(),
                out=out.data_ptr(),
                mode=0,
                tokens=tokens,
                top=top,
                hidden=hidden,
                experts=experts,
                max_tokens=capacity,
                ranks=ranks,
                dtype=BF16,
            )
            if library.uniserve_expert_exchange_launch(ctypes.byref(call), ctypes.c_void_p(stream.cuda_stream)) != 0:
                raise RuntimeError(f"dispatch: {message(call)}")
            stream.synchronize()
            # Stand in for the experts: each received row sums weight * factor over its
            # local routes, times the row.
            if owner >= 0:
                live_rows = int(live_view[0].item())
                received = states[:live_rows].float()
                routes = ids[:live_rows].long()
                mine = (routes >= owner * local) & (routes < (owner + 1) * local)
                taken = torch.where(
                    mine, weights[:live_rows] * factor[routes.clamp(min=0, max=experts - 1)], 0.0
                )
                combined[:live_rows] = (received * taken.sum(dim=1, keepdim=True)).to(torch.bfloat16)
            stream.synchronize()
            call.mode = 1
            if library.uniserve_expert_exchange_launch(ctypes.byref(call), ctypes.c_void_p(stream.cuda_stream)) != 0:
                raise RuntimeError(f"combine: {message(call)}")
            stream.synchronize()
        if tokens:
            routes = route[:count].long()
            expected = (
                hidden_rows[:count].float()
                * (route_weights[:count] * factor[routes]).sum(dim=1, keepdim=True)
            )
            got = out[:count].float()
            error = (got - expected).abs().max().item()
            bound = 0.02 * expected.abs().max().item() + 1e-3
            if not error <= bound:
                raise RuntimeError(f"trial {trial}: combine differs by {error}, bound {bound}")
            if tokens > count and not bool((out[count:tokens] == 0).all()):
                raise RuntimeError(f"trial {trial}: a padded row is not zero")
        print(f"  rank {rank} trial {trial}: {tokens} rows, {count} live, step {step}: ok", flush=True)
    before = (info.states, info.combined)
    if library.uniserve_expert_exchange_suspend(ctypes.byref(life), None) == 0:
        if library.uniserve_expert_exchange_restore(ctypes.byref(life), None) != 0:
            raise RuntimeError(f"restore: {message(life)}")
        after = Info(transport=life.transport)
        library.uniserve_expert_exchange_info(ctypes.byref(after), None)
        if (after.states, after.combined) != before:
            raise RuntimeError("restore moved the views: a captured graph addresses the old ones")
    else:
        print(f"  rank {rank}: this backend does not suspend: {message(life)}", flush=True)
    if library.uniserve_expert_exchange_release(ctypes.byref(life), None) != 0:
        raise RuntimeError(f"release: {message(life)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=pathlib.Path)
    parser.add_argument("--attention", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("devices", type=int, nargs="+")
    arguments = parser.parse_args()
    library = ctypes.CDLL(str(arguments.module / "module.so"))
    for name in ("create", "step", "suspend", "restore", "release", "info", "launch"):
        entry = getattr(library, f"uniserve_expert_exchange_{name}")
        entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        entry.restype = ctypes.c_int32
    ranks = len(arguments.devices)
    channel = Channel(ranks)
    failures = []
    options = (ranks, 2048, 8, 128, arguments.capacity, arguments.attention)
    threads = [
        threading.Thread(
            target=rank_body,
            args=(library, channel, rank, device, options, failures, channel.meeting),
        )
        for rank, device in enumerate(arguments.devices)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
