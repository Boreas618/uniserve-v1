"""The GPUs of this host and the leases end-to-end cases hold on them.

The GPU recipe runs the end-to-end cases in parallel (``pytest -n`` with
``--dist loadgroup``); each case serves on the GPUs it leases for its
duration, so independent cases share the host while no two of them share a
GPU. The host's GPUs are the devices NVML reports (:func:`host_devices`),
or the indices ``--e2e-devices`` states.

A lease is a set of advisory ``flock`` locks, one per GPU, in a directory
of the user's in the system's temporary directory, so the concurrent runs
of one user on one host respect one another; a host lock beside them is held
shared by every case and exclusively by a case that observes the whole host.
A lease is taken under an allocator lock that one case at a time holds: it
takes the free GPUs it finds and then whichever further GPUs their holders
release first, trying every GPU it lacks each :data:`POLL_S` seconds, so a
case that needs many GPUs is not starved by cases that need few, and a GPU
that frees is taken while the case waits on another. A lease's holder never
needs the allocator lock to release, so a waiting allocator always proceeds.
"""

from __future__ import annotations

import dataclasses
import fcntl
import os
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import IO

import pynvml

#: How often, in seconds, a lease short of GPUs tries the ones it lacks: a
#: case holds its GPUs for minutes, so a fraction of a second is negligible.
POLL_S = 0.5


def host_devices() -> tuple[int, ...]:
    """The indices of the GPUs NVML reports on this host."""
    pynvml.nvmlInit()
    try:
        return tuple(range(pynvml.nvmlDeviceGetCount()))
    finally:
        pynvml.nvmlShutdown()


@dataclasses.dataclass(frozen=True)
class Lease:
    """The GPUs a case holds, by their indices on this host."""

    devices: tuple[int, ...]

    @property
    def visible(self) -> str:
        """The devices as CUDA's device visibility lists them: a server
        launched with ``CUDA_VISIBLE_DEVICES`` set to this sees exactly the
        leased GPUs, as ``cuda:0`` onwards.
        """  # noqa: D205
        return ",".join(str(device) for device in self.devices)


def _lock_directory() -> Path:
    """The user's directory of lease locks on this host."""
    directory = Path(tempfile.gettempdir(), f"uniserve-gpus-{os.getuid()}")
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory


def _lock(path: Path, mode: int) -> IO[str]:
    """`path` opened and locked with `mode` (blocking unless it says not)."""
    handle = path.open("a+")
    try:
        fcntl.flock(handle, mode)
    except BlockingIOError:
        handle.close()
        raise
    return handle


@contextmanager
def lease_gpus(
    devices: Sequence[int], count: int, *, host_wide: bool = False
) -> Iterator[Lease]:
    """Hold `count` of `devices` until the block ends, and the host
    exclusively for a case that observes the whole host (`host_wide`).

    Raises:
        ValueError: When `count` exceeds the devices.
    """  # noqa: D205
    if count > len(devices):
        raise ValueError(f"{count} of {len(devices)} GPUs cannot be leased")
    wanted = count
    directory = _lock_directory()
    with ExitStack() as stack:
        stack.enter_context(
            _lock(
                directory / "host.lock",
                fcntl.LOCK_EX if host_wide else fcntl.LOCK_SH,
            )
        )
        held: list[int] = []
        with _lock(directory / "allocator.lock", fcntl.LOCK_EX):
            # The free GPUs, then whichever further GPU its holder releases
            # first: waiting on one GPU in particular would leave another
            # that frees meanwhile idle, and every case queued behind this
            # one with it.
            while True:
                for device in devices:
                    if len(held) == wanted:
                        break
                    if device in held:
                        continue
                    try:
                        handle = _lock(
                            directory / f"gpu{device}.lock",
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    except BlockingIOError:
                        continue
                    stack.callback(handle.close)
                    held.append(device)
                if len(held) == wanted:
                    break
                time.sleep(POLL_S)
        yield Lease(tuple(sorted(held)))
