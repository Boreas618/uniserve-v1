"""A process that holds a GPU's free storage until its input ends.

Run as ``python -m uniserve_testing.updates.ballast`` with one GPU visible
to it (``CUDA_VISIBLE_DEVICES``), it allocates every byte the device lets
it, prints the bytes it holds on one line once it holds them, and holds
them until its standard input ends: its starter closes it to free them, and
the kernel closes it when the starter ends however it ends. It reads no
arguments.
"""

from __future__ import annotations

import sys

import torch

#: The smallest request the ballast makes, in bytes: PyTorch's caching
#: allocator rounds a large allocation up to 2 MiB, so a smaller request
#: holds nothing a larger one left free.
GRANULE = 2 << 20


def main() -> None:
    """Hold the device's free storage, report it, and wait for the end of
    standard input.
    """  # noqa: D205
    free, _ = torch.cuda.mem_get_info()
    held: list[torch.Tensor] = []

    # The device's free bytes are not one allocation's: the allocator's
    # granularity and the device's fragmentation leave room a single request
    # of all of them does not take. Each request that fails is halved, and
    # each that succeeds is made again, until a granule fails.
    request = free
    while request >= GRANULE:
        try:
            held.append(torch.empty(request, dtype=torch.uint8, device="cuda"))
        except torch.OutOfMemoryError:
            request //= 2

    print(sum(block.numel() for block in held), flush=True)
    sys.stdin.read()


if __name__ == "__main__":
    main()
