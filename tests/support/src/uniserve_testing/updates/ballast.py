"""A process that holds a GPU's free storage until its input ends.

Run as ``python -m uniserve_testing.updates.ballast LEAVE`` with one GPU
visible to it (``CUDA_VISIBLE_DEVICES``), it allocates the device's free
bytes but ``LEAVE`` of them, prints the bytes it holds on one line once it
holds them, and holds them until its standard input ends: its starter closes
it to free them, and the kernel closes it when the starter ends however it
ends.
"""

from __future__ import annotations

import sys

import torch

#: The smallest request the ballast makes, in bytes: PyTorch's caching
#: allocator rounds a large allocation up to 2 MiB, so a smaller request
#: holds nothing a larger one left free.
GRANULE = 2 << 20


def main() -> None:
    """Hold the device's free storage but the bytes the first argument states,
    report what it holds, and wait for the end of standard input.
    """  # noqa: D205
    leave = int(sys.argv[1])
    free, _ = torch.cuda.mem_get_info()
    held: list[torch.Tensor] = []

    # The bytes to hold are not one allocation's: the allocator's granularity
    # and the device's fragmentation leave room a single request of all of
    # them does not take. Each request that fails is halved, and each that
    # succeeds is made again for what is still to hold, until a granule fails
    # or nothing is left to hold.
    left = max(0, free - leave)
    request = left
    while request >= GRANULE and left >= GRANULE:
        request = min(request, left)
        try:
            held.append(torch.empty(request, dtype=torch.uint8, device="cuda"))
            left -= request
        except torch.OutOfMemoryError:
            request //= 2

    print(sum(block.numel() for block in held), flush=True)
    sys.stdin.read()


if __name__ == "__main__":
    main()
