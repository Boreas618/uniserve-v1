"""Host lists of the eager sampling methods, copied to the device.

What the caller of a sampling method knows on the host (which rows take a
shaping stage, token sets, draft tokens) arrives as lists
(:mod:`uniserve.sampling.select`). :func:`upload` copies them to the
device without waiting for the stream. A CUDA tensor built from host values
(``torch.tensor(values, device=...)``) copies from pageable memory and
synchronizes the stream instead: the host would wait for the forward whose
logits it selects from before queuing the selection, and the device would
idle until the host had queued the selection and the next step.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain

import torch


def upload(
    device: torch.device,
    *lists: Sequence[int] | Sequence[float],
    dtype: torch.dtype = torch.long,
) -> tuple[torch.Tensor, ...]:
    """Copy ``lists`` to ``device`` as one-dimensional ``dtype`` tensors.

    The lists are packed into one host tensor and copied once; each result
    is a view of that copy, in argument order. On CUDA the host tensor is
    page-locked and the copy is queued on the current stream without
    waiting for it. The caching host allocator reuses the page-locked block
    only after the copy has run, so the host values need not outlive the
    call. On the host the results view the packed tensor itself.
    """
    lengths = [len(values) for values in lists]
    packed = torch.tensor(
        list(chain.from_iterable(lists)),
        dtype=dtype,
        pin_memory=device.type == "cuda",
    )
    return packed.to(device, non_blocking=True).split(lengths)


__all__ = ["upload"]
