"""Copies that lay a stored latent trajectory out in a denoising rank's pages.

A resumed request's latent preparation (a video request's, or the image a
token request had begun) receives the stored pages of its trajectory it
reads in the host latent pools and the copies the worker planned from the
stored ranks' layouts to the rank's own (the copy manifest library,
``crates/worker/src/storage/latent/restore.rs``): runs of bytes, each within
one stored page, to byte offsets of the trajectory's pages laid back to back
in logical order. The video and image executors run them with
:func:`copy_runs` after they wrote the trajectory's initial noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

__all__ = ["copy_runs"]


def copy_runs(
    target: torch.Tensor,
    stored: Sequence[torch.Tensor],
    copies: Iterable[Sequence[int]],
) -> None:
    """Copy runs of bytes of stored pages into ``target``.

    ``target`` and every ``stored`` page are flat ``uint8`` tensors; each
    copy ``[page, offset, at, bytes]`` writes ``bytes`` bytes from byte
    ``offset`` of ``stored[page]`` at byte ``at`` of ``target``, in the
    current stream's order, without waiting on the host.

    Raises:
        ValueError: A copy reaches past its page or past ``target``.
    """
    for page, offset, at, length in copies:
        source = stored[page]
        if offset + length > source.numel() or at + length > target.numel():
            raise ValueError(
                f"a latent copy of {length} bytes reaches past its page or "
                "its trajectory's pages"
            )
        target[at : at + length].copy_(
            source[offset : offset + length], non_blocking=True
        )
