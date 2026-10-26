"""Layer weights copied into a ring of device slots ahead of their use.

Two schedules share one mechanism, :class:`SlotCopies`. Distributed expert
weights (``uniserve.distributed.experts.WeightPrefetch``) read the other
ranks' shards of the next expert layer from peer memory into two slots;
host weights (``uniserve.loading.host_weights.HostStack``) stream the
projections of a repeated layer stack from page-locked host memory into a
few slots. Each schedule decides which layer is copied when; this module
orders those copies against the streams that read the slots.

The worker owns the memory: the slots, the sources the copies read and the
storage they live in. A schedule receives the byte movement of one layer as
a function (``copy(layer, stream)``) and owns the stream it runs on and the
events around it. The worker releases the memory only after the schedule's
:meth:`SlotCopies.close` drained that stream.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = ["SlotCopies"]


class SlotCopies:
    """Copies of each layer's weights into its slot, ordered by events.

    Layer ``i`` lives in slot ``i % slots``. A copy runs on this object's
    stream once the slot's last recorded read completed and then records the
    slot's readiness; a reader's stream waits for that readiness before it
    reads and records its read after. Every event is external: inside a CUDA
    graph capture its record and wait become graph nodes, so a captured
    reader orders against the copies that the graph's break work submits on
    this stream at every replay.

    Args:
        device: The CUDA device of the slots.
        slots: The number of slots.
        layers: The number of layers, each with its copy.
        copy: ``copy(layer, stream)`` enqueues the bytes of ``layer`` into
            its slot on ``stream`` (a ``torch.cuda.Stream``), without host
            synchronization. Every address it uses stays valid until
            :meth:`close` returned.

    Raises:
        ValueError: No slot or no layer, or ``device`` is not a CUDA device.
    """

    def __init__(
        self,
        device: torch.device | str | int,
        slots: int,
        layers: int,
        copy: Callable[[int, torch.cuda.Stream], None],
    ) -> None:
        if slots < 1 or layers < 1:
            raise ValueError("slot copies need a slot and a layer")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("slot copies run on a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.layers = layers
        self.stream = torch.cuda.Stream(device=self.device)
        self._copy = copy
        # `ready[s]`: recorded on the copy stream once slot `s` holds its
        # newest copy. `consumed[s]`: recorded on a reading stream once it
        # read slot `s`.
        self._ready = [torch.cuda.Event(external=True) for _ in range(slots)]
        self._consumed = [torch.cuda.Event(external=True) for _ in range(slots)]
        # Torch creates an event at its first record and skips a wait on one
        # it never created, which inside a capture would drop the wait node.
        # Recording each once on the idle copy stream creates it complete.
        for event in (*self._ready, *self._consumed):
            event.record(self.stream)
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` drained the copies."""
        return self._closed

    def slot(self, layer: int) -> int:
        """The slot ``layer`` is copied into."""
        return layer % len(self._ready)

    def copy(self, layer: int) -> None:
        """Copy ``layer`` into its slot after the slot's last recorded read.

        No host synchronization is performed.

        Raises:
            RuntimeError: The copies are closed.
            IndexError: ``layer`` has no copy.
        """
        if self._closed:
            raise RuntimeError("layer weight copies are closed")
        if not 0 <= layer < self.layers:
            raise IndexError(
                f"layer {layer} is outside the {self.layers} copied layers"
            )
        slot = self.slot(layer)
        self.stream.wait_event(self._consumed[slot])
        self._copy(layer, self.stream)
        self._ready[slot].record(self.stream)

    def wait(self, layer: int, compute: torch.cuda.Stream) -> None:
        """Make ``layer``'s slot readable on ``compute``."""
        compute.wait_event(self._ready[self.slot(layer)])

    def release(self, layer: int, compute: torch.cuda.Stream) -> None:
        """Release ``layer``'s slot after the reads ``compute`` has queued."""
        self._consumed[self.slot(layer)].record(compute)

    def release_all(self, compute: torch.cuda.Stream) -> None:
        """Order every slot's next copy after the work ``compute`` queued."""
        for event in self._consumed:
            event.record(compute)

    def close(self) -> None:
        """Drain the copy stream and refuse further copies.

        The caller retires the slots' readers and every captured graph that
        submits copies first. Once this returns, the worker may release the
        memory the copies address.
        """
        if not self._closed:
            self.stream.synchronize()
            self._closed = True
