"""Distributed expert weights reach their slots before the layers read them.

``WeightPrefetch`` schedules the worker's per-layer copy of peers' shards
into two slots: each expert layer waits for its own copy, prefetches the
next layer into the other slot, and releases its slot after reading it; a
captured invocation submits each copy as break work at every replay. Here
the copy reads each layer's bytes from a device tensor on the stream the
schedule names, in place of the worker's peer reads. Either the copies or
the first read are held at a device-side gate, so a missing order between a
copy and a read changes the values the layers read.
"""

import pytest
import torch
from torch import nn

from tests.python.fixtures.cuda_stream import blocked_stream
from uniserve.cuda_graph import capture
from uniserve.distributed.experts import WeightPrefetch

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

LAYERS, ELEMENTS = 4, 1 << 20


class _Weights:
    """Each layer's remote bytes, the two slots and the copy into them."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.sources = torch.empty(LAYERS, ELEMENTS, device=device)
        self.slots = torch.zeros(2, ELEMENTS, device=device)
        # Every copy waits for `copies` and the first read for `reads`; a
        # test records one on a held stream to gate it. A captured read
        # waits through a graph node, so its gate is external and exists
        # before the capture.
        idle = torch.cuda.current_stream(device)
        self.copies = torch.cuda.Event()
        self.reads = torch.cuda.Event(external=True)
        self.copies.record(idle)
        self.reads.record(idle)

    def copy(self, layer: int, stream: int) -> None:
        stream = torch.cuda.ExternalStream(stream, device=self.device)
        stream.wait_event(self.copies)
        with torch.cuda.stream(stream):
            self.slots[layer % 2].copy_(self.sources[layer])


def _invocation(prefetch, layers, weights, outputs) -> None:
    """One invocation: every expert layer reads its slot into its output."""
    compute = torch.cuda.current_stream(weights.device)
    with prefetch.activate():
        for index, layer in enumerate(layers):
            prefetch.before(layer)
            if index == 0:
                compute.wait_event(weights.reads)
            outputs[index].copy_(weights.slots[index % 2])
            prefetch.after(layer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
@pytest.mark.parametrize("held", ["copies", "reads"])
@torch.inference_mode()
def test_each_layer_reads_its_own_weights(captured: bool, held: str):
    device = torch.device("cuda:0")
    layers = [nn.Identity() for _ in range(LAYERS)]
    weights = _Weights(device)
    prefetch = WeightPrefetch(layers, device, weights.copy)
    outputs = torch.zeros(LAYERS, ELEMENTS, device=device)
    compute = torch.cuda.Stream(device=device)

    def body() -> None:
        _invocation(prefetch, layers, weights, outputs)

    graph = capture(body, stream=compute) if captured else None
    try:
        # A captured invocation copies each layer again at every replay.
        for _ in range(2):
            weights.sources.normal_()
            outputs.zero_()
            torch.cuda.synchronize(device)
            # Held copies show that each read waits for its layer's copy;
            # a held first read shows that the copy into its slot for the
            # layer after next waits for that read.
            with blocked_stream(device) as gate:
                idle = torch.cuda.current_stream(device)
                weights.copies.record(gate if held == "copies" else idle)
                weights.reads.record(gate if held == "reads" else idle)
                with torch.cuda.stream(compute):
                    if graph is None:
                        body()
                    else:
                        graph.replay()
                assert not compute.query()
            compute.synchronize()
            torch.testing.assert_close(outputs, weights.sources, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize(device)
        if graph is not None:
            graph.release()

    prefetch.close()
    with pytest.raises(RuntimeError, match="closed"):
        with prefetch.activate():
            pass
