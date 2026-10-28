"""A repeated layer stack moves its projections to host storage and slots.

The worker allocates the page-locked host region and the device slots; the
library states which modules stream, how a layer's projection matrices lie
in a slot and how many slots the stack streams through, rebinds the
parameters to slot views, and copies each layer into its slot ahead of its
projection calls on a stream of its own (``HostStack``, scheduled by
``Ring``).
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.python.fixtures.cuda_stream import blocked_stream
from uniserve.loading import host_weights
from uniserve.nn import Linear, MergedColumnParallelLinear

pytestmark = pytest.mark.unit

WIDTH, HIDDEN, LAYERS = 64, 192, 10


class _Layer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.projection = MergedColumnParallelLinear(
            WIDTH,
            {"a": HIDDEN, "b": HIDDEN},
            bias=True,
            device=device,
            dtype=torch.bfloat16,
        )
        self.down = Linear(
            HIDDEN, WIDTH, bias=False, device=device, dtype=torch.bfloat16
        )

    def forward(self, value):
        out = self.projection(value)
        return value + self.down(torch.tanh(out["a"] * out["b"]))


def _loaded(device):
    torch.manual_seed(3)
    model = nn.Module()
    model.body = nn.Module()
    model.body.layers = nn.ModuleList(_Layer(device) for _ in range(LAYERS))
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.2)
    return SimpleNamespace(module=model)


def _evaluate(loaded, value):
    for layer in loaded.module.body.layers:
        value = layer(value)
    return value


class _Buffers:
    """What each buffer of a ring holds, from the copies it names."""

    def __init__(self, ring: host_weights.Ring) -> None:
        self.ring = ring
        self.held: list[int | None] = [None] * ring.slots

    def acquire(self, layer: int) -> list[int] | None:
        copies = self.ring.acquire(layer)
        for copied in copies or []:
            self.held[copied % self.ring.slots] = copied
        return copies


def test_the_stack_is_the_largest_container_of_four_layers_of_one_class():
    model = nn.Module()
    model.small = nn.ModuleList(nn.Linear(4, 4) for _ in range(6))
    model.body = nn.Module()
    model.body.layers = nn.ModuleDict(
        {str(index): nn.Linear(16, 16) for index in range(4)}
    )
    model.mixed = nn.ModuleList(
        (nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.Linear(64, 64))
    )

    assert host_weights.layer_stack(model) == tuple(model.body.layers.values())
    with pytest.raises(ValueError, match="no stack of four layers"):
        host_weights.layer_stack(model.mixed)


def test_layers_of_differing_projection_layout_are_refused():
    loaded = _loaded("cpu")
    layers = loaded.module.body.layers
    layers[3].down = Linear(
        HIDDEN, WIDTH, bias=False, device="cpu", dtype=torch.float32
    )
    with pytest.raises(ValueError, match="one common projection weight"):
        host_weights.layout(loaded, "body")


def test_weights_off_cuda_are_refused():
    with pytest.raises(ValueError, match="one CUDA device"):
        host_weights.layout(_loaded("cpu"), "body")


def test_a_stack_streams_through_the_fewest_slots_of_four_or_more_dividing_it():
    assert host_weights.slots(10) == 5
    assert host_weights.slots(50) == 5
    assert host_weights.slots(48) == 4
    assert host_weights.slots(7) == 7
    with pytest.raises(ValueError, match="no slot count of at least four"):
        host_weights.slots(3)


def test_a_first_pass_copies_each_layer_once_ahead_of_its_use():
    ring = host_weights.Ring(10)
    copies = [ring.acquire(layer) for layer in range(10)]

    # The first layer and the three after it, then one more layer per
    # step: two slots hold live layers.
    assert copies[0] == [0, 1, 2, 3]
    for layer, copied in enumerate(copies[1:], start=1):
        assert copied == [(layer + 3) % 10], f"layer {layer}"
    assert sorted(sum(copies, [])[:10]) == list(range(10))


def test_a_layer_already_live_needs_no_copy_or_wait():
    ring = host_weights.Ring(8)
    assert ring.acquire(0) is not None
    assert ring.acquire(0) is None
    assert ring.acquire(1) is not None
    # The previous layer stays live while the next one starts.
    assert ring.acquire(0) is None


def test_no_copy_targets_the_slot_of_a_live_layer():
    buffers = _Buffers(host_weights.Ring(10))
    previous = None
    for _ in range(3):
        for layer in range(10):
            copies = buffers.acquire(layer) or []
            # The layer before stays live while this one starts.
            live = {layer} if previous is None else {previous, layer}
            for copied in copies:
                if copied != layer:
                    assert all(copied % 5 != other % 5 for other in live), (
                        f"layer {layer} copied {copied} over a live slot"
                    )
            # Each layer finds itself in its slot when first used.
            assert buffers.held[layer % 5] == layer
            previous = layer


def test_an_interrupted_pass_restarts_and_recopies_what_it_lacks():
    buffers = _Buffers(host_weights.Ring(10))
    for layer in range(6):
        buffers.acquire(layer)

    # A new pass starts at layer 0, whose slot now holds layer 5.
    assert buffers.acquire(0)[0] == 0
    # The remaining layers of the new pass are each in their slot when
    # first used.
    for layer in range(1, 10):
        buffers.acquire(layer)
        assert buffers.held[layer % 5] == layer


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_bound_layers_compute_from_their_slots_over_repeated_passes():
    loaded = _loaded("cuda")
    value = torch.randn(96, WIDTH, device="cuda", dtype=torch.bfloat16)
    expected = _evaluate(loaded, value)

    layers, layer_bytes = host_weights.layout(loaded, "body")
    assert layers == LAYERS
    host = torch.empty(layers * layer_bytes, dtype=torch.uint8).pin_memory()
    slots = [
        torch.empty(layer_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(host_weights.slots(layers))
    ]
    stack = host_weights.bind(loaded, "body", False, host, *slots)

    # The computing stream starts only once every pass is enqueued, so a
    # copy that did not wait for the reads of its slot's previous layer
    # would overwrite it before they run.
    compute = torch.cuda.Stream()
    torch.cuda.synchronize()
    with blocked_stream("cuda") as gate:
        opened = torch.cuda.Event()
        opened.record(gate)
        compute.wait_event(opened)
        with torch.cuda.stream(compute):
            results = [_evaluate(loaded, value) for _ in range(3)]
    compute.synchronize()

    for result in results:
        assert torch.equal(result, expected)
    # Every projection matrix now views its layer's slot.
    for index, layer in enumerate(loaded.module.body.layers):
        storage = slots[index % len(slots)].untyped_storage().data_ptr()
        assert layer.down.weight.untyped_storage().data_ptr() == storage
    stack.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_a_filled_region_binds_the_matrices_it_holds_without_copying():
    loaded = _loaded("cuda")
    value = torch.randn(96, WIDTH, device="cuda", dtype=torch.bfloat16)
    expected = _evaluate(loaded, value)
    layers, layer_bytes = host_weights.layout(loaded, "body")
    host = torch.empty(layers * layer_bytes, dtype=torch.uint8).pin_memory()

    def slots():
        return [
            torch.empty(layer_bytes, dtype=torch.uint8, device="cuda")
            for _ in range(host_weights.slots(layers))
        ]

    # A model of other values whose matrices are views of the region at
    # their places, as binding a worker's retained weights leaves them.
    restarted = _loaded("cuda")
    places = host_weights.placements(restarted, "body")
    first = host_weights.bind(loaded, "body", False, host, *slots())
    for parameter, offset in places:
        parameter.data = torch.empty(0, dtype=parameter.dtype).set_(
            host.untyped_storage(),
            offset // parameter.element_size(),
            parameter.shape,
            parameter.stride(),
        )

    second = host_weights.bind(restarted, "body", True, host, *slots())
    assert torch.equal(_evaluate(restarted, value), expected)
    first.close()
    second.close()
