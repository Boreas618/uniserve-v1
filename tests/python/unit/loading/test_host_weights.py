"""A repeated layer stack moves its projections to host memory and slots.

The worker runtime owns the page-locked host region, the device slots and
the copy schedule; the library states which modules stream, how a layer's
projection matrices lie in a slot, and rebinds the parameters to slot views
with hooks around every projection call.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_bound_layers_compute_from_their_slots_between_the_hooks():
    loaded = _loaded("cuda")
    value = torch.randn(96, WIDTH, device="cuda", dtype=torch.bfloat16)
    expected = _evaluate(loaded, value)

    layers, layer_bytes = host_weights.layout(loaded, "body")
    assert layers == LAYERS
    host = torch.empty(layers * layer_bytes, dtype=torch.uint8).pin_memory()
    slots = [
        torch.empty(layer_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(5)
    ]
    calls = []

    def before(layer, stream):
        # The runtime's part: make the layer readable in its slot.
        calls.append(("before", layer))
        start = layer * layer_bytes
        slots[layer % 5].copy_(host[start : start + layer_bytes])

    def after(layer, stream):
        calls.append(("after", layer))

    host_weights.bind(loaded, "body", before, after, False, host, *slots)

    assert torch.equal(_evaluate(loaded, value), expected)
    # Each layer's two projection calls are bracketed, in layer order.
    assert calls == [
        (event, layer)
        for layer in range(LAYERS)
        for _ in range(2)
        for event in ("before", "after")
    ]
    # Every projection matrix now views its layer's slot.
    for index, layer in enumerate(loaded.module.body.layers):
        storage = slots[index % 5].untyped_storage().data_ptr()
        assert layer.down.weight.untyped_storage().data_ptr() == storage


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
            for _ in range(5)
        ]

    def hooks(buffers):
        def before(layer, stream):
            start = layer * layer_bytes
            buffers[layer % 5].copy_(host[start : start + layer_bytes])

        return before, lambda layer, stream: None

    # A model of other values whose matrices are views of the region at
    # their places, as binding a worker's retained weights leaves them.
    restarted = _loaded("cuda")
    places = host_weights.placements(restarted, "body")
    first = slots()
    host_weights.bind(loaded, "body", *hooks(first), False, host, *first)
    for parameter, offset in places:
        parameter.data = torch.empty(0, dtype=parameter.dtype).set_(
            host.untyped_storage(),
            offset // parameter.element_size(),
            parameter.shape,
            parameter.stride(),
        )

    second = slots()
    host_weights.bind(restarted, "body", *hooks(second), True, host, *second)
    assert torch.equal(_evaluate(restarted, value), expected)
