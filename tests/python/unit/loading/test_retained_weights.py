"""Publishing a loaded model's weights and binding them in a new process.

A worker publishes its loaded tensors into retained memory storage by
storage (``Layout``) and its successor rebuilds them over that memory from
the manifest and the server's records (``Retained``). The test plays the
worker's part with one CUDA byte tensor as the retained memory.
"""

import pytest
import torch
from torch import nn

from uniserve.loading.retained import Layout, Retained
from uniserve.quantization import Quantizer

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="retained weights are CUDA"
    ),
]

DEVICE = torch.device("cuda", 0)


class _Model(nn.Module):
    """Coalesced branches, aliases, an encoded weight and buffers.

    The branches share one storage, the head is tied to the embedding, and
    one module is reachable under two paths.
    """

    def __init__(self, device):
        super().__init__()
        self.embedding = nn.Embedding(32, 16, device=device)
        self.head = nn.Linear(16, 32, bias=False, device=device)
        self.head.weight = self.embedding.weight
        self.gate = nn.Linear(16, 24, device=device)
        self.up = nn.Linear(16, 24, device=device)
        self.shared = self.up
        self.projection = nn.Linear(16, 16, bias=False, device=device)
        self.register_buffer("table", torch.empty(5, 3, device=device))
        self.register_buffer("derived", None)


def _loaded():
    torch.manual_seed(5)
    model = _Model(DEVICE)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
        model.table.normal_()
        # Loading coalesces the two branches into one storage, gate first.
        backing = torch.cat([model.gate.weight, model.up.weight])
        model.gate.weight.data = backing[:24]
        model.up.weight.data = backing[24:]
        model.projection.weight = nn.Parameter(
            Quantizer("fp8", axis=0).quantize(model.projection.weight),
            requires_grad=False,
        )
        model.derived = torch.arange(7, device=DEVICE)
    return model


def _snapshot(model):
    return {
        name: (
            value.dequantize().clone()
            if hasattr(value, "dequantize") and value.is_floating_point()
            else value.detach().clone()
        )
        for name, value in [
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        ]
    }


def _publish(model, only=None):
    """Publish ``model`` into one byte tensor as a worker would.

    Every tensor is published, or those ``only`` names.
    """
    layout = Layout.describe(model, only=only)
    sizes = layout.storage_bytes()
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += -(-size // 512) * 512
    memory = torch.empty(total, dtype=torch.uint8, device=DEVICE)
    for index, offset in enumerate(offsets):
        layout.publish(index, memory, offset)
    storages = layout.field_storages()
    records = {
        "field_ids": layout.field_ids(),
        "field_storages": [0] * len(storages),
        "field_offsets": [
            offsets[storage] + offset
            for storage, offset in zip(
                storages, layout.field_offsets(), strict=True
            )
        ],
        "field_dtypes": layout.field_dtypes(),
        "field_ranks": layout.field_ranks(),
        "field_shapes": layout.field_shapes(),
        "field_strides": layout.field_strides(),
    }
    return layout.manifest(), memory, records


def test_published_weights_move_into_retained_memory_unchanged():
    model = _loaded()
    expected = _snapshot(model)
    _, memory, _ = _publish(model)

    start = memory.data_ptr()
    end = start + memory.numel()
    for name, value in _snapshot(model).items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    for tensor in [model.gate.weight, model.table, model.derived]:
        assert start <= tensor.data_ptr() < end
    for field in model.projection.weight.buffers().values():
        assert start <= field.data_ptr() < end
    # Coalesced branches stay one matrix, and the tied pair one parameter.
    assert model.up.weight.data_ptr() == (
        model.gate.weight.data_ptr() + model.gate.weight.nbytes
    )
    assert model.head.weight is model.embedding.weight


def test_a_new_model_binds_the_published_weights():
    published = _loaded()
    expected = _snapshot(published)
    manifest, memory, records = _publish(published)

    # The fresh model's placeholders are dense meta tensors, as before
    # loading: binding installs the encoded weight in their place.
    model = _Model("meta")
    retained = Retained.build(manifest, [memory], load=(), **records)
    retained.install(model, {id(module) for module in model.modules()})

    for name, value in _snapshot(model).items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert model.head.weight is model.embedding.weight
    assert model.shared is model.up
    assert model.projection.weight.quantizer == Quantizer("fp8", axis=0)
    assert model.up.weight.data_ptr() == (
        model.gate.weight.data_ptr() + model.gate.weight.nbytes
    )


def test_a_worker_that_read_some_tensors_publishes_only_those():
    published = _loaded()
    manifest, memory, records = _publish(published)
    # Its successor read the projection again and bound the rest; it
    # publishes the projection alone, under the same manifest.
    successor = _loaded()
    read = frozenset({"p:projection.weight"})
    partial_manifest, partial_memory, partial_records = _publish(
        successor, only=read
    )

    assert partial_manifest == manifest
    tensors = {name.rsplit("/", 1)[0] for name in partial_records["field_ids"]}
    assert tensors == read

    # Binding takes the retained tensors but the projection, which the
    # binder reads itself.
    model = _Model("meta")
    retained = Retained.build(manifest, [memory], load=read, **records)
    retained.install(model, {id(module) for module in model.modules()})
    assert model.projection.weight.is_meta
    assert not model.gate.weight.is_meta
