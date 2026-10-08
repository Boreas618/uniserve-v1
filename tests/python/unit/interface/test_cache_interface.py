"""The worker's cache interface describes, resets and converts unit pools."""

import subprocess
import sys

import pytest
import torch
from torch import nn

from uniserve_models.cache.units import UnitPool
from uniserve_models.interface import cache
from uniserve_models.model import CausalLM, TransformerDecoder
from uniserve_models.nn.attention import Attention

pytestmark = pytest.mark.unit

WINDOWED = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
FULL = [5, 11]


class _Layer(nn.Module):
    def __init__(self, index, window, dtype, heads):
        super().__init__()
        # The decoder takes a layer's cache dtype from its first parameter.
        self.scale = nn.Parameter(torch.zeros((), dtype=dtype))
        name = f"backbone.layers.{index}.attention"
        self.attention = (
            Attention(*heads[1], cache_name=name)
            if index % 6 == 5
            else Attention(*heads[0], cache_name=name, window=window)
        )


def _model(
    *,
    layers=12,
    window=8,
    dtype=torch.float32,
    heads=((4, 4, 4), (2, 1, 8)),
):
    """Five windowed layers per full layer; windowed rows twice as wide."""
    backbone = TransformerDecoder(
        nn.Embedding(8, 4),
        nn.ModuleDict(
            {
                str(index): _Layer(index, window, dtype, heads)
                for index in range(layers)
            }
        ),
        nn.Identity(),
    )
    return CausalLM(backbone, nn.Identity())


def test_hybrid_plan_groups_layers_by_window_and_page_shape():
    plan = cache.cache_plan(_model(), 16, "")

    windowed, full = plan["groups"]
    # The windowed rows are the widest, so their pages hold ``block_size``
    # tokens; a full page holds twice as many in one unit. Twelve layers
    # give two columns: five units per windowed page.
    assert (plan["columns"], plan["plane_bytes"]) == (2, 16 * 16 * 4)
    assert plan["unit_bytes"] == 2 * 2 * (16 * 16 * 4 + 1)
    assert (plan["quantized"], plan["dtype"]) == (False, "float32")
    assert windowed == {
        "window": 8,
        "page_tokens": 16,
        "units_per_page": 5,
        "layers": [f"backbone.layers.{index}.attention" for index in WINDOWED],
        "layer_ids": WINDOWED,
        "axis_layers": WINDOWED,
        "num_kv_heads": 4,
        "total_kv_heads": 4,
        "kv_head_offset": 0,
        "head_dim": 4,
        "dtype": "float32",
        "compute_dtype": "float32",
    }
    assert (full["window"], full["page_tokens"], full["units_per_page"]) == (
        -1,
        32,
        1,
    )
    assert full["layer_ids"] == FULL == full["axis_layers"]


def test_plan_publishes_the_configured_storage():
    fp8 = cache.cache_plan(_model(layers=6), 16, "float8_e4m3fn")
    assert (fp8["quantized"], fp8["dtype"]) == (True, "float8_e4m3fn")
    assert {group["dtype"] for group in fp8["groups"]} == {"float8_e4m3fn"}
    # One byte per element plus a flag and an FP32 scale per column/field.
    assert fp8["unit_bytes"] == fp8["columns"] * 2 * (fp8["plane_bytes"] + 5)

    dense = cache.cache_plan(_model(layers=6), 16, "bfloat16")
    assert (dense["quantized"], dense["dtype"]) == (False, "bfloat16")
    assert {group["compute_dtype"] for group in dense["groups"]} == {"float32"}
    with pytest.raises(ValueError, match="dtype"):
        cache.cache_plan(_model(layers=6), 16, "complex")


def test_a_pipeline_stage_places_its_layers_in_the_group_axis():
    model = _model()
    # A later stage keeps only layers 6..11, as pipeline binding prunes the
    # decoder; the logical cache identities survive.
    for index in range(6):
        del model.backbone.layers[str(index)]
    windowed, full = cache.cache_plan(model, 16, "")["groups"]
    assert windowed["layer_ids"] == [6, 7, 8, 9, 10]
    assert windowed["axis_layers"] == WINDOWED
    assert (full["layer_ids"], full["axis_layers"]) == ([11], FULL)


def test_portable_attention_reads_every_page():
    model = _model(layers=6)
    plan = cache.cache_plan(model, 64, "")
    assert cache.reads_pages(model, "torch", plan)


@pytest.mark.gpu
def test_native_attention_bounds_the_hybrid_page_size():
    heads = ((16, 8, 256), (16, 2, 512))
    hybrid = _model(dtype=torch.bfloat16, heads=heads).cuda()
    # Full-attention rows occupy half the bytes, so a 64-token base page
    # gives them 128-token pages, beyond their native reader's limit.
    assert not cache.reads_pages(
        hybrid, "auto", cache.cache_plan(hybrid, 64, "")
    )
    plan = cache.cache_plan(hybrid, 32, "")
    assert cache.reads_pages(hybrid, "auto", plan)
    assert sorted(group["page_tokens"] for group in plan["groups"]) == [32, 64]

    uniform = _model(layers=1, dtype=torch.bfloat16, heads=heads).cuda()
    assert cache.reads_pages(uniform, "auto", cache.cache_plan(uniform, 64, ""))


def _fields(plan, num_units):
    """Allocate a plan's unit fields as the worker does."""
    fields = {}
    for field in ("key", "value"):
        fields[f"{field}.values"] = torch.zeros(
            plan["columns"], num_units, plan["plane_bytes"], dtype=torch.uint8
        )
        fields[f"{field}.initialized"] = torch.zeros(
            plan["columns"], num_units, dtype=torch.bool
        )
        if plan["quantized"]:
            fields[f"{field}.scale"] = torch.ones(plan["columns"], num_units)
    return fields


def test_unit_bindings_reset_the_units_rust_selects():
    plan = cache.cache_plan(_model(layers=6), 16, "float8_e4m3fn")
    fields = _fields(plan, 4)
    for name, tensor in fields.items():
        tensor.fill_(1 if name.endswith("initialized") else 3)
    units = torch.tensor([2, 1, 0, 0], dtype=torch.int64)
    bound = cache.bind_units(
        {**fields, "unit_tables": torch.zeros(1)}, units, 0
    )

    bound["recycle"](1)
    assert not torch.any(fields["key.initialized"][:, 2])
    assert torch.all(fields["value.scale"][:, 2] == 1)
    assert torch.all(fields["key.values"][:, 2] == 3)

    bound["zero"](2)
    assert torch.count_nonzero(fields["value.values"][:, (1, 2)]).item() == 0
    assert torch.all(fields["key.scale"][:, 1] == 1)
    assert torch.all(fields["key.values"][:, 3] == 3)

    bound["mark_initialized"](2)
    assert torch.all(fields["value.initialized"][:, (1, 2)])
    assert torch.all(fields["value.initialized"][:, 0])


def test_conversion_binding_encodes_scratch_spans_like_writes():
    model = _model(layers=2, heads=((4, 4, 4), (4, 4, 4)), window=None)
    plan = {**cache.cache_plan(model, 4, "float8_e4m3fn"), "num_units": 3}
    (group,) = plan["groups"]
    layers, heads, dim = len(group["layers"]), group["num_kv_heads"], 4
    elements = 4 * layers * heads * dim
    scales_count = 4 * layers * heads
    fields = _fields(plan, 3)
    scratch = torch.zeros(20 * elements + 8 * scales_count, dtype=torch.uint8)
    convert = cache.bind_conversion(
        plan, fields, scratch, elements, scales_count, 0
    )["convert"]

    # Three tokens at offset 1 of the destination page, from a source page
    # of 8 tokens starting at token 2, with one scale group of two heads.
    generator = torch.Generator().manual_seed(5)
    shape = (3, layers, heads, dim)
    encoded = (torch.randn(2, *shape, generator=generator) * 100).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(1, 2, layers, 2, generator=generator) + 0.5
    nbytes = encoded[0].numel()
    for field in (0, 1):
        scratch[field * 8 * elements : field * 8 * elements + nbytes].copy_(
            encoded[field].reshape(-1).view(torch.uint8)
        )
    scratch[20 * elements : 20 * elements + scales.numel() * 4].copy_(
        scales.reshape(-1).view(torch.uint8)
    )
    convert(0, "float8_e4m3fn", "bfloat16", 3, 1, 2, 8, 2, 1, [2])

    expected = UnitPool.from_plan(plan, _fields(plan, 3))
    head_scales = scales[0].repeat_interleave(2, dim=-1)
    decoded = (
        (encoded.float() * head_scales[:, None, :, :, None])
        .to(torch.bfloat16)
        .float()
    )
    for layer, name in enumerate(group["layers"]):
        expected.state(name).write(
            (2,),
            start=1,
            key=decoded[0, :, layer],
            value=decoded[1, :, layer],
        )
    actual = UnitPool.from_plan(plan, fields)
    for field in ("key", "value"):
        assert torch.equal(actual.scales_of(field), expected.scales_of(field))
        for name in group["layers"]:
            assert torch.equal(
                actual.state(name).initialized[field],
                expected.state(name).initialized[field],
            )
        assert torch.equal(
            actual.planes_of(0, field).view(torch.uint8),
            expected.planes_of(0, field).view(torch.uint8),
        )


def test_interface_import_does_not_initialize_cuda():
    code = (
        "import torch\n"
        "from uniserve_models.interface import cache\n"
        "assert set(cache.functions()) == {'model.cache_plan', "
        "'model.reads_pages', 'cache.bind_units', 'cache.bind_conversion'}\n"
        "assert not torch.cuda.is_initialized()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
