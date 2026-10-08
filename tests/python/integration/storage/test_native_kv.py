"""Native KV storage driven through the real numerical cache interface.

The worker's storage harness (``crates/worker/examples/storage_harness.rs``,
built with ``cargo build -p uniserve-worker --example storage_harness``)
allocates two KV unit pools in Rust, binds this package's cache interface
to them, lets the test write the producer's pages through DLPack views,
exports the producer's slot through one transport, imports it into the
consumer and returns both pools' Rust-owned tensors.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from uniserve_models.cache.units import UnitPool
from uniserve_models.interface import cache
from uniserve_models.model import CausalLM, TransformerDecoder
from uniserve_models.nn.attention import Attention

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

TOKENS = 24
WINDOW = 8


def _harness():
    import tvm_ffi

    root = Path(__file__).resolve().parents[4]
    path = Path(
        os.environ.get(
            "UNISERVE_STORAGE_HARNESS",
            root / "target" / "debug" / "examples" / "libstorage_harness.so",
        )
    )
    if not path.exists():
        pytest.fail(
            f"{path} is missing: build it with "
            "`cargo build -p uniserve-worker --example storage_harness`"
        )
    return tvm_ffi.load_module(str(path))


class _Layer(nn.Module):
    def __init__(self, index: int, dtype: torch.dtype) -> None:
        super().__init__()
        # The decoder takes a layer's cache dtype from its first parameter.
        self.scale = nn.Parameter(torch.zeros((), dtype=dtype))
        name = f"backbone.layers.{index}.attention"
        # Five windowed layers per full layer; windowed rows twice as wide,
        # so the pool has two groups with 5 and 1 units per page.
        self.attention = (
            Attention(2, 1, 8, cache_name=name)
            if index % 6 == 5
            else Attention(4, 4, 4, cache_name=name, window=WINDOW)
        )


def _model() -> CausalLM:
    backbone = TransformerDecoder(
        nn.Embedding(8, 4),
        nn.ModuleDict(
            {str(index): _Layer(index, torch.bfloat16) for index in range(12)}
        ),
        nn.Identity(),
    )
    return CausalLM(backbone, nn.Identity())


def _units(plan, first: int) -> list[list[int]]:
    """Distinct units for one slot holding ``TOKENS`` tokens per group."""
    groups = []
    for group in plan["groups"]:
        pages = -(-TOKENS // int(group["page_tokens"]))
        count = pages * int(group["units_per_page"])
        groups.append(list(range(first, first + count)))
        first += count
    return groups


def _pool(views, layout) -> UnitPool:
    tensors = {
        str(name): torch.from_dlpack(value) for name, value in views.items()
    }
    return UnitPool.from_plan(layout, tensors)


def _blocks(pool: UnitPool, plan, units: list[list[int]], name: str):
    """The block ids one layer reads: its unit position in every page."""
    first = 0
    for index, group in enumerate(plan["groups"]):
        rows = int(group["units_per_page"])
        if name in [str(layer) for layer in group["layers"]]:
            row = pool.table(name) - first
            return tuple(units[index][row::rows]), group
        first += rows
    raise KeyError(name)


def _history(group) -> int:
    window = int(group["window"])
    return 0 if window < 0 else max(0, TOKENS - window)


def _fill(plan, units):
    def fill(views, layout):
        pool = _pool(views, layout)
        device = pool.planes_of(0, "key").device
        for group in plan["groups"]:
            for name in group["layers"]:
                name = str(name)
                state = pool.state(name)
                blocks, _ = _blocks(pool, plan, units, name)
                heads = state.key.shape[2]
                dim = state.key.shape[3]
                key, value = (
                    torch.randn(
                        (TOKENS, heads, dim),
                        generator=torch.Generator().manual_seed(
                            sum(map(ord, name)) * 2 + field
                        ),
                    ).to(device=device, dtype=torch.bfloat16)
                    for field in range(2)
                )
                state.write(blocks, start=0, key=key, value=value)
        torch.cuda.synchronize(device)

    return fill


def _transfer(transport, source_kv, destination_kv, block_sizes, devices):
    model = _model()
    source_plan = cache.cache_plan(model, block_sizes[0], source_kv)
    destination_plan = cache.cache_plan(model, block_sizes[1], destination_kv)
    source_units = _units(source_plan, 1)
    destination_units = _units(destination_plan, 3)
    functions = cache.functions()
    result = _harness()["kv_transfer"](
        {
            "source_plan": source_plan,
            "destination_plan": destination_plan,
            "num_units": 48,
            "tokens": TOKENS,
            "source_device": devices[0],
            "destination_device": devices[1],
            "transport": transport,
            "bind_units": functions["cache.bind_units"],
            "bind_conversion": functions["cache.bind_conversion"],
            "fill": _fill(source_plan, source_units),
            "source_units": source_units,
            "destination_units": destination_units,
        }
    )
    source = _pool(result["source"], result["source_layout"])
    destination = _pool(result["destination"], result["destination_layout"])
    assert int(result["exported_extent"]) == TOKENS
    return (
        (source, source_plan, source_units),
        (destination, destination_plan, destination_units),
        result,
    )


def _decoded(pool, plan, units, name):
    blocks, group = _blocks(pool, plan, units, name)
    start = _history(group)
    key, value = pool.state(name).read(
        blocks, start=start, length=TOKENS - start
    )
    return key.float().cpu(), value.float().cpu()


def _require_devices(count: int) -> None:
    if torch.cuda.device_count() < count:
        pytest.fail(f"this test needs {count} CUDA devices")


@pytest.mark.parametrize(
    ("transport", "kv_dtype", "devices"),
    [
        ("local", "", (0, 0)),
        ("local", "float8_e4m3fn", (0, 0)),
        ("cuda_vmm", "float8_e4m3fn", (0, 1)),
    ],
)
def test_direct_imports_copy_pages_and_mark_them_initialized(
    transport, kv_dtype, devices
):
    _require_devices(max(devices) + 1)
    source, destination, result = _transfer(
        transport, kv_dtype, kv_dtype, (16, 16), devices
    )
    views = {
        str(name): torch.from_dlpack(value)
        for name, value in result["destination"].items()
    }
    # The Rust planes reach torch with the declared layout.
    plan = destination[1]
    assert views["key.values"].dtype == torch.uint8
    assert views["key.values"].shape == (
        int(plan["columns"]),
        48,
        int(plan["plane_bytes"]),
    )
    assert views["key.initialized"].dtype == torch.bool
    assert views["unit_tables"].dtype == torch.int32
    assert int(views["verified_lengths"][1]) == TOKENS

    # Same layouts copy encoded pages and scales unchanged.
    for group in source[1]["groups"]:
        for name in group["layers"]:
            expected = _decoded(*source, str(name))
            actual = _decoded(*destination, str(name))
            assert torch.equal(expected[0], actual[0]), name
            assert torch.equal(expected[1], actual[1]), name


@pytest.mark.parametrize(
    ("transport", "kinds", "devices"),
    [
        ("shm", ("", "float8_e4m3fn"), (0, 0)),
        ("cuda_vmm", ("float8_e4m3fn", "float8_e4m3fn"), (0, 1)),
    ],
)
def test_converted_imports_match_writes_of_the_decoded_source(
    transport, kinds, devices
):
    _require_devices(max(devices) + 1)
    source, destination, result = _transfer(transport, *kinds, (16, 8), devices)
    _, plan, units = destination
    views = {
        str(name): torch.from_dlpack(value)
        for name, value in result["destination"].items()
    }

    # The reference writes the decoded source into a fresh pool of the
    # destination's layout, as the numerical library encodes it.
    fields = {}
    for name, tensor in views.items():
        if name.split(".")[-1] in {"values", "scale", "initialized"}:
            fields[name] = torch.zeros_like(tensor)
            if name.endswith(".scale"):
                fields[name].fill_(1)
    reference = UnitPool.from_plan({**plan, "num_units": 48}, fields)
    device = fields["key.values"].device
    for group in plan["groups"]:
        for name in group["layers"]:
            name = str(name)
            blocks, value_group = _blocks(reference, plan, units, name)
            key, value = _decoded(*source, name)
            reference.state(name).write(
                blocks,
                start=_history(value_group),
                key=key.to(device=device, dtype=torch.bfloat16),
                value=value.to(device=device, dtype=torch.bfloat16),
            )
    torch.cuda.synchronize(device)

    imported = [unit for group in units for unit in group]
    for name, expected in fields.items():
        assert torch.equal(expected[:, imported], views[name][:, imported]), (
            name
        )
