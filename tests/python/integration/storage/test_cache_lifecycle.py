"""A host prefix cache through the lifecycle of a request's pages.

The worker's storage harness (``crates/worker/examples/storage_harness.rs``,
built with ``cargo build -p uniserve-worker --example storage_harness``)
allocates one cache in host memory, binds this package's cache interface to
it and drives it through installs, an export, an abandoned and an adopted
import, a guidance prefix slot and the releases. After each step it reports
whether another owner could reuse a slot's units and hands the test the
cache's live block tables.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from uniserve.models import CausalLM, TransformerDecoder
from uniserve.nn.attention import Attention
from uniserve.prefix_cache import plan as cache
from uniserve.worker import import_path

pytestmark = pytest.mark.integration

TOKENS = 24
BLOCK_SIZE = 16
# Units of the pool, sentinel included.
NUM_UNITS = 16
# The harness's slots: the exporting slot, the importing slot and the
# request's guidance prefix slot.
SOURCE, DESTINATION, PREFIX = 1, 2, 3


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
    def __init__(self) -> None:
        super().__init__()
        # The decoder takes a layer's cache dtype from its first parameter.
        self.scale = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.attention = Attention(
            2, 1, 8, cache_name="backbone.layers.0.attention"
        )


def _model() -> CausalLM:
    backbone = TransformerDecoder(
        nn.Embedding(8, 4), nn.ModuleDict({"0": _Layer()}), nn.Identity()
    )
    return CausalLM(backbone, nn.Identity())


def _units(plan, first: int) -> list[list[int]]:
    """Distinct units for one slot holding ``TOKENS`` tokens per group."""
    groups = []
    for group in plan.planes.groups:
        count = -(-TOKENS // group.page_tokens) * group.units_per_page
        groups.append(list(range(first, first + count)))
        first += count
    return groups


def _count(units: list[list[int]]) -> int:
    return sum(len(group) for group in units)


def test_a_request_s_units_return_once_no_export_or_import_holds_them():
    plan = cache.cache_plan(_model(), BLOCK_SIZE, "")
    source = _units(plan, 1)
    destination = _units(plan, 1 + _count(source))
    prefix = _units(plan, 1 + _count(source) + _count(destination))
    used = _count(source) + _count(destination) + _count(prefix)
    assert 1 + used <= NUM_UNITS

    steps = {}

    def observe(name, views, facts):
        tables = {
            key: torch.from_dlpack(views[key]).clone()
            for key in ("alloced_lens", "verified_lengths", "unit_tables")
        }
        steps[str(name)] = (
            {str(key): bool(value) for key, value in facts.items()},
            tables,
        )

    result = _harness()["cache_lifecycle"](
        {
            "import": import_path,
            "plan": plan,
            "num_units": NUM_UNITS,
            "tokens": TOKENS,
            "units": [source, destination, prefix],
            "observe": observe,
        }
    )
    assert int(result["exported_extent"]) == TOKENS

    def facts(step):
        return steps[step][0]

    def lengths(step, name):
        return steps[step][1][name].tolist()

    # Freshly assigned units carry no access.
    assert facts("installed") == {"source_reusable": True}
    assert lengths("installed", "alloced_lens")[SOURCE] == TOKENS
    # An export holds the units it reads.
    assert facts("exported") == {"source_reusable": False}
    # An import holds the units it resets until it is abandoned and retired;
    # the abandoned import verified nothing.
    assert facts("importing") == {"destination_reusable": False}
    assert facts("abandoned") == {"destination_reusable": True}
    assert lengths("abandoned", "verified_lengths")[DESTINATION] == 0
    # The adopted import's extent is the slot's verified prefix.
    adopted = lengths("adopted", "verified_lengths")
    assert adopted[DESTINATION] == TOKENS
    assert lengths("adopted", "alloced_lens")[DESTINATION] == TOKENS
    # The guidance prefix slot stays while only another request's prefixes
    # are released.
    assert lengths("prefixed", "alloced_lens")[PREFIX] == TOKENS
    # A revoked export whose readers retired returns its units.
    assert facts("unexported") == {"source_reusable": True}

    # Releasing the slots and the request's prefixes clears their rows.
    released = steps["released"][1]
    for slot in (SOURCE, DESTINATION, PREFIX):
        assert released["alloced_lens"][slot] == 0
        assert released["verified_lengths"][slot] == 0
        assert not released["unit_tables"][:, slot].any()
